"""CAN bus ingest -> the physics backend (PS section B, plan P3).

A real UAV does not receive its operating point over HTTP from a web slider, and the
twin must not read the engine's internal state. This bridge is the only link: it reads
CAN frames (catalogue in can_bus.py) and feeds

    set-point frames (0x10x)  ->  POST /params     what the autopilot commands
    sensor frames    (0x20x)  ->  POST /measured   what the engine's sensors report

main.py drives its own physics from the set-points and scores the AI and the physics
residuals on the measured sensors. Heartbeat gaps (0x2FF) are counted as lost frames.
Out-of-range and unknown frames are dropped and counted, never forwarded.

Usage:
    python can_ingest.py --bus udp                        # laptop virtual bus (aircraft_sim.py)
    python can_ingest.py --bus socketcan:can0             # real bus via python-can
    python can_ingest.py --loopback --api http://127.0.0.1:8000   # scripted climb, no bus
    python can_ingest.py --selftest
"""
import argparse
import struct
import sys
import time
from typing import Callable, Iterable, Optional

import httpx

import can_bus
from can_bus import SIGNALS, decode, encode


class Bridge:
    """Accumulates decoded signals and posts them at a bounded rate.

    Frames arrive far faster than the twin needs them; posting every frame would flood
    the backend. Changes are batched and flushed every `period` seconds.
    """

    def __init__(self, post_params: Callable[[dict], None],
                 post_measured: Optional[Callable[[dict], None]] = None, period: float = 0.2):
        self.post_params = post_params
        self.post_measured = post_measured
        self.period = period
        self.pending_params: dict = {}
        self.pending_measured: dict = {}
        self.last_flush = 0.0
        self.last_seq: Optional[int] = None
        self.stats = {"frames": 0, "decoded": 0, "rejected": 0, "posts": 0, "measured_posts": 0, "lost": 0}

    def on_frame(self, can_id: int, data: bytes, now: Optional[float] = None):
        self.stats["frames"] += 1
        sig = SIGNALS.get(can_id)
        if sig is None:
            return                               # other ECUs share the bus - ignore
        decoded = decode(can_id, data)
        if decoded is None:
            self.stats["rejected"] += 1
            return
        self.stats["decoded"] += 1
        name, value = decoded
        if sig.kind == "setpoint":
            self.pending_params[name] = value
        elif sig.kind == "sensor":
            self.pending_measured[name] = value
        else:
            seq = int(value)
            if self.last_seq is not None and seq > self.last_seq + 1:
                self.stats["lost"] += seq - self.last_seq - 1
            self.last_seq = seq
        self.flush(now)

    def flush(self, now: Optional[float] = None, force: bool = False):
        now = time.monotonic() if now is None else now
        if not (force or now - self.last_flush >= self.period):
            return
        if self.pending_params:
            self.post_params(dict(self.pending_params))
            self.stats["posts"] += 1
            self.pending_params.clear()
        if self.pending_measured and self.post_measured is not None:
            self.post_measured(dict(self.pending_measured))
            self.stats["measured_posts"] += 1
            self.pending_measured.clear()
        self.last_flush = now


def http_poster(api: str, path: str, extra: Optional[dict] = None) -> Callable[[dict], None]:
    client = httpx.Client(timeout=2.0)

    def post(body: dict):
        try:
            client.post(f"{api}{path}", json={**body, **(extra or {})})
        except Exception as e:                  # backend down: keep reading the bus
            print(f"[can] {path} failed: {e}")
    return post


def scripted_climb(seconds: float = 20.0, hz: float = 50.0) -> Iterable[tuple[float, int, bytes]]:
    """A takeoff-and-climb profile as set-point frames: (t, id, payload)."""
    n = int(seconds * hz)
    for i in range(n):
        t = i / hz
        frac = min(1.0, t / (seconds * 0.6))
        for name, value in (("throttle", 0.35 + 0.45 * frac), ("altitude", 2500.0 * frac),
                            ("airspeed", 25.0 + 20.0 * frac), ("aoa", 6.0 - 4.0 * frac), ("isa_dev_c", 0.0)):
            can_id, payload = encode(name, value)
            yield t, can_id, payload


def run_bus(bus, bridge: Bridge, report_every: float = 10.0):
    print(f"[can] listening on {type(bus).__name__}")
    last_report = time.monotonic()
    try:
        while True:
            frame = bus.recv(timeout=1.0)
            if frame is not None:
                bridge.on_frame(*frame)
            bridge.flush()
            if time.monotonic() - last_report >= report_every:
                print(f"[can] {bridge.stats}", flush=True)
                last_report = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        bus.close()


def selftest() -> int:
    ok = True
    for name, (_, sig) in can_bus.BY_NAME.items():
        for v in (sig.lo, (sig.lo + sig.hi) / 2, sig.hi):
            can_id, payload = encode(name, v)
            back = decode(can_id, payload)
            ok &= back is not None and back[0] == name and abs(back[1] - v) <= sig.scale
            frame = can_bus.unpack_frame(can_bus.pack_frame(can_id, payload))
            ok &= frame == (can_id, payload)
    ok &= decode(0x101, struct.pack("<H", 20000)) is None      # throttle 2.0 rejected
    ok &= decode(0x7FF, b"\x00\x00") is None                     # unknown id
    ok &= decode(0x102, b"\x01") is None                         # short frame
    ok &= encode("cht", 260.0) == encode("cht", 200.0)           # a pinned sensor stays at its stop

    params, measured = [], []
    bridge = Bridge(params.append, measured.append, period=0.2)
    for t, can_id, payload in scripted_climb(seconds=10.0):
        bridge.on_frame(can_id, payload, now=t)
    for i, t in enumerate((10.0, 10.1, 10.3)):
        bridge.on_frame(*encode("egt", 700.0 + i), now=t)
        bridge.on_frame(*encode("heartbeat", [0, 1, 5][i]), now=t)          # frames 2-4 lost
    bridge.on_frame(0x101, struct.pack("<H", 20000), now=10.4)              # corrupted frame
    bridge.flush(force=True)
    last = params[-1]
    ok &= abs(last["altitude"] - 2500.0) < 1.0 and abs(last["throttle"] - 0.80) < 1e-3
    ok &= 45 <= len(params) <= 60 and bridge.stats["rejected"] == 1 and bridge.stats["lost"] == 3
    ok &= bool(measured) and abs(measured[-1]["egt"] - 702.0) < 0.2

    udp = "skipped"
    try:
        bus = can_bus.UdpBus(port=can_bus.DEFAULT_UDP_PORT + 1)
        bus.recv(timeout=0.01)                                   # join before sending
        bus.send(*encode("oil_pressure", 55.5))
        got = bus.recv(timeout=1.0)
        bus.close()
        udp = "ok" if got is not None and decode(*got) == ("oil_pressure", 55.5) else "FAILED"
        ok &= udp == "ok"
    except OSError as e:
        udp = f"unavailable ({e})"
    print(f"selftest {'PASSED' if ok else 'FAILED'}: {bridge.stats}, {len(params)} param posts, "
          f"{len(measured)} measured posts, udp bus {udp}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--bus", default="udp", help="udp, udp://group:port, or <python-can interface>:<channel>")
    ap.add_argument("--interface", help="legacy: python-can interface (use --bus interface:channel)")
    ap.add_argument("--channel", default="can0", help="legacy: python-can channel")
    ap.add_argument("--loopback", action="store_true", help="replay a scripted climb without any bus")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    # Tagged so main.py lets the bus, not the cockpit sliders, own the set-points.
    bridge = Bridge(http_poster(a.api, "/params", {"source": "can"}), http_poster(a.api, "/measured"))
    if a.loopback:
        start = time.monotonic()
        for t, can_id, payload in scripted_climb():
            time.sleep(max(0.0, start + t - time.monotonic()))
            bridge.on_frame(can_id, payload)
        bridge.flush(force=True)
        print(f"[can] loopback done: {bridge.stats}")
        return
    spec = f"{a.interface}:{a.channel}" if a.interface else a.bus
    run_bus(can_bus.open_bus(spec), bridge)


if __name__ == "__main__":
    main()
