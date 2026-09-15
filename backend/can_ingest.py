"""CAN bus ingest -> the physics backend's /params (plan phase 10, PS section B).

A real UAV does not receive its operating point over HTTP from a web slider; the
autopilot or engine control unit puts it on the CAN bus. This bridge reads those
frames and feeds the SAME /params boundary the dashboard already uses, so the
digital twin can run from live vehicle data without any change to main.py.

FRAME LAYOUT (little-endian, one signal per frame, 11-bit standard IDs)
    ID     signal        type    scale    unit     range accepted
    0x101  throttle      uint16  1e-4     0..1     0 .. 1
    0x102  altitude      int32   0.1      m        -500 .. 12000
    0x103  airspeed      uint16  0.01     m/s      0 .. 120
    0x104  aoa           int16   0.01     deg      -20 .. 25
    0x105  isa_dev_c     int16   0.01     degC     -30 .. 50
Out-of-range values are dropped and counted, never forwarded - a corrupted frame
must not be able to command the twin.

BUS BACKENDS
    --interface socketcan --channel can0|vcan0   Linux SocketCAN (needs python-can)
    --interface virtual                          python-can virtual bus (any OS)
    --loopback                                   no python-can: an in-process bus
                                                 that replays a scripted climb, for
                                                 demos and tests on a laptop
python-can is optional and imported only when a real bus is requested:
    pip install python-can

Usage:
    python can_ingest.py --interface socketcan --channel can0
    python can_ingest.py --loopback --api http://127.0.0.1:8000
    python can_ingest.py --selftest
"""
import argparse
import struct
import sys
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import httpx


@dataclass(frozen=True)
class Signal:
    name: str
    fmt: str        # struct format, little-endian
    scale: float
    lo: float
    hi: float


SIGNALS = {
    0x101: Signal("throttle", "<H", 1e-4, 0.0, 1.0),
    0x102: Signal("altitude", "<i", 0.1, -500.0, 12000.0),
    0x103: Signal("airspeed", "<H", 0.01, 0.0, 120.0),
    0x104: Signal("aoa", "<h", 0.01, -20.0, 25.0),
    0x105: Signal("isa_dev_c", "<h", 0.01, -30.0, 50.0),
}
BY_NAME = {s.name: (can_id, s) for can_id, s in SIGNALS.items()}


def encode(name: str, value: float) -> tuple[int, bytes]:
    """(arbitration_id, payload) for one signal - used by the loopback and tests."""
    can_id, sig = BY_NAME[name]
    return can_id, struct.pack(sig.fmt, int(round(value / sig.scale)))


def decode(can_id: int, data: bytes) -> Optional[tuple[str, float]]:
    """(param, value) or None for an unknown ID, a short frame or an out-of-range value."""
    sig = SIGNALS.get(can_id)
    if sig is None or len(data) < struct.calcsize(sig.fmt):
        return None
    raw, = struct.unpack_from(sig.fmt, data)
    value = raw * sig.scale
    if not (sig.lo <= value <= sig.hi):
        return None
    return sig.name, round(value, 4)


class Bridge:
    """Accumulates decoded signals and posts them to /params at a bounded rate.

    Frames arrive far faster than the twin needs new set-points; posting every frame
    would flood the backend. Changes are batched and flushed every `period` seconds.
    """

    def __init__(self, post: Callable[[dict], None], period: float = 0.2):
        self.post = post
        self.period = period
        self.pending: dict = {}
        self.last_flush = 0.0
        self.stats = {"frames": 0, "decoded": 0, "rejected": 0, "posts": 0}

    def on_frame(self, can_id: int, data: bytes, now: Optional[float] = None):
        self.stats["frames"] += 1
        if can_id not in SIGNALS:
            return                               # other ECUs share the bus - ignore
        decoded = decode(can_id, data)
        if decoded is None:
            self.stats["rejected"] += 1
            return
        self.stats["decoded"] += 1
        name, value = decoded
        self.pending[name] = value
        self.flush(now)

    def flush(self, now: Optional[float] = None, force: bool = False):
        now = time.monotonic() if now is None else now
        if self.pending and (force or now - self.last_flush >= self.period):
            self.post(dict(self.pending))
            self.stats["posts"] += 1
            self.pending.clear()
            self.last_flush = now


def http_poster(api: str) -> Callable[[dict], None]:
    client = httpx.Client(timeout=2.0)

    def post(params: dict):
        try:
            client.post(f"{api}/params", json=params)
        except Exception as e:                  # backend down: keep reading the bus
            print(f"[can] /params failed: {e}")
    return post


def scripted_climb(seconds: float = 20.0, hz: float = 50.0) -> Iterable[tuple[float, int, bytes]]:
    """A takeoff-and-climb profile as CAN frames: (t, id, payload)."""
    n = int(seconds * hz)
    for i in range(n):
        t = i / hz
        frac = min(1.0, t / (seconds * 0.6))
        for name, value in (("throttle", 0.35 + 0.45 * frac), ("altitude", 2500.0 * frac),
                            ("airspeed", 25.0 + 20.0 * frac), ("aoa", 6.0 - 4.0 * frac), ("isa_dev_c", 0.0)):
            can_id, payload = encode(name, value)
            yield t, can_id, payload


def run_bus(interface: str, channel: str, bridge: Bridge):
    try:
        import can                               # optional dependency
    except ImportError:
        sys.exit("python-can is not installed - pip install python-can, or use --loopback")
    with can.Bus(interface=interface, channel=channel) as bus:
        print(f"[can] listening on {interface}:{channel}")
        while True:
            msg = bus.recv(timeout=1.0)
            if msg is not None and not msg.is_error_frame:
                bridge.on_frame(msg.arbitration_id, bytes(msg.data))
            bridge.flush()


def selftest() -> int:
    ok = True
    for name, (_, sig) in BY_NAME.items():
        for v in (sig.lo, (sig.lo + sig.hi) / 2, sig.hi):
            can_id, payload = encode(name, v)
            back = decode(can_id, payload)
            ok &= back is not None and back[0] == name and abs(back[1] - v) <= sig.scale
    ok &= decode(0x101, struct.pack("<H", 20000)) is None      # throttle 2.0 rejected
    ok &= decode(0x7FF, b"\x00\x00") is None                     # unknown id
    ok &= decode(0x102, b"\x01") is None                         # short frame
    posts = []
    bridge = Bridge(posts.append, period=0.2)
    for t, can_id, payload in scripted_climb(seconds=10.0):
        bridge.on_frame(can_id, payload, now=t)
    bridge.on_frame(0x101, struct.pack("<H", 20000), now=10.0)   # corrupted frame
    bridge.flush(force=True)
    last = posts[-1]
    ok &= abs(last["altitude"] - 2500.0) < 1.0 and abs(last["throttle"] - 0.80) < 1e-3
    ok &= 45 <= len(posts) <= 60 and bridge.stats["rejected"] == 1
    print(f"selftest {'PASSED' if ok else 'FAILED'}: {bridge.stats}, {len(posts)} posts, last {last}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--interface", default="socketcan")
    ap.add_argument("--channel", default="can0")
    ap.add_argument("--loopback", action="store_true", help="replay a scripted climb without python-can")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    bridge = Bridge(http_poster(a.api))
    if a.loopback:
        start = time.monotonic()
        for t, can_id, payload in scripted_climb():
            time.sleep(max(0.0, start + t - time.monotonic()))
            bridge.on_frame(can_id, payload)
        bridge.flush(force=True)
        print(f"[can] loopback done: {bridge.stats}")
        return
    run_bus(a.interface, a.channel, bridge)


if __name__ == "__main__":
    main()
