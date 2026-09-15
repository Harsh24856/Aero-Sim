"""CAN frame catalogue and bus transports shared by the aircraft and the twin.

The digital twin must not share memory with the engine it monitors. The aircraft side
(aircraft_sim.py, or a real ECU) puts classic CAN frames on a bus; the twin side
(can_ingest.py) only ever sees those frames. This module is the contract between them.

FRAME LAYOUT - 11-bit standard IDs, little-endian, one signal per frame
    ID     signal         type    scale    unit    accepted range    kind
    0x101  throttle       uint16  1e-4     0..1    0 .. 1            set-point
    0x102  altitude       int32   0.1      m       -500 .. 12000     set-point
    0x103  airspeed       uint16  0.01     m/s     0 .. 120          set-point
    0x104  aoa            int16   0.01     deg     -20 .. 25         set-point
    0x105  isa_dev_c      int16   0.01     degC    -30 .. 50         set-point
    0x201  rpm            uint16  1        rpm     0 .. 8000         sensor
    0x202  egt            uint16  0.1      C       0 .. 1000         sensor
    0x203  cht            uint16  0.1      C       0 .. 200          sensor
    0x204  oil_pressure   uint16  0.01     psi     0 .. 150          sensor
    0x205  oil_temp       uint16  0.1      C       0 .. 180          sensor
    0x206  vibx           uint16  2e-5     g       0 .. 1            sensor
    0x207  viby           uint16  2e-5     g       0 .. 1            sensor
    0x208  vibz           uint16  2e-5     g       0 .. 1            sensor
    0x209  fuel_flow      uint16  0.01     L/h     0 .. 200          sensor
    0x2FF  heartbeat      uint32  1        seq     0 .. 2^32-1       meta
Sensor ranges are the physics v3 sensor stops. The sender clamps to them (a sensor pinned
at its stop is still a valid reading); the receiver drops anything outside them, so a
corrupted frame can never command the twin or feed the AI.

TRANSPORTS
    udp[://group:port]   UDP multicast on this machine (default 239.255.42.99:42199).
                         Any number of processes publish and listen, like Linux vcan0,
                         with no drivers - for a laptop demo. Payloads are SocketCAN
                         `struct can_frame` bytes, so a real bridge can forward them 1:1.
    <python-can iface>:<channel>   e.g. socketcan:can0, pcan:PCAN_USBBUS1, kvaser:0 -
                         a real bus through python-can (pip install python-can).
"""
import socket
import struct
from dataclasses import dataclass
from typing import Optional

CAN_FRAME = struct.Struct("<IB3x8s")          # SocketCAN struct can_frame, 16 bytes
CAN_SFF_MASK = 0x7FF
DEFAULT_UDP_GROUP = "239.255.42.99"
DEFAULT_UDP_PORT = 42199


@dataclass(frozen=True)
class Signal:
    name: str
    fmt: str        # struct format, little-endian
    scale: float
    lo: float
    hi: float
    kind: str       # "setpoint" | "sensor" | "meta"


SIGNALS = {
    0x101: Signal("throttle", "<H", 1e-4, 0.0, 1.0, "setpoint"),
    0x102: Signal("altitude", "<i", 0.1, -500.0, 12000.0, "setpoint"),
    0x103: Signal("airspeed", "<H", 0.01, 0.0, 120.0, "setpoint"),
    0x104: Signal("aoa", "<h", 0.01, -20.0, 25.0, "setpoint"),
    0x105: Signal("isa_dev_c", "<h", 0.01, -30.0, 50.0, "setpoint"),
    0x201: Signal("rpm", "<H", 1.0, 0.0, 8000.0, "sensor"),
    0x202: Signal("egt", "<H", 0.1, 0.0, 1000.0, "sensor"),
    0x203: Signal("cht", "<H", 0.1, 0.0, 200.0, "sensor"),
    0x204: Signal("oil_pressure", "<H", 0.01, 0.0, 150.0, "sensor"),
    0x205: Signal("oil_temp", "<H", 0.1, 0.0, 180.0, "sensor"),
    0x206: Signal("vibx", "<H", 2e-5, 0.0, 1.0, "sensor"),
    0x207: Signal("viby", "<H", 2e-5, 0.0, 1.0, "sensor"),
    0x208: Signal("vibz", "<H", 2e-5, 0.0, 1.0, "sensor"),
    0x209: Signal("fuel_flow", "<H", 0.01, 0.0, 200.0, "sensor"),
    0x2FF: Signal("heartbeat", "<I", 1.0, 0.0, float(2**32 - 1), "meta"),
}
BY_NAME = {s.name: (can_id, s) for can_id, s in SIGNALS.items()}
SETPOINTS = [s.name for s in SIGNALS.values() if s.kind == "setpoint"]
SENSORS = [s.name for s in SIGNALS.values() if s.kind == "sensor"]


def encode(name: str, value: float) -> tuple[int, bytes]:
    """(arbitration_id, payload). Clamps to the signal range - a reading pinned at a
    sensor stop is still sent as that stop."""
    can_id, sig = BY_NAME[name]
    v = min(sig.hi, max(sig.lo, float(value)))
    return can_id, struct.pack(sig.fmt, int(round(v / sig.scale)))


def decode(can_id: int, data: bytes) -> Optional[tuple[str, float]]:
    """(name, value), or None for an unknown ID, a short frame or an out-of-range value."""
    sig = SIGNALS.get(can_id)
    if sig is None or len(data) < struct.calcsize(sig.fmt):
        return None
    raw, = struct.unpack_from(sig.fmt, data)
    value = raw * sig.scale
    if not (sig.lo - sig.scale <= value <= sig.hi + sig.scale):
        return None
    return sig.name, round(min(sig.hi, max(sig.lo, value)), 6)


def pack_frame(can_id: int, data: bytes) -> bytes:
    if len(data) > 8:
        raise ValueError("classic CAN frames carry at most 8 data bytes")
    return CAN_FRAME.pack(can_id & CAN_SFF_MASK, len(data), data.ljust(8, b"\x00"))


def unpack_frame(buf: bytes) -> Optional[tuple[int, bytes]]:
    if len(buf) != CAN_FRAME.size:
        return None
    can_id, dlc, data = CAN_FRAME.unpack(buf)
    return can_id & CAN_SFF_MASK, data[:min(dlc, 8)]


class UdpBus:
    """Multicast UDP 'virtual CAN': every process joined to the group sees every frame."""

    def __init__(self, group: str = DEFAULT_UDP_GROUP, port: int = DEFAULT_UDP_PORT):
        self.group, self.port = group, port
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.tx.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 0)      # never leaves the host
        self.tx.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
        self.tx.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton("127.0.0.1"))
        self.rx: Optional[socket.socket] = None

    def _open_rx(self):
        rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        rx.bind(("", self.port))
        mreq = struct.pack("4s4s", socket.inet_aton(self.group), socket.inet_aton("127.0.0.1"))
        rx.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        self.rx = rx

    def send(self, can_id: int, data: bytes):
        self.tx.sendto(pack_frame(can_id, data), (self.group, self.port))

    def recv(self, timeout: float = 1.0) -> Optional[tuple[int, bytes]]:
        if self.rx is None:
            self._open_rx()
        self.rx.settimeout(timeout)
        try:
            buf, _ = self.rx.recvfrom(64)
        except socket.timeout:
            return None
        return unpack_frame(buf)

    def close(self):
        for s in (self.tx, self.rx):
            if s is not None:
                s.close()


class PythonCanBus:
    """A real CAN interface through python-can (optional dependency)."""

    def __init__(self, interface: str, channel: str):
        try:
            import can
        except ImportError as e:
            raise SystemExit("python-can is not installed - pip install python-can, or use --bus udp") from e
        self._can = can
        self.bus = can.Bus(interface=interface, channel=channel)

    def send(self, can_id: int, data: bytes):
        self.bus.send(self._can.Message(arbitration_id=can_id, data=data, is_extended_id=False))

    def recv(self, timeout: float = 1.0) -> Optional[tuple[int, bytes]]:
        msg = self.bus.recv(timeout=timeout)
        if msg is None or msg.is_error_frame:
            return None
        return msg.arbitration_id, bytes(msg.data)

    def close(self):
        self.bus.shutdown()


def open_bus(spec: str = "udp"):
    """'udp', 'udp://group:port', or '<python-can interface>:<channel>'."""
    if spec == "udp" or spec.startswith("udp://"):
        group, port = DEFAULT_UDP_GROUP, DEFAULT_UDP_PORT
        if spec.startswith("udp://"):
            host, _, p = spec[len("udp://"):].partition(":")
            group, port = host or group, int(p) if p else port
        return UdpBus(group, port)
    interface, sep, channel = spec.partition(":")
    if not sep:
        raise ValueError(f"bus spec {spec!r}: use 'udp' or '<interface>:<channel>'")
    return PythonCanBus(interface, channel)
