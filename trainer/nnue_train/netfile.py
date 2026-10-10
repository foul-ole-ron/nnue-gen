"""Binary network file, shared with the C++ evaluator.

Little-endian layout:
    magic  b"TWNN"
    u32    version, n_inputs, hidden, QA, QB, SCALE
    i16    w1[n_inputs][hidden]   (feature-major)
    i16    b1[hidden]
    i16    w2[2 * hidden]         (side-to-move half first)
    i16    b2
"""

import os
import struct
from pathlib import Path

import numpy as np

from nnue_train.features import NUM_FEATURES
from nnue_train.model import HIDDEN
from nnue_train.quantize import QA, QB, SCALE, QuantizedNet

MAGIC = b"TWNN"
VERSION = 2
HEADER = struct.Struct("<4s6I")
I16 = np.dtype("<i2")


def write_net(net: QuantizedNet, path: str | Path) -> None:
    """Write atomically: a reader never sees a half-written file."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(HEADER.pack(MAGIC, VERSION, NUM_FEATURES, HIDDEN, QA, QB, SCALE))
        for arr in (net.w1, net.b1, net.w2, net.b2):
            f.write(np.ascontiguousarray(arr, dtype=I16).tobytes())
    os.replace(tmp, path)


def read_net(path: str | Path) -> QuantizedNet:
    data = Path(path).read_bytes()
    if len(data) < HEADER.size:
        raise ValueError("file too short")
    magic, version, n_inputs, hidden, qa, qb, scale = HEADER.unpack_from(data)
    if magic != MAGIC or version != VERSION:
        raise ValueError(f"bad magic/version: {magic!r} v{version}")
    if (n_inputs, hidden, qa, qb, scale) != (NUM_FEATURES, HIDDEN, QA, QB, SCALE):
        raise ValueError("network dimensions or constants do not match this build")
    sizes = [n_inputs * hidden, hidden, 2 * hidden, 1]
    if len(data) != HEADER.size + 2 * sum(sizes):
        raise ValueError("unexpected file size")
    values = np.frombuffer(data, dtype=I16, offset=HEADER.size).astype(np.int16)
    w1, b1, w2, b2 = np.split(values, np.cumsum(sizes)[:-1])
    return QuantizedNet(w1=w1.reshape(n_inputs, hidden), b1=b1, w2=w2, b2=b2)
