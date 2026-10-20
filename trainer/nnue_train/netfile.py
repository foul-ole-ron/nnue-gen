"""Binary network file, shared with the C++ evaluator.

Little-endian layout:
    magic  b"TWNN"
    u32    version, n_inputs, hidden, QA, QB, SCALE, n_moves, n_local, policy_context, policy_hidden, QP, POLICY_SCALE
    i16    w1[n_inputs][hidden]   (feature-major)
    i16    b1[hidden]
    i16    w2[2 * hidden]         (side-to-move half first)
    i16    b2
    i16    wg[2 * hidden][policy_context]
    i32    bg[policy_context]
    i16    m1[n_local + policy_context][policy_hidden]   (local feature rows first, order as in board.py;
                                                          the scaled-feature rows are zero, see quantize.py)
    i32    mb1[policy_hidden]
    i16    m2[policy_hidden]
    i32    mb2
    i16    wpass[2 * hidden]
    i32    bpass
"""

import os
import struct
from pathlib import Path

import numpy as np

from nnue_train.board import NUM_LOCAL, NUM_MOVES
from nnue_train.features import NUM_FEATURES
from nnue_train.model import HIDDEN, POLICY_CONTEXT, POLICY_HIDDEN, POLICY_NAMES
from nnue_train.quantize import POLICY_BIASES, POLICY_SCALE, QA, QB, QP, SCALE, QuantizedNet

MAGIC = b"TWNN"
VERSION = 6
HEADER = struct.Struct("<4s12I")
I16 = np.dtype("<i2")
I32 = np.dtype("<i4")

VALUE_SHAPES = {"w1": (NUM_FEATURES, HIDDEN), "b1": (HIDDEN,), "w2": (2 * HIDDEN,), "b2": (1,)}
POLICY_SHAPES = {
    "wg": (2 * HIDDEN, POLICY_CONTEXT),
    "bg": (POLICY_CONTEXT,),
    "m1": (NUM_LOCAL + POLICY_CONTEXT, POLICY_HIDDEN),
    "mb1": (POLICY_HIDDEN,),
    "m2": (POLICY_HIDDEN,),
    "mb2": (1,),
    "wpass": (2 * HIDDEN,),
    "bpass": (1,),
}
assert tuple(POLICY_SHAPES) == POLICY_NAMES
SHAPES = VALUE_SHAPES | POLICY_SHAPES
DTYPES = {name: I32 if name in POLICY_BIASES else I16 for name in SHAPES}
DIMS = (NUM_FEATURES, HIDDEN, QA, QB, SCALE, NUM_MOVES, NUM_LOCAL, POLICY_CONTEXT, POLICY_HIDDEN, QP, POLICY_SCALE)


def write_net(net: QuantizedNet, path: str | Path) -> None:
    """Write atomically: a reader never sees a half-written file."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(HEADER.pack(MAGIC, VERSION, *DIMS))
        for name, dtype in DTYPES.items():
            f.write(np.ascontiguousarray(getattr(net, name), dtype=dtype).tobytes())
    os.replace(tmp, path)


def read_net(path: str | Path) -> QuantizedNet:
    data = Path(path).read_bytes()
    if len(data) < HEADER.size:
        raise ValueError("file too short")
    magic, version, *dims = HEADER.unpack_from(data)
    if magic != MAGIC or version != VERSION:
        raise ValueError(f"bad magic/version: {magic!r} v{version}")
    if tuple(dims) != DIMS:
        raise ValueError("network dimensions or constants do not match this build")
    if len(data) != HEADER.size + sum(int(np.prod(SHAPES[n])) * DTYPES[n].itemsize for n in SHAPES):
        raise ValueError("unexpected file size")

    arrays, offset = {}, HEADER.size
    for name, shape in SHAPES.items():
        dtype, count = DTYPES[name], int(np.prod(shape))
        arrays[name] = np.frombuffer(data, dtype=dtype, count=count, offset=offset).reshape(shape).astype(
            dtype.newbyteorder("="))
        offset += count * dtype.itemsize
    return QuantizedNet(**arrays)
