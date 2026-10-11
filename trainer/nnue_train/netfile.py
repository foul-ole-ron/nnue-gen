"""Binary network file, shared with the C++ evaluator.

Little-endian layout:
    magic  b"TWNN"
    u32    version, n_inputs, hidden, QA, QB, SCALE, n_moves, n_local, policy_context, policy_hidden
    i16    w1[n_inputs][hidden]   (feature-major)
    i16    b1[hidden]
    i16    w2[2 * hidden]         (side-to-move half first)
    i16    b2
    f32    wg[2 * hidden][policy_context]
    f32    bg[policy_context]
    f32    m1[n_local + policy_context][policy_hidden]   (local feature rows first, order as in board.py)
    f32    mb1[policy_hidden]
    f32    m2[policy_hidden]
    f32    mb2
    f32    wpass[2 * hidden]
    f32    bpass
"""

import os
import struct
from pathlib import Path

import numpy as np

from nnue_train.board import NUM_LOCAL, NUM_MOVES
from nnue_train.features import NUM_FEATURES
from nnue_train.model import HIDDEN, POLICY_CONTEXT, POLICY_HIDDEN, POLICY_NAMES
from nnue_train.quantize import QA, QB, SCALE, QuantizedNet

MAGIC = b"TWNN"
VERSION = 5
HEADER = struct.Struct("<4s10I")
I16 = np.dtype("<i2")
F32 = np.dtype("<f4")

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


def write_net(net: QuantizedNet, path: str | Path) -> None:
    """Write atomically: a reader never sees a half-written file."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(HEADER.pack(MAGIC, VERSION, NUM_FEATURES, HIDDEN, QA, QB, SCALE,
                            NUM_MOVES, NUM_LOCAL, POLICY_CONTEXT, POLICY_HIDDEN))
        for name in VALUE_SHAPES:
            f.write(np.ascontiguousarray(getattr(net, name), dtype=I16).tobytes())
        for name in POLICY_SHAPES:
            f.write(np.ascontiguousarray(getattr(net, name), dtype=F32).tobytes())
    os.replace(tmp, path)


def read_net(path: str | Path) -> QuantizedNet:
    data = Path(path).read_bytes()
    if len(data) < HEADER.size:
        raise ValueError("file too short")
    magic, version, *dims = HEADER.unpack_from(data)
    if magic != MAGIC or version != VERSION:
        raise ValueError(f"bad magic/version: {magic!r} v{version}")
    if tuple(dims) != (NUM_FEATURES, HIDDEN, QA, QB, SCALE, NUM_MOVES, NUM_LOCAL, POLICY_CONTEXT, POLICY_HIDDEN):
        raise ValueError("network dimensions or constants do not match this build")
    n_i16 = sum(int(np.prod(s)) for s in VALUE_SHAPES.values())
    n_f32 = sum(int(np.prod(s)) for s in POLICY_SHAPES.values())
    if len(data) != HEADER.size + 2 * n_i16 + 4 * n_f32:
        raise ValueError("unexpected file size")

    arrays, offset = {}, HEADER.size
    for shapes, dtype in ((VALUE_SHAPES, I16), (POLICY_SHAPES, F32)):
        for name, shape in shapes.items():
            count = int(np.prod(shape))
            arrays[name] = np.frombuffer(data, dtype=dtype, count=count, offset=offset).reshape(shape).astype(
                dtype.newbyteorder("="))
            offset += count * dtype.itemsize
    return QuantizedNet(**arrays)
