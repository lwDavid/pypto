# Packed FP4 types in PyPTO

PyPTO exposes two distinct packed FP4 data types. This page describes their
shared type and byte-carrier ABI, numerical casts, and current operator support.
The short type table is in [Types](../user/language/00-types.md).

## Type and unit contract

| DSL type | Uppercase alias | Storage | PTO type |
| -------- | --------------- | ------- | -------- |
| `pl.fp4e2m1x2` | `pl.FP4E2M1X2` | Two E2M1 values per byte | `!pto.f4E2M1x2` |
| `pl.fp4e1m2x2` | `pl.FP4E1M2X2` | Two E1M2 values per byte | `!pto.f4E1M2x2` |

Both types have `get_bit() == 8` and `get_byte() == 1`. Tensor and tile shapes,
`valid_shape`, strides, and offsets count byte carriers. A `[2,256]` tensor
occupies 512 bytes and contains 1,024 FP4 values. A carrier extent or offset
need not be even; each carrier already contains a complete pair of values.
Operator-specific alignment and layout restrictions still apply.

There is no unpacked FP4 type or nibble-counting ABI. The old `pl.FP4`,
`DataType.FP4`, and `DT_FP4` names are removed. JIT specialization, function
arguments and returns, runtime allocation, and persisted argument metadata
preserve carrier geometry without doubling or halving a dimension.

The two formats are distinct even when their shapes and byte sizes match.
Changing a function annotation does not convert data: dependency calls,
including void calls and submits, reject E2M1/E1M2 format mismatches.
Use an explicit numerical `pl.cast` to convert between the formats.

## Torch and runtime carriers

- `torch.float4_e2m1fn_x2` maps directly to `FP4E2M1X2`, with its physical shape
  unchanged. An E1M2 annotation rejects this E2M1-specific Torch dtype.
- An explicit annotation for either packed type accepts `torch.uint8` carriers.
  Unannotated UINT8 data remains UINT8; the annotation supplies the FP4 format.
- E2M1 return allocation uses Torch's native x2 dtype when available, otherwise
  UINT8. E1M2 uses UINT8 because Torch has no corresponding native dtype.
- Compiled and reloaded argument metadata keeps the numerical format and the
  same carrier shape. Packed runtime tensors require rank at least one and a
  positive final carrier extent.

Raw carrier storage is not a host numerical implementation. `torch_codegen`
rejects packed FP4 references instead of treating their bytes as FP32 values.
Use a reference that explicitly decodes the selected format when checking
numerical results.

## Operator support on Ascend950

Both formats support ND numerical casts. Additional transfer layouts, DN
casts, and MXFP4 quantization are deferred.

| Operation | Support |
| --------- | ------- |
| Packed GM views | ND carrier dimensions and strides; PTOAS v0.67 handles DMA descriptor conversion |
| Either packed format ↔ UINT8/INT8 alias | Same-shape, dense addressed 2D row-major Vec storage; PYPTO or DSA_RP planning |
| Either packed format ↔ BF16 cast | Native TCVT on row-major, unboxed Vec; the last axis expands/contracts 2:1 |
| Either packed format → FP8E4M3FN/FP8E5M2 cast | Legalized through BF16 and FP32, with a warning; a host reference or lookup table remains an alternative |
| E2M1↔E1M2 conversion | Numerical cast through BF16; carrier shape is unchanged |
| Other wider-type casts | Existing legalization with lossless intermediates; see the encoding constraint below |
| DN, NZ, column-major, column-vector `[M,1]`, or layout-changing packed views | Additional transfer layouts and DN casts are deferred |
| Direct packed `reshape` or `transpose` | Unsupported; reshape a byte alias where the existing alias contract permits it |
| Native packed FP4 `matmul_mx` | Unsupported by the pinned PTOAS geometry contract; cast ND operands to FP8 first |
| MXFP4 `quant_mx` | Deferred; `quant_mx` remains MXFP8-only |
| Ascend910/A2A3 packed FP4 execution | Unsupported |

PTO GM views retain carrier coordinates. The conversion to the lower-level
instruction's value geometry belongs to PTOAS; scaling the GM shape again in
PyPTO would double its row pitch.

## Numerical cast constraints

BF16 encoding supports `rint`, `round`, `floor`, `ceil`, and `trunc`; `odd` is
rejected. Packing requires complete pairs: a positive even source extent or a
provably even dynamic `valid_shape`, such as `2 * n`. Decoding expands the
carrier extent without requiring it to be even.

All packed casts require row-major `none_box` storage. Decoding through BF16 preserves every FP4 value exactly. Implicit encoding chains must also preserve
the original source through BF16: FP16/FP32 and larger integer sources are
rejected because that intermediate can double-round. Write an explicit BF16
cast first when that intermediate rounding is intended. The pinned simulator
misrounds fractional FP4 conversions; numerical acceptance uses A5 hardware.

## Decode a UINT8 carrier through a UB alias (Ascend950)

Keep the GM tensor in UINT8 and alias its UB storage with the encoded format:

```python
import pypto.language as pl


@pl.jit(auto_scope=False)
def decode_bytes(
    packed: pl.Tensor[[32, 256], pl.UINT8],
    output: pl.Out[pl.Tensor[[32, 512], pl.BF16]],
):
    with pl.spmd(1):
        row = pl.tile.get_block_idx() * 32
        payload = pl.tile.load(packed, [row, 0], [32, 256])
        fp4 = pl.reinterpret_view(payload, pl.fp4e2m1x2)
        decoded = pl.cast(fp4, pl.BF16)
        pl.tile.store(decoded, [row, 0], output)
    return output
```

The alias keeps the `[32,256]` carrier shape and the same 8,192-byte allocation.
Only the numerical cast creates a fresh BF16 `[32,512]` tile. Use
`pl.fp4e1m2x2` instead when the source bytes encode E1M2.

The alias requires `slayout=none_box`, dense storage, zero tile `start_offset`,
and an address-backed MemRef with a static UB byte offset. Its source byte span
must equal the carrier element count. Noncontiguous UB slices, including
intermediate aliases or reshapes of them, and addressless inputs are rejected.
GM load offsets may be nonzero or dynamic; they are distinct from UB offsets.
An explicit alias shape must match the source. Reshape a byte view separately
before aliasing back. InCore tensor aliases lower to the same tile path;
orchestration tensor aliases remain unsupported.

PYPTO and DSA_RP planning emit addressed aliases. PTOAS planning remains
unsupported for this path because its `treshape` rejects the packed FP4 type.
Aliasing preserves bits; it does not numerically convert one FP4 format into
the other.

## See also

- [Types](../user/language/00-types.md) — public aliases and carrier ABI
- [Operators / MX](ir/05-operators.md) — existing MXFP8 matrix paths
