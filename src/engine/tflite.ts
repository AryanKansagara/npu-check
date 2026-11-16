// TFLite FlatBuffer reader (schema.fbs: Model, OperatorCode, SubGraph, Tensor, Operator, Buffer)
// and a small forward-layout builder used by tests.
import type { Dim, DType, Issue, ModelIR, ModelNode, TensorInfo } from './types.ts'

const TFL_DTYPES: Record<number, DType> = {
  0: 'fp32', 1: 'fp16', 2: 'int32', 3: 'uint8', 4: 'int64', 5: 'string', 6: 'bool', 7: 'int16', 8: 'complex', 9: 'int8',
  10: 'fp64', 11: 'complex', 12: 'uint64', 15: 'uint32', 16: 'uint16', 17: 'int4', 18: 'bf16', 20: 'uint4', 21: 'fp8', 22: 'fp8',
}

// BuiltinOperator names indexed by enum value (tensorflow/compiler/mlir/lite/schema/schema.fbs, checked 2026-11-15).
export const TFL_NAMES = `
ADD AVERAGE_POOL_2D CONCATENATION CONV_2D DEPTHWISE_CONV_2D DEPTH_TO_SPACE DEQUANTIZE EMBEDDING_LOOKUP FLOOR FULLY_CONNECTED HASHTABLE_LOOKUP
L2_NORMALIZATION L2_POOL_2D LOCAL_RESPONSE_NORMALIZATION LOGISTIC LSH_PROJECTION LSTM MAX_POOL_2D MUL RELU RELU_N1_TO_1 RELU6 RESHAPE
RESIZE_BILINEAR RNN SOFTMAX SPACE_TO_DEPTH SVDF TANH CONCAT_EMBEDDINGS SKIP_GRAM CALL CUSTOM EMBEDDING_LOOKUP_SPARSE PAD UNIDIRECTIONAL_SEQUENCE_RNN
GATHER BATCH_TO_SPACE_ND SPACE_TO_BATCH_ND TRANSPOSE MEAN SUB DIV SQUEEZE UNIDIRECTIONAL_SEQUENCE_LSTM STRIDED_SLICE BIDIRECTIONAL_SEQUENCE_RNN
EXP TOPK_V2 SPLIT LOG_SOFTMAX DELEGATE BIDIRECTIONAL_SEQUENCE_LSTM CAST PRELU MAXIMUM ARG_MAX MINIMUM LESS NEG PADV2 GREATER GREATER_EQUAL
LESS_EQUAL SELECT SLICE SIN TRANSPOSE_CONV SPARSE_TO_DENSE TILE EXPAND_DIMS EQUAL NOT_EQUAL LOG SUM SQRT RSQRT SHAPE POW ARG_MIN FAKE_QUANT
REDUCE_PROD REDUCE_MAX PACK LOGICAL_OR ONE_HOT LOGICAL_AND LOGICAL_NOT UNPACK REDUCE_MIN FLOOR_DIV REDUCE_ANY SQUARE ZEROS_LIKE FILL
FLOOR_MOD RANGE RESIZE_NEAREST_NEIGHBOR LEAKY_RELU SQUARED_DIFFERENCE MIRROR_PAD ABS SPLIT_V UNIQUE CEIL REVERSE_V2 ADD_N GATHER_ND
COS WHERE RANK ELU REVERSE_SEQUENCE MATRIX_DIAG QUANTIZE MATRIX_SET_DIAG ROUND HARD_SWISH IF WHILE NON_MAX_SUPPRESSION_V4 NON_MAX_SUPPRESSION_V5
SCATTER_ND SELECT_V2 DENSIFY SEGMENT_SUM BATCH_MATMUL PLACEHOLDER_FOR_GREATER_OP_CODES CUMSUM CALL_ONCE BROADCAST_TO RFFT2D CONV_3D
_ _ _ HASHTABLE HASHTABLE_FIND HASHTABLE_IMPORT HASHTABLE_SIZE REDUCE_ALL CONV_3D_TRANSPOSE VAR_HANDLE READ_VARIABLE ASSIGN_VARIABLE
BROADCAST_ARGS RANDOM_STANDARD_NORMAL BUCKETIZE RANDOM_UNIFORM MULTINOMIAL GELU DYNAMIC_UPDATE_SLICE RELU_0_TO_1 UNSORTED_SEGMENT_PROD
UNSORTED_SEGMENT_MAX UNSORTED_SEGMENT_SUM ATAN2 UNSORTED_SEGMENT_MIN SIGN BITCAST BITWISE_XOR RIGHT_SHIFT STABLEHLO_LOGISTIC STABLEHLO_ADD
STABLEHLO_DIVIDE STABLEHLO_MULTIPLY STABLEHLO_MAXIMUM STABLEHLO_RESHAPE STABLEHLO_CLAMP STABLEHLO_CONCATENATE STABLEHLO_BROADCAST_IN_DIM
STABLEHLO_CONVOLUTION STABLEHLO_SLICE STABLEHLO_CUSTOM_CALL STABLEHLO_REDUCE STABLEHLO_ABS STABLEHLO_AND STABLEHLO_COSINE STABLEHLO_EXPONENTIAL
STABLEHLO_FLOOR STABLEHLO_LOG STABLEHLO_MINIMUM STABLEHLO_NEGATE STABLEHLO_OR STABLEHLO_POWER STABLEHLO_REMAINDER STABLEHLO_RSQRT STABLEHLO_SELECT
STABLEHLO_SUBTRACT STABLEHLO_TANH STABLEHLO_SCATTER STABLEHLO_COMPARE STABLEHLO_CONVERT STABLEHLO_DYNAMIC_SLICE STABLEHLO_DYNAMIC_UPDATE_SLICE
STABLEHLO_PAD STABLEHLO_IOTA STABLEHLO_DOT_GENERAL STABLEHLO_REDUCE_WINDOW STABLEHLO_SORT STABLEHLO_WHILE STABLEHLO_GATHER STABLEHLO_TRANSPOSE
DILATE STABLEHLO_RNG_BIT_GENERATOR REDUCE_WINDOW STABLEHLO_COMPOSITE STABLEHLO_SHIFT_LEFT STABLEHLO_CBRT STABLEHLO_CASE
`.trim().split(/\s+/)

// TFLite builtins mapped to the ONNX op the support table knows. Anything else keeps its TFLite name.
const TFL_TO_ONNX: Record<string, string> = {
  ADD: 'Add', AVERAGE_POOL_2D: 'AveragePool', CONCATENATION: 'Concat', CONV_2D: 'Conv', DEPTHWISE_CONV_2D: 'Conv', DEPTH_TO_SPACE: 'DepthToSpace',
  DEQUANTIZE: 'DequantizeLinear', EMBEDDING_LOOKUP: 'Gather', FLOOR: 'Floor', FULLY_CONNECTED: 'Gemm', L2_NORMALIZATION: 'LpNormalization',
  LOGISTIC: 'Sigmoid', LSTM: 'LSTM', MAX_POOL_2D: 'MaxPool', MUL: 'Mul', RELU: 'Relu', RELU6: 'Clip', RESHAPE: 'Reshape', RESIZE_BILINEAR: 'Resize',
  SOFTMAX: 'Softmax', SPACE_TO_DEPTH: 'SpaceToDepth', TANH: 'Tanh', PAD: 'Pad', GATHER: 'Gather', TRANSPOSE: 'Transpose', MEAN: 'ReduceMean',
  SUB: 'Sub', DIV: 'Div', SQUEEZE: 'Squeeze', UNIDIRECTIONAL_SEQUENCE_LSTM: 'LSTM', STRIDED_SLICE: 'Slice', EXP: 'Exp', TOPK_V2: 'TopK',
  SPLIT: 'Split', LOG_SOFTMAX: 'LogSoftmax', CAST: 'Cast', PRELU: 'PRelu', MAXIMUM: 'Max', ARG_MAX: 'ArgMax', MINIMUM: 'Min', NEG: 'Neg',
  GREATER: 'Greater', SELECT: 'Where', SLICE: 'Slice', SIN: 'Sin', TRANSPOSE_CONV: 'ConvTranspose', TILE: 'Tile', EXPAND_DIMS: 'Unsqueeze',
  EQUAL: 'Equal', LOG: 'Log', SUM: 'ReduceSum', SQRT: 'Sqrt', RSQRT: 'Rsqrt', SHAPE: 'Shape', POW: 'Pow', REDUCE_MAX: 'ReduceMax', PACK: 'Concat',
  ONE_HOT: 'OneHot', UNPACK: 'Split', SQUARE: 'Mul', FILL: 'Expand', RANGE: 'Range', RESIZE_NEAREST_NEIGHBOR: 'Resize', LEAKY_RELU: 'LeakyRelu',
  SQUARED_DIFFERENCE: 'Sub', MIRROR_PAD: 'Pad', ABS: 'Abs', SPLIT_V: 'Split', UNIQUE: 'Unique', GATHER_ND: 'GatherND', COS: 'Cos', WHERE: 'NonZero',
  ELU: 'Elu', QUANTIZE: 'QuantizeLinear', HARD_SWISH: 'HardSwish', IF: 'If', WHILE: 'Loop', NON_MAX_SUPPRESSION_V4: 'NonMaxSuppression',
  NON_MAX_SUPPRESSION_V5: 'NonMaxSuppression', SCATTER_ND: 'ScatterND', SELECT_V2: 'Where', BATCH_MATMUL: 'MatMul', GELU: 'Gelu', LOCAL_RESPONSE_NORMALIZATION: 'LRN',
  RELU_N1_TO_1: 'Clip', RELU_0_TO_1: 'Clip', BIDIRECTIONAL_SEQUENCE_LSTM: 'LSTM', LESS: 'Less', PADV2: 'Pad', GREATER_EQUAL: 'GreaterOrEqual',
  LESS_EQUAL: 'LessOrEqual', ARG_MIN: 'ArgMin', REDUCE_PROD: 'ReduceProd', LOGICAL_OR: 'Or', LOGICAL_AND: 'And', LOGICAL_NOT: 'Not', REDUCE_MIN: 'ReduceMin',
  FLOOR_MOD: 'Mod', CEIL: 'Ceil', ADD_N: 'Sum', ROUND: 'Round', CUMSUM: 'CumSum', BROADCAST_TO: 'Expand', SIGN: 'Sign', CONV_3D: 'Conv',
  FLOOR_DIV: 'Div', ZEROS_LIKE: 'ConstantOfShape',
}

class FB {
  buf: Uint8Array
  dv: DataView
  constructor(buf: Uint8Array) { this.buf = buf; this.dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength) }
  u32(p: number) { if (p + 4 > this.buf.length) throw new Error(`FlatBuffer offset ${p} is past the end of the file`); return this.dv.getUint32(p, true) }
  root() { return this.u32(0) }
  /** Absolute position of field `i` in the table at `t`, or 0 when absent. */
  field(t: number, i: number) {
    const vt = t - this.dv.getInt32(t, true)
    const vtSize = this.dv.getUint16(vt, true)
    const o = 4 + i * 2
    if (o >= vtSize) return 0
    const off = this.dv.getUint16(vt + o, true)
    return off ? t + off : 0
  }
  scalar(t: number, i: number, kind: 'u8' | 'i32' | 'u32', def = 0) {
    const p = this.field(t, i)
    if (!p) return def
    return kind === 'u8' ? this.dv.getUint8(p) : kind === 'i32' ? this.dv.getInt32(p, true) : this.dv.getUint32(p, true)
  }
  ref(t: number, i: number) { const p = this.field(t, i); return p ? p + this.u32(p) : 0 }
  vec(t: number, i: number): { pos: number; len: number } | null {
    const v = this.ref(t, i)
    return v ? { pos: v + 4, len: this.u32(v) } : null
  }
  str(t: number, i: number) {
    const v = this.vec(t, i)
    return v ? new TextDecoder().decode(this.buf.subarray(v.pos, v.pos + v.len)) : ''
  }
  i32s(t: number, i: number): number[] {
    const v = this.vec(t, i)
    if (!v) return []
    const out: number[] = []
    for (let k = 0; k < v.len; k++) out.push(this.dv.getInt32(v.pos + k * 4, true))
    return out
  }
  tables(t: number, i: number): number[] {
    const v = this.vec(t, i)
    if (!v) return []
    const out: number[] = []
    for (let k = 0; k < v.len; k++) { const p = v.pos + k * 4; out.push(p + this.u32(p)) }
    return out
  }
}

export const isTflite = (b: Uint8Array) => b.length > 8 && String.fromCharCode(b[4], b[5], b[6], b[7]) === 'TFL3'

export function parseTflite(buf: Uint8Array): ModelIR {
  const fb = new FB(buf)
  const issues: Issue[] = []
  const m = fb.root()
  const version = fb.scalar(m, 0, 'u32')
  const codes = fb.tables(m, 1).map(c => {
    const dep = fb.scalar(c, 0, 'u8')
    const code = Math.max(dep, fb.scalar(c, 3, 'i32'))
    const custom = fb.str(c, 1)
    return { code, custom }
  })
  const buffers = fb.tables(m, 4).map(b => {
    const d = fb.vec(b, 0)
    const sizeP = fb.field(b, 2)
    return d ? d.len : sizeP ? Number(fb.dv.getBigUint64(sizeP, true)) : 0
  })
  const subgraphs = fb.tables(m, 2)
  if (!subgraphs.length) throw new Error('TFLite model has no subgraphs.')
  if (subgraphs.length > 1) issues.push({ level: 'info', where: 'model', message: `${subgraphs.length} subgraphs (control flow or signatures); only the main subgraph is analysed.` })
  const sg = subgraphs[0]
  const tensors: Record<string, TensorInfo> = {}
  const names: string[] = []
  fb.tables(sg, 0).forEach((t, i) => {
    const name = fb.str(t, 3) || `tensor_${i}`
    const shape: Dim[] = fb.i32s(t, 0)
    const sig = fb.i32s(t, 7)
    sig.forEach((d, k) => { if (d < 0) shape[k] = k === 0 ? 'batch' : `dim${k}` })
    const bufIdx = fb.scalar(t, 2, 'u32')
    const info: TensorInfo = { name, dtype: TFL_DTYPES[fb.scalar(t, 1, 'u8')] ?? 'unknown', shape }
    if (bufIdx > 0 && buffers[bufIdx] > 0) { info.init = true; info.bytes = buffers[bufIdx] }
    const q = fb.ref(t, 4)
    if (q) {
      const scales = fb.vec(q, 2)?.len ?? 0
      if (scales) info.quant = { scales, axis: fb.scalar(q, 6, 'i32') }
    }
    tensors[name] = info
    names.push(name)
  })
  const nameOf = (i: number) => i < 0 ? '' : names[i] ?? `tensor_${i}`
  const nodes: ModelNode[] = fb.tables(sg, 3).map((o, i) => {
    const c = codes[fb.scalar(o, 0, 'u32')]
    if (!c) issues.push({ level: 'error', where: `operator ${i}`, message: 'opcode_index points past the operator_codes table.' })
    const builtin = TFL_NAMES[c?.code ?? -1] && TFL_NAMES[c?.code ?? -1] !== '_' ? TFL_NAMES[c!.code] : `BUILTIN_${c?.code}`
    let [tfl, op] = c?.code === 32 ? [`CUSTOM:${c.custom}`, c.custom || 'Custom'] : [builtin, TFL_TO_ONNX[builtin] ?? builtin]
    const inputs = fb.i32s(o, 1).map(nameOf).filter(Boolean)
    // DEQUANTIZE on fp16 constants is how fp16-weight models expand weights to fp32; it is a cast, not QDQ.
    if (tfl === 'DEQUANTIZE' && tensors[inputs[0]]?.dtype === 'fp16') op = 'Cast'
    const n: ModelNode = { name: `${tfl.toLowerCase()}_${i}`, op, domain: c?.code === 32 ? 'custom' : '', inputs, outputs: fb.i32s(o, 2).map(nameOf), attrs: { tflite: tfl, layout: 'NHWC' } }
    if (tfl === 'DEPTHWISE_CONV_2D') n.attrs.depthwise = 1
    if (tfl === 'RELU6') { n.attrs.min = 0; n.attrs.max = 6 }
    return n
  })
  return {
    format: 'tflite', producer: `TFLite schema v${version}`, opset: [], graphName: fb.str(sg, 4) || 'main', fileSize: buf.length,
    inputs: fb.i32s(sg, 1).map(nameOf), outputs: fb.i32s(sg, 2).map(nameOf), tensors, nodes, issues, meta: { description: fb.str(m, 3) },
  }
}

// ---------- Builder (tests) ----------

export type FbVal =
  | { u8: number } | { i32: number } | { u32: number } | { str: string } | { vi32: number[] } | { vf32: number[] }
  | { vu8: Uint8Array } | { tbl: FbTable } | { vtbl: FbTable[] }
export type FbTable = (FbVal | undefined)[]

/** Lays out tables front to back so every uoffset points forward. Not size-optimal; correct. */
export function buildFlatbuffer(root: FbTable, ident = 'TFL3'): Uint8Array {
  const bytes: number[] = []
  const pad4 = () => { while (bytes.length % 4) bytes.push(0) }
  const u32 = (v: number) => { bytes.push(v & 255, (v >>> 8) & 255, (v >>> 16) & 255, (v >>> 24) & 255) }
  const patch = (at: number, target: number) => { const v = target - at; bytes[at] = v & 255; bytes[at + 1] = (v >>> 8) & 255; bytes[at + 2] = (v >>> 16) & 255; bytes[at + 3] = (v >>> 24) & 255 }
  const writeVecOrStr = (v: FbVal): number => {
    pad4()
    const at = bytes.length
    if ('str' in v) { const b = new TextEncoder().encode(v.str); u32(b.length); bytes.push(...b, 0) }
    else if ('vi32' in v) { u32(v.vi32.length); v.vi32.forEach(x => u32(x >>> 0)) }
    else if ('vf32' in v) { u32(v.vf32.length); const f = new Float32Array(v.vf32); new Uint8Array(f.buffer).forEach(b => bytes.push(b)) }
    else if ('vu8' in v) { u32(v.vu8.length); bytes.push(...v.vu8) }
    else if ('vtbl' in v) {
      u32(v.vtbl.length)
      const slots = v.vtbl.map(() => { const p = bytes.length; u32(0); return p })
      v.vtbl.forEach((t, k) => patch(slots[k], writeTable(t)))
    }
    return at
  }
  const writeTable = (t: FbTable): number => {
    pad4()
    // vtable: size, table size, offsets (each field 4 bytes inline)
    const n = t.length
    const vtSize = 4 + n * 2
    const vtAt = bytes.length
    const offs = t.map((f, i) => f ? 4 + 4 * t.slice(0, i).filter(Boolean).length : 0)
    const tblSize = 4 + 4 * t.filter(Boolean).length
    bytes.push(vtSize & 255, vtSize >> 8, tblSize & 255, tblSize >> 8)
    offs.forEach(o => bytes.push(o & 255, o >> 8))
    pad4()
    const at = bytes.length
    const so = at - vtAt
    u32(so)
    const later: [number, FbVal][] = []
    for (const f of t) {
      if (!f) continue
      if ('u8' in f) u32(f.u8)
      else if ('i32' in f) u32(f.i32 >>> 0)
      else if ('u32' in f) u32(f.u32)
      else { later.push([bytes.length, f]); u32(0) }
    }
    for (const [slot, f] of later) patch(slot, 'tbl' in f ? writeTable(f.tbl) : writeVecOrStr(f))
    return at
  }
  u32(0)
  const id = new TextEncoder().encode(ident)
  bytes.push(...id)
  const r = writeTable(root)
  bytes[0] = r & 255; bytes[1] = (r >>> 8) & 255; bytes[2] = (r >>> 16) & 255; bytes[3] = (r >>> 24) & 255
  return Uint8Array.from(bytes)
}
