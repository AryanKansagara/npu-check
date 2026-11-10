// TFLite FlatBuffer reader (schema.fbs: Model, OperatorCode, SubGraph, Tensor, Operator, Buffer)
// and a small forward-layout builder used by tests.
import type { Dim, DType, Issue, ModelIR, ModelNode, TensorInfo } from './types.ts'

const TFL_DTYPES: Record<number, DType> = {
  0: 'fp32', 1: 'fp16', 2: 'int32', 3: 'uint8', 4: 'int64', 5: 'string', 6: 'bool', 7: 'int16', 8: 'complex', 9: 'int8',
  10: 'fp64', 11: 'complex', 12: 'uint64', 15: 'uint32', 16: 'uint16', 17: 'int4',
}

// BuiltinOperator enum values we name explicitly, mapped to the ONNX op the support table knows.
const BUILTIN: Record<number, [string, string]> = {
  0: ['ADD', 'Add'], 1: ['AVERAGE_POOL_2D', 'AveragePool'], 2: ['CONCATENATION', 'Concat'], 3: ['CONV_2D', 'Conv'],
  4: ['DEPTHWISE_CONV_2D', 'Conv'], 5: ['DEPTH_TO_SPACE', 'DepthToSpace'], 6: ['DEQUANTIZE', 'DequantizeLinear'],
  7: ['EMBEDDING_LOOKUP', 'Gather'], 8: ['FLOOR', 'Floor'], 9: ['FULLY_CONNECTED', 'Gemm'], 11: ['L2_NORMALIZATION', 'LpNormalization'],
  14: ['LOGISTIC', 'Sigmoid'], 16: ['LSTM', 'LSTM'], 17: ['MAX_POOL_2D', 'MaxPool'], 18: ['MUL', 'Mul'], 19: ['RELU', 'Relu'],
  21: ['RELU6', 'Clip'], 22: ['RESHAPE', 'Reshape'], 23: ['RESIZE_BILINEAR', 'Resize'], 25: ['SOFTMAX', 'Softmax'],
  26: ['SPACE_TO_DEPTH', 'SpaceToDepth'], 28: ['TANH', 'Tanh'], 32: ['CUSTOM', 'Custom'], 34: ['PAD', 'Pad'], 36: ['GATHER', 'Gather'],
  39: ['TRANSPOSE', 'Transpose'], 40: ['MEAN', 'ReduceMean'], 41: ['SUB', 'Sub'], 42: ['DIV', 'Div'], 43: ['SQUEEZE', 'Squeeze'],
  44: ['UNIDIRECTIONAL_SEQUENCE_LSTM', 'LSTM'], 45: ['STRIDED_SLICE', 'Slice'], 47: ['EXP', 'Exp'], 48: ['TOPK_V2', 'TopK'],
  49: ['SPLIT', 'Split'], 50: ['LOG_SOFTMAX', 'LogSoftmax'], 53: ['CAST', 'Cast'], 54: ['PRELU', 'PRelu'], 55: ['MAXIMUM', 'Max'],
  56: ['ARG_MAX', 'ArgMax'], 57: ['MINIMUM', 'Min'], 59: ['NEG', 'Neg'], 61: ['GREATER', 'Greater'], 64: ['SELECT', 'Where'],
  65: ['SLICE', 'Slice'], 66: ['SIN', 'Sin'], 67: ['TRANSPOSE_CONV', 'ConvTranspose'], 69: ['TILE', 'Tile'], 70: ['EXPAND_DIMS', 'Unsqueeze'],
  71: ['EQUAL', 'Equal'], 73: ['LOG', 'Log'], 74: ['SUM', 'ReduceSum'], 75: ['SQRT', 'Sqrt'], 76: ['RSQRT', 'Rsqrt'], 77: ['SHAPE', 'Shape'],
  78: ['POW', 'Pow'], 82: ['REDUCE_MAX', 'ReduceMax'], 83: ['PACK', 'Concat'], 85: ['ONE_HOT', 'OneHot'], 88: ['UNPACK', 'Split'],
  92: ['SQUARE', 'Mul'], 94: ['FILL', 'Expand'], 96: ['RANGE', 'Range'], 97: ['RESIZE_NEAREST_NEIGHBOR', 'Resize'], 98: ['LEAKY_RELU', 'LeakyRelu'],
  99: ['SQUARED_DIFFERENCE', 'Sub'], 100: ['MIRROR_PAD', 'Pad'], 101: ['ABS', 'Abs'], 102: ['SPLIT_V', 'Split'], 103: ['UNIQUE', 'Unique'],
  107: ['GATHER_ND', 'GatherND'], 108: ['COS', 'Cos'], 109: ['WHERE', 'NonZero'], 111: ['ELU', 'Elu'], 114: ['QUANTIZE', 'QuantizeLinear'],
  117: ['HARD_SWISH', 'HardSwish'], 118: ['IF', 'If'], 119: ['WHILE', 'Loop'], 120: ['NON_MAX_SUPPRESSION_V4', 'NonMaxSuppression'],
  121: ['NON_MAX_SUPPRESSION_V5', 'NonMaxSuppression'], 122: ['SCATTER_ND', 'ScatterND'], 123: ['SELECT_V2', 'Where'],
  126: ['BATCH_MATMUL', 'MatMul'], 150: ['GELU', 'Gelu'],
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
    const [tfl, op] = c?.code === 32 ? [`CUSTOM:${c.custom}`, c.custom || 'Custom'] : BUILTIN[c?.code ?? -1] ?? [`BUILTIN_${c?.code}`, `TFL_${c?.code}`]
    const n: ModelNode = { name: `${tfl.toLowerCase()}_${i}`, op, domain: c?.code === 32 ? 'custom' : '', inputs: fb.i32s(o, 1).map(nameOf).filter(Boolean), outputs: fb.i32s(o, 2).map(nameOf), attrs: { tflite: tfl, layout: 'NHWC' } }
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
