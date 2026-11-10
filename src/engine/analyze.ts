// Readiness analysis: shapes, op inventory, NPU assignment, partitions, MACs, activation memory,
// quantization readiness, roofline latency and fix recommendations.
import { rule, type Support } from './support.ts'
import { DTYPE_BYTES, type Dim, type ModelIR, type ModelNode, type TensorInfo } from './types.ts'

// ---------- Device profiles ----------

export interface DeviceProfile {
  id: string
  name: string
  /** Dense int8 NPU throughput, TOPS. fp16 is assumed to run at half this rate. */
  npuTops: number
  /** DRAM bandwidth shared by NPU and CPU, GB/s */
  memGBs: number
  /** Sustained fp32 CPU throughput for fallback ops, GFLOPS */
  cpuGflops: number
  /** On-chip NPU scratch memory, MB; activations above this spill to DRAM */
  vtcmMB: number
  /** Fixed cost of each CPU<->NPU hand-off, ms */
  switchMs: number
  /** Fraction of peak a real graph achieves */
  util: number
  note: string
}

export const DEVICES: DeviceProfile[] = [
  { id: 'laptop-45', name: 'Laptop NPU, 45 TOPS class', npuTops: 45, memGBs: 135, cpuGflops: 600, vtcmMB: 8, switchMs: 0.15, util: 0.35, note: 'Uses the publicly quoted 45 TOPS and 135 GB/s figures for Snapdragon X Elite class laptops. CPU, VTCM, hand-off and utilisation are assumptions.' },
  { id: 'phone-flagship', name: 'Flagship phone (illustrative)', npuTops: 40, memGBs: 77, cpuGflops: 300, vtcmMB: 8, switchMs: 0.2, util: 0.3, note: 'Illustrative flagship phone with LPDDR5X. Not a published spec for any specific SoC; edit to match your device.' },
  { id: 'phone-mid', name: 'Mid-range phone (illustrative)', npuTops: 12, memGBs: 51, cpuGflops: 90, vtcmMB: 4, switchMs: 0.3, util: 0.3, note: 'Illustrative mid-tier SoC with LPDDR5. Edit to match your device.' },
]

export type FixId = 'pin-dims' | 'replace-ops' | 'quantize-int8' | 'split-post'
export const FIX_LABELS: Record<FixId, string> = {
  'pin-dims': 'Fix dynamic shapes',
  'replace-ops': 'Replace unsupported ops',
  'quantize-int8': 'Quantize to int8 (QDQ)',
  'split-post': 'Move post-processing to the CPU',
}

export interface Options {
  /** Values for symbolic dims, used for estimates and when the pin-dims fix is applied */
  pins: Record<string, number>
  applied: FixId[]
  device: DeviceProfile
}

// ---------- Helpers ----------

export const numel = (s: Dim[] | null | undefined, pins: Record<string, number> = {}) =>
  !s ? 0 : s.reduce<number>((a, d) => a * (typeof d === 'number' ? Math.max(d, 0) : pins[d] ?? 1), 1)

const ints = (v: unknown): number[] | undefined => Array.isArray(v) && v.every(x => typeof x === 'number') ? v as number[] : undefined

function bcast(a: Dim[], b: Dim[]): Dim[] {
  const n = Math.max(a.length, b.length)
  const out: Dim[] = []
  for (let i = 0; i < n; i++) {
    const x = a[a.length - n + i] ?? 1, y = b[b.length - n + i] ?? 1
    out.push(x === 1 ? y : y === 1 ? x : typeof x === 'number' ? x : y)
  }
  return out
}

/** Symbols that look like a batch dimension default to 1, sequence-like to 128, image-like to 224. */
export function suggestPin(sym: string): number {
  if (/batch|^n$|^b$|^bs$|unk__\d+|^\?$/i.test(sym)) return 1
  if (/seq|len|token|sequence|time|^t$|^s$/i.test(sym)) return 128
  if (/height|width|^h$|^w$|img|size/i.test(sym)) return 224
  return 1
}

export function symbols(ir: ModelIR): string[] {
  const s = new Set<string>()
  for (const n of ir.inputs) for (const d of ir.tensors[n]?.shape ?? []) if (typeof d === 'string' && d !== '?') s.add(d)
  for (const n of ir.inputs) if ((ir.tensors[n]?.shape ?? []).includes('?')) s.add('?')
  return [...s]
}

/** Kahn's algorithm, keeping file order among ready nodes. Cycles (invalid) are appended in file order. */
export function topoOrder(ir: ModelIR): number[] {
  const producer = new Map<string, number>()
  ir.nodes.forEach((n, i) => n.outputs.forEach(o => o && producer.set(o, i)))
  const indeg = ir.nodes.map(() => 0)
  const users: number[][] = ir.nodes.map(() => [])
  ir.nodes.forEach((n, i) => {
    const deps = new Set(n.inputs.map(x => producer.get(x)).filter((p): p is number => p !== undefined && p !== i))
    indeg[i] = deps.size
    deps.forEach(p => users[p].push(i))
  })
  const ready = indeg.map((d, i) => d === 0 ? i : -1).filter(i => i >= 0)
  const out: number[] = []
  const done = new Set<number>()
  while (ready.length) {
    ready.sort((a, b) => a - b)
    const i = ready.shift()!
    out.push(i); done.add(i)
    for (const u of users[i]) if (--indeg[u] === 0) ready.push(u)
  }
  if (out.length < ir.nodes.length) ir.nodes.forEach((_, i) => { if (!done.has(i)) out.push(i) })
  return out
}

// ---------- Shape inference ----------

/**
 * Shapes for every tensor. With `pins`, symbolic input dims are replaced by numbers first.
 * Uses declared value_info when present and infers the common ops otherwise; unknown stays null.
 */
export function inferShapes(ir: ModelIR, pins?: Record<string, number>, order = topoOrder(ir)): Record<string, Dim[] | null> {
  const sub = (s: Dim[] | null): Dim[] | null => !s ? null : pins ? s.map(d => typeof d === 'string' && d !== '?' ? pins[d] ?? suggestPin(d) : d) : s
  const S: Record<string, Dim[] | null> = {}
  for (const t of Object.values(ir.tensors)) S[t.name] = sub(t.shape)
  if (ir.format !== 'onnx') return S
  const val = (n: string) => ir.tensors[n]?.values
  for (const i of order) {
    const n = ir.nodes[i]
    if (n.outputs.every(o => !o || ir.tensors[o]?.shape)) continue
    const ins = n.inputs.map(x => x ? S[x] ?? null : null)
    const out = inferNode(n, ins, val, S)
    n.outputs.forEach((o, k) => { if (o && !(ir.tensors[o]?.shape)) S[o] = out[k] ?? null })
  }
  return S
}

const UNARY = new Set(['Relu', 'Sigmoid', 'Tanh', 'Erf', 'Sqrt', 'Exp', 'Log', 'Clip', 'LeakyRelu', 'Gelu', 'HardSwish', 'HardSigmoid', 'Neg', 'Abs', 'Cast', 'Softmax', 'LogSoftmax', 'BatchNormalization', 'LayerNormalization', 'InstanceNormalization', 'LpNormalization', 'Dropout', 'Identity', 'QuantizeLinear', 'DequantizeLinear', 'Sign', 'Not', 'Floor', 'Ceil', 'Round', 'Elu', 'PRelu', 'Sin', 'Cos', 'Reciprocal', 'CumSum', 'ScatterND', 'ScatterElements', 'FastGelu', 'QuickGelu', 'SkipLayerNormalization'])
const BINARY = new Set(['Add', 'Sub', 'Mul', 'Div', 'Pow', 'Max', 'Min', 'Equal', 'Greater', 'GreaterOrEqual', 'Less', 'LessOrEqual', 'And', 'Or', 'Where', 'Mod', 'Sum'])

function inferNode(n: ModelNode, ins: (Dim[] | null)[], val: (n: string) => number[] | undefined, S: Record<string, Dim[] | null>): (Dim[] | null)[] {
  const a = ins[0], A = n.attrs
  const v = (k: number) => n.inputs[k] ? val(n.inputs[k]) : undefined
  if (UNARY.has(n.op)) return [a]
  if (BINARY.has(n.op)) return [ins.reduce<Dim[] | null>((acc, s) => acc && s ? bcast(acc, s) : null, a ?? null)]
  if (!a && n.op !== 'Constant') return []
  switch (n.op) {
    case 'Conv': case 'MaxPool': case 'AveragePool': case 'QLinearConv': case 'ConvTranspose': {
      const w = n.op === 'QLinearConv' ? ins[3] : ins[1]
      const sp = a!.length - 2
      const k = ints(A.kernel_shape) ?? (w ? w.slice(2) as number[] : Array(sp).fill(1))
      const st = ints(A.strides) ?? Array(sp).fill(1), dil = ints(A.dilations) ?? Array(sp).fill(1)
      const pads = ints(A.pads) ?? Array(sp * 2).fill(0)
      const ap = typeof A.auto_pad === 'string' ? A.auto_pad : 'NOTSET'
      const cOut: Dim = n.op === 'MaxPool' || n.op === 'AveragePool' ? a![1] : n.op === 'ConvTranspose' ? (w ? (w[1] as number) * (typeof A.group === 'number' ? A.group : 1) : '?') : w ? w[0] : '?'
      const spatial = a!.slice(2).map((x, i) => {
        const kk = (dil[i] ?? 1) * ((k[i] as number) - 1) + 1
        if (typeof x !== 'number') return st[i] === 1 && (ap.startsWith('SAME') || pads[i] + pads[i + sp] === kk - 1) ? x : '?'
        if (n.op === 'ConvTranspose') return (st[i] ?? 1) * (x - 1) + kk - pads[i] - pads[i + sp]
        if (ap.startsWith('SAME')) return Math.ceil(x / (st[i] ?? 1))
        const ceil = A.ceil_mode === 1 ? Math.ceil : Math.floor
        return ceil((x + pads[i] + pads[i + sp] - kk) / (st[i] ?? 1)) + 1
      })
      return [[a![0], cOut, ...spatial]]
    }
    case 'GlobalAveragePool': case 'GlobalMaxPool': return [[a![0], a![1], ...a!.slice(2).map(() => 1)]]
    case 'Flatten': {
      const ax = typeof A.axis === 'number' ? (A.axis < 0 ? A.axis + a!.length : A.axis) : 1
      const prod = (s: Dim[]) => s.every(d => typeof d === 'number') ? (s as number[]).reduce((x, y) => x * y, 1) : s.length === 1 ? s[0] : '?'
      return [[prod(a!.slice(0, ax)), prod(a!.slice(ax))]]
    }
    case 'Gemm': case 'QLinearMatMul': case 'MatMul': case 'MatMulInteger': case 'FusedMatMul': {
      const b = n.op === 'QLinearMatMul' ? ins[3] : ins[1]
      if (!b) return []
      if (n.op === 'Gemm') return [[A.transA === 1 ? a![1] : a![0], A.transB === 1 ? b[0] : b[1]]]
      if (b.length === 1) return [a!.slice(0, -1)]
      if (a!.length === 1) return [[...b.slice(0, -2), b[b.length - 1]]]
      const batch = bcast(a!.slice(0, -2), b.slice(0, -2))
      return [[...batch, a![a!.length - 2], b[b.length - 1]]]
    }
    case 'Reshape': {
      const shp = v(1)
      if (!shp) return [null]
      const out: Dim[] = shp.map((d, i) => d === 0 && A.allowzero !== 1 ? a![i] : d)
      const neg = out.indexOf(-1)
      if (neg >= 0) {
        const known = out.filter((_, i) => i !== neg)
        if (a!.every(d => typeof d === 'number') && known.every(d => typeof d === 'number')) {
          out[neg] = numel(a) / Math.max(1, (known as number[]).reduce((x, y) => x * y, 1))
        } else {
          // Symbolic: cancel matching symbols, e.g. [b, seq, 4, 32] -> [b, seq, 128].
          const rest = a!.filter(d => !known.includes(d))
          const statics = rest.filter(d => typeof d === 'number') as number[]
          const syms = rest.filter(d => typeof d !== 'number')
          const knownStatic = known.filter(d => typeof d === 'number' && !a!.includes(d)) as number[]
          const ratio = statics.reduce((x, y) => x * y, 1) / Math.max(1, knownStatic.reduce((x, y) => x * y, 1))
          out[neg] = syms.length === 1 && ratio === 1 ? syms[0] : syms.length === 0 && Number.isInteger(ratio) ? ratio : '?'
        }
      }
      return [out]
    }
    case 'Transpose': {
      const perm = ints(A.perm) ?? a!.map((_, i) => a!.length - 1 - i)
      return [perm.map(p => a![p])]
    }
    case 'Concat': {
      const ax = (A.axis as number ?? 0) < 0 ? (A.axis as number) + a!.length : A.axis as number ?? 0
      if (ins.some(s => !s)) return [null]
      const sum = ins.map(s => s![ax])
      const out = [...a!]
      out[ax] = sum.every(d => typeof d === 'number') ? (sum as number[]).reduce((x, y) => x + y, 0) : '?'
      return [out]
    }
    case 'Unsqueeze': {
      const axes = ints(A.axes) ?? v(1) ?? []
      const r = a!.length + axes.length
      const norm = axes.map(x => x < 0 ? x + r : x)
      const out: Dim[] = []
      let j = 0
      for (let i = 0; i < r; i++) out.push(norm.includes(i) ? 1 : a![j++])
      return [out]
    }
    case 'Squeeze': {
      const axes = (ints(A.axes) ?? v(1))?.map(x => x < 0 ? x + a!.length : x)
      return [a!.filter((d, i) => axes ? !axes.includes(i) : d !== 1)]
    }
    case 'Split': {
      const ax = ((A.axis as number) ?? 0) < 0 ? (A.axis as number) + a!.length : (A.axis as number) ?? 0
      const parts = ints(A.split) ?? v(1)
      return n.outputs.map((_, k) => {
        const o = [...a!]
        const d = a![ax]
        o[ax] = parts ? parts[k] : typeof d === 'number' ? d / n.outputs.length : '?'
        return o
      })
    }
    case 'Gather': {
      const ax = ((A.axis as number) ?? 0) < 0 ? (A.axis as number) + a!.length : (A.axis as number) ?? 0
      const idx = ins[1] ?? []
      return [[...a!.slice(0, ax), ...idx, ...a!.slice(ax + 1)]]
    }
    case 'Slice': {
      const starts = v(1), ends = v(2), axes = v(3), steps = v(4)
      if (!starts || !ends) return [a!.map(() => '?')]
      const out = [...a!]
      starts.forEach((s, k) => {
        const ax0 = axes ? axes[k] : k
        const ax = ax0 < 0 ? ax0 + a!.length : ax0
        const d = a![ax], st = steps?.[k] ?? 1
        if (typeof d !== 'number') { out[ax] = '?'; return }
        const clamp = (x: number) => Math.min(Math.max(x < 0 ? x + d : x, 0), d)
        out[ax] = Math.max(0, Math.ceil((clamp(ends[k]) - clamp(s)) / st))
      })
      return [out]
    }
    case 'ReduceMean': case 'ReduceSum': case 'ReduceMax': case 'ReduceMin': case 'ReduceProd': case 'ArgMax': case 'ArgMin': {
      const ax = (ints(A.axes) ?? (typeof A.axis === 'number' ? [A.axis] : v(1)) ?? a!.map((_, i) => i)).map(x => x < 0 ? x + a!.length : x)
      const keep = A.keepdims !== 0
      return [keep ? a!.map((d, i) => ax.includes(i) ? 1 : d) : a!.filter((_, i) => !ax.includes(i))]
    }
    case 'Shape': return [[a!.length]]
    case 'ConstantOfShape': { const s = v(0); return [s ?? null] }
    case 'Constant': return [S[n.outputs[0]] ?? null]
    case 'TopK': {
      const ax = ((A.axis as number) ?? -1) < 0 ? ((A.axis as number) ?? -1) + a!.length : A.axis as number
      const k = v(1)?.[0]
      const o = [...a!]; o[ax] = k ?? '?'
      return [o, o]
    }
    case 'NonZero': return [[a!.length, '?']]
    case 'NonMaxSuppression': return [['?', 3]]
    case 'Unique': return [['?']]
    case 'Range': return [['?']]
    case 'Expand': { const s = v(1); return [s ? bcast(a!, s) : null] }
    case 'Tile': { const r = v(1); return [r ? a!.map((d, i) => typeof d === 'number' ? d * r[i] : '?') : null] }
    case 'Pad': {
      const p = ints(A.pads) ?? v(1)
      return [p ? a!.map((d, i) => typeof d === 'number' ? d + p[i] + p[i + a!.length] : '?') : null]
    }
    case 'DepthToSpace': case 'SpaceToDepth': {
      const b = (A.blocksize as number) ?? 1
      const up = n.op === 'DepthToSpace'
      const f = (d: Dim, m: number) => typeof d === 'number' ? d * m : '?'
      return [[a![0], f(a![1], up ? 1 / (b * b) : b * b), f(a![2], up ? b : 1 / b), f(a![3], up ? b : 1 / b)]]
    }
    case 'Resize': {
      const sizes = v(3)
      return [sizes ?? a!.map((d, i) => i < 2 ? d : '?')]
    }
    case 'Einsum': {
      const eq = String(A.equation ?? '').replace(/\s/g, '')
      const [lhs, rhs] = eq.split('->')
      if (!rhs) return [null]
      const dims: Record<string, Dim> = {}
      lhs.split(',').forEach((term, k) => [...term].forEach((c, i) => { const d = ins[k]?.[i]; if (d !== undefined && (dims[c] === undefined || dims[c] === 1)) dims[c] = d }))
      return [[...rhs].map(c => dims[c] ?? '?')]
    }
  }
  return []
}

// ---------- Analysis ----------

export type NodeStatus = 'npu' | 'cpu' | 'removed'
export interface NodeResult {
  index: number
  name: string
  op: string
  status: NodeStatus
  support: Support
  reason: string
  macs: number
  flops: number
  outShape: Dim[] | null
  ms: number
}

export interface OpRow { op: string; domain: string; count: number; support: Support; npu: number; cpu: number; macs: number; note: string; replace?: string }
export type Bucket = 'fp32' | 'fp16' | 'int8' | 'int4' | 'other'
export interface Finding {
  id: string
  severity: 'blocker' | 'warn' | 'info'
  title: string
  detail: string
  nodes?: string[]
  fix?: FixId
}
export interface IoRow { name: string; dtype: string; shape: Dim[] | null; dynamic: string[] }

export interface Analysis {
  format: ModelIR['format']
  nodeCount: number
  opRows: OpRow[]
  nodes: NodeResult[]
  inputs: IoRow[]
  outputs: IoRow[]
  opset: number | null
  params: number
  weightBytes: number
  externalBytes: number
  dtypes: Record<Bucket, { tensors: number; params: number; bytes: number }>
  macs: number
  flops: number
  peakBytes: number
  peakAt: string
  largest: { name: string; bytes: number } | null
  npuNodes: number
  cpuNodes: number
  npuPartitions: number
  cpuPartitions: number
  partitions: number
  switches: number
  quant: { qdq: number; qlinear: number; perChannel: boolean; intWeightShare: number; verdict: 'quantized-per-channel' | 'quantized-per-tensor' | 'weight-only' | 'float'; precision: 'int8' | 'fp16' }
  latency: { npuMs: number; cpuMs: number; switchMs: number; totalMs: number; computeShare: number; tokensPerSec?: number; prefillMs?: number; kvCacheBytes?: number }
  findings: Finding[]
  available: FixId[]
  score: number
  symbols: string[]
}

const bucketOf = (dtype: string): Bucket => {
  if (dtype === 'fp32' || dtype === 'fp64' || dtype === 'ggml:F32' || dtype === 'ggml:F64') return 'fp32'
  if (dtype === 'fp16' || dtype === 'bf16' || dtype === 'ggml:F16' || dtype === 'ggml:BF16') return 'fp16'
  if (dtype === 'int8' || dtype === 'uint8' || /^ggml:(Q8|I8|Q6|Q5)/.test(dtype)) return 'int8'
  if (dtype === 'int4' || dtype === 'uint4' || /^ggml:(Q4|Q3|Q2|IQ|TQ)/.test(dtype)) return 'int4'
  return 'other'
}

export const tensorBytes = (t: TensorInfo, shape: Dim[] | null, pins: Record<string, number> = {}) =>
  t.bytes ?? numel(shape, pins) * (DTYPE_BYTES[t.dtype] ?? 4)

const SPLIT_SHAPES = new Set(['NonZero', 'NonMaxSuppression', 'Unique'])

export function analyze(ir: ModelIR, opts: Options): Analysis {
  const { device, applied } = opts
  const pinned = applied.includes('pin-dims')
  const quantFix = applied.includes('quantize-int8')
  const syms = symbols(ir)
  const pins: Record<string, number> = Object.fromEntries(syms.map(s => [s, opts.pins[s] ?? suggestPin(s)]))
  const order = topoOrder(ir)
  const symShapes = inferShapes(ir, undefined, order)
  const numShapes = inferShapes(ir, pins, order)
  const checkShapes = pinned ? numShapes : symShapes

  // Weights
  const dtypes: Analysis['dtypes'] = { fp32: { tensors: 0, params: 0, bytes: 0 }, fp16: { tensors: 0, params: 0, bytes: 0 }, int8: { tensors: 0, params: 0, bytes: 0 }, int4: { tensors: 0, params: 0, bytes: 0 }, other: { tensors: 0, params: 0, bytes: 0 } }
  let params = 0, weightBytes = 0, externalBytes = 0
  for (const t of Object.values(ir.tensors)) {
    if (!t.init) continue
    const p = numel(t.shape)
    const b = tensorBytes(t, t.shape)
    const k = bucketOf(t.dtype)
    dtypes[k].tensors++; dtypes[k].params += p; dtypes[k].bytes += b
    params += p; weightBytes += b
    if (t.external) externalBytes += b
  }
  const floatParams = dtypes.fp32.params + dtypes.fp16.params
  const intParams = dtypes.int8.params + dtypes.int4.params

  // Quantization
  const qdq = ir.nodes.filter(n => n.op === 'QuantizeLinear' || n.op === 'DequantizeLinear').length
  const qlinear = ir.nodes.filter(n => /^QLinear|Integer$/.test(n.op)).length
  const perChannel = ir.nodes.some(n => n.op === 'DequantizeLinear' && numel(ir.tensors[n.inputs[1]]?.shape) > 1) ||
    Object.values(ir.tensors).some(t => (t.quant?.scales ?? 0) > 1)
  const tflQuant = Object.values(ir.tensors).some(t => t.quant && (t.dtype === 'int8' || t.dtype === 'uint8') && !t.init)
  const intWeightShare = params ? intParams / params : 0
  const quantized = qdq > 0 || qlinear > 0 || tflQuant
  const verdict: Analysis['quant']['verdict'] = quantized ? (perChannel ? 'quantized-per-channel' : 'quantized-per-tensor') : intWeightShare > 0.5 ? 'weight-only' : 'float'
  const precision: 'int8' | 'fp16' = quantized || quantFix ? 'int8' : 'fp16'
  const actScale = quantFix && !quantized ? 0.25 : 1
  const wScale = (t: TensorInfo) => quantFix && !quantized && (t.dtype === 'fp32') ? 0.25 : quantFix && !quantized && t.dtype === 'fp16' ? 0.5 : 1

  // Consumers / producers
  const producer = new Map<string, number>()
  ir.nodes.forEach((n, i) => n.outputs.forEach(o => o && producer.set(o, i)))
  const consumers = new Map<string, number[]>()
  ir.nodes.forEach((n, i) => n.inputs.forEach(x => { if (x) { const l = consumers.get(x) ?? []; l.push(i); consumers.set(x, l) } }))

  // Post-processing tail: the first post op and everything downstream of it.
  const tail = new Set<number>()
  if (applied.includes('split-post')) {
    const stack = ir.nodes.map((n, i) => rule(n.op, n.domain).post ? i : -1).filter(i => i >= 0)
    while (stack.length) {
      const i = stack.pop()!
      if (tail.has(i)) continue
      tail.add(i)
      for (const o of ir.nodes[i].outputs) for (const c of consumers.get(o) ?? []) stack.push(c)
    }
  }

  const isDyn = (s: Dim[] | null | undefined) => !!s && s.some(d => typeof d === 'string')
  const nodes: NodeResult[] = ir.nodes.map((n, i) => {
    const r = rule(n.op, n.domain)
    const outShape = numShapes[n.outputs[0]] ?? null
    const { macs, flops } = cost(n, numShapes, ir)
    const base = { index: i, name: n.name, op: n.op, support: r.status, macs, flops, outShape: checkShapes[n.outputs[0]] ?? outShape, ms: 0 }
    if (tail.has(i)) return { ...base, status: 'removed' as const, reason: 'Moved to CPU post-processing outside the NPU graph.' }
    if (n.op === 'Constant') return { ...base, status: 'npu' as const, reason: 'Constant, folded.' }
    if (r.status === 'cpu') {
      if (r.replace && applied.includes('replace-ops') && !SPLIT_SHAPES.has(n.op)) return { ...base, status: 'npu' as const, reason: `Rewritten: ${r.replace}` }
      return { ...base, status: 'cpu' as const, reason: r.note }
    }
    const dyn = n.outputs.find(o => isDyn(checkShapes[o]))
    if (dyn) {
      const sym = (checkShapes[dyn] ?? []).filter(d => typeof d === 'string')
      const dataDep = sym.includes('?')
      return { ...base, status: 'cpu' as const, reason: dataDep ? `Output "${dyn}" has a data-dependent shape.` : `Dynamic shape (${[...new Set(sym)].join(', ')}): NPU graphs are compiled for fixed shapes.` }
    }
    if (r.status === 'constrained') {
      const bad = constraintViolation(n, numShapes, ir)
      if (bad) return { ...base, status: 'cpu' as const, reason: bad }
    }
    return { ...base, status: 'npu' as const, reason: r.note }
  })

  // Partitions: connected components of same-status nodes; switches: tensors crossing the boundary.
  const active = nodes.filter(n => n.status !== 'removed')
  const parent = ir.nodes.map((_, i) => i)
  const find = (x: number): number => parent[x] === x ? x : (parent[x] = find(parent[x]))
  const switchTensors = new Set<string>()
  ir.nodes.forEach((n, i) => {
    if (nodes[i].status === 'removed') return
    for (const x of n.inputs) {
      const p = producer.get(x)
      if (p === undefined || nodes[p].status === 'removed' || ir.nodes[p].op === 'Constant') continue
      if (nodes[p].status === nodes[i].status) parent[find(p)] = find(i)
      else switchTensors.add(x)
    }
  })
  const comps = (s: NodeStatus) => new Set(active.filter(n => n.status === s && n.op !== 'Constant').map(n => find(n.index))).size
  const npuPartitions = comps('npu'), cpuPartitions = comps('cpu')
  const switches = switchTensors.size + (tail.size ? 1 : 0)

  // Peak activation memory by liveness over the topological order.
  const isAct = (name: string) => !!name && !ir.tensors[name]?.init
  const lastUse = new Map<string, number>()
  order.forEach((ni, step) => ir.nodes[ni].inputs.forEach(x => isAct(x) && lastUse.set(x, step)))
  const outputsSet = new Set(ir.outputs)
  const actBytes = (name: string) => numel(numShapes[name], pins) * (DTYPE_BYTES[ir.tensors[name]?.dtype ?? 'fp32'] ?? 4) * actScale
  let live = 0, peakBytes = 0, peakAt = ''
  const liveSet = new Set<string>()
  for (const x of ir.inputs) { liveSet.add(x); live += actBytes(x) }
  let largest: Analysis['largest'] = null
  order.forEach((ni, step) => {
    const n = ir.nodes[ni]
    for (const o of n.outputs) {
      if (!isAct(o) || n.op === 'Constant') continue
      const b = actBytes(o)
      liveSet.add(o); live += b
      if (!largest || b > largest.bytes) largest = { name: o, bytes: b }
    }
    if (live > peakBytes) { peakBytes = live; peakAt = n.name }
    for (const x of [...n.inputs, ...n.outputs]) {
      if (liveSet.has(x) && (lastUse.get(x) ?? -1) <= step && !outputsSet.has(x)) { liveSet.delete(x); live -= actBytes(x) }
    }
  })

  // Roofline latency per node.
  const npuRate = device.npuTops * 1e12 * device.util * (precision === 'int8' ? 1 : 0.5)
  const bw = device.memGBs * 1e9
  let npuMs = 0, cpuMs = 0, computeMs = 0
  nodes.forEach((r, i) => {
    const n = ir.nodes[i]
    const bytes = n.inputs.reduce((s, x) => s + (ir.tensors[x]?.init ? tensorBytes(ir.tensors[x], ir.tensors[x].shape) * wScale(ir.tensors[x]) : actBytes(x)), 0) + n.outputs.reduce((s, o) => s + actBytes(o), 0)
    const onNpu = r.status === 'npu'
    const compute = r.flops / (onNpu ? npuRate : device.cpuGflops * 1e9) * 1000
    const mem = bytes / bw * 1000
    r.ms = Math.max(compute, mem)
    computeMs += compute >= mem ? r.ms : 0
    if (onNpu) npuMs += r.ms
    else cpuMs += r.ms
  })
  const switchBytes = [...switchTensors].reduce((s, t) => s + actBytes(t), 0)
  const switchMs = switches * device.switchMs + switchBytes / bw * 1000
  const totalMs = npuMs + cpuMs + switchMs
  const latency: Analysis['latency'] = { npuMs, cpuMs, switchMs, totalMs, computeShare: totalMs ? computeMs / totalMs : 0 }

  if (ir.llm) {
    // Decode is memory-bound: every token streams all weights once.
    const wb = weightBytes * (quantFix && dtypes.int4.params < params / 2 ? 0.25 : 1)
    latency.tokensPerSec = bw * device.util * 2 / wb
    const prefillTokens = 512
    latency.prefillMs = 2 * params * prefillTokens / npuRate * 1000
    const L = ir.llm
    if (L.blockCount && L.embeddingLength && L.contextLength) {
      const kvRatio = L.headCount && L.headCountKv ? L.headCountKv / L.headCount : 1
      latency.kvCacheBytes = 2 * L.blockCount * Math.min(L.contextLength, pins.ctx ?? 4096) * L.embeddingLength * kvRatio * 2
    }
    latency.totalMs = 1000 / latency.tokensPerSec
    latency.npuMs = latency.totalMs
  }

  // Op inventory
  const rows = new Map<string, OpRow>()
  nodes.forEach((r, i) => {
    const n = ir.nodes[i]
    const key = `${n.domain}|${n.op}`
    const rl = rule(n.op, n.domain)
    const row = rows.get(key) ?? { op: n.op, domain: n.domain, count: 0, support: rl.status, npu: 0, cpu: 0, macs: 0, note: rl.note, replace: rl.replace }
    row.count++
    if (r.status === 'npu') row.npu++
    else if (r.status === 'cpu') row.cpu++
    row.macs += r.macs
    rows.set(key, row)
  })
  const rank: Record<Support, number> = { cpu: 0, constrained: 1, npu: 2 }
  const opRows = [...rows.values()].sort((a, b) => b.cpu - a.cpu || rank[a.support] - rank[b.support] || b.count - a.count || a.op.localeCompare(b.op))

  const io = (name: string): IoRow => {
    const t = ir.tensors[name]
    const shape = symShapes[name] ?? t?.shape ?? null
    return { name, dtype: t?.dtype ?? 'unknown', shape, dynamic: (shape ?? []).filter((d): d is string => typeof d === 'string') }
  }
  const inputs = ir.inputs.map(io), outputs = ir.outputs.map(io)
  const mainOpset = ir.opset.find(o => o.domain === '' || o.domain === 'ai.onnx')?.version ?? null

  const out: Analysis = {
    format: ir.format, nodeCount: ir.nodes.length, opRows, nodes, inputs, outputs, opset: mainOpset,
    params, weightBytes, externalBytes, dtypes,
    macs: nodes.reduce((s, n) => s + n.macs, 0), flops: nodes.reduce((s, n) => s + n.flops, 0),
    peakBytes, peakAt, largest,
    npuNodes: nodes.filter(n => n.status === 'npu' && n.op !== 'Constant').length, cpuNodes: nodes.filter(n => n.status === 'cpu').length,
    npuPartitions, cpuPartitions, partitions: npuPartitions + cpuPartitions, switches,
    quant: { qdq, qlinear, perChannel, intWeightShare, verdict, precision },
    latency, findings: [], available: [], score: 0, symbols: syms,
  }
  out.findings = findings(ir, out, opts, pins, floatParams)
  out.available = [...new Set(out.findings.map(f => f.fix).filter((f): f is FixId => !!f))]
  out.score = score(out)
  return out
}

function constraintViolation(n: ModelNode, S: Record<string, Dim[] | null>, ir: ModelIR): string | null {
  const w = ir.tensors[n.inputs[1]]
  switch (n.op) {
    case 'Conv': case 'ConvTranspose':
      if ((S[n.inputs[1]]?.length ?? 4) === 5) return '3D convolution has no NPU kernel.'
      if (w && !w.init && ir.format === 'onnx' && !ir.nodes.some(m => m.outputs.includes(n.inputs[1]) && m.op === 'DequantizeLinear')) return 'Weights are not constant (computed at runtime).'
      if (n.op === 'ConvTranspose' && (ints(n.attrs.dilations) ?? []).some(d => d > 1)) return 'Dilated transposed convolution falls back.'
      return null
    case 'Resize': return n.attrs.mode === 'cubic' ? 'Cubic resize falls back; use linear.' : null
    case 'TopK': return n.inputs[1] && !ir.tensors[n.inputs[1]]?.values ? 'k is not a constant.' : null
    case 'ReduceMean': case 'ReduceSum': case 'ReduceMax': case 'ReduceMin':
      return n.inputs[1] && !ir.tensors[n.inputs[1]]?.values ? 'Reduction axes are computed at runtime.' : null
  }
  return null
}

/** MACs for Conv/MatMul/Gemm; FLOPs = 2 x MACs plus one per output element for everything else. */
export function cost(n: ModelNode, S: Record<string, Dim[] | null>, ir: ModelIR): { macs: number; flops: number } {
  const out = S[n.outputs[0]]
  const outN = numel(out)
  let macs = 0
  const tflite = n.attrs.layout === 'NHWC'
  if (n.op === 'Conv' || n.op === 'QLinearConv' || n.op === 'ConvTranspose') {
    const w = S[n.inputs[n.op === 'QLinearConv' ? 3 : 1]]
    if (w && out) {
      const wN = numel(w)
      const cOut = tflite ? (n.attrs.depthwise ? w[3] : w[0]) : n.op === 'ConvTranspose' ? (w[1] as number) * ((n.attrs.group as number) || 1) : w[0]
      if (n.op === 'ConvTranspose') macs = numel(S[n.inputs[0]]) * wN / numel([w[0]])
      else macs = outN * wN / Math.max(1, typeof cOut === 'number' ? cOut : 1)
    }
  } else if (n.op === 'MatMul' || n.op === 'QLinearMatMul' || n.op === 'MatMulInteger' || n.op === 'FusedMatMul') {
    const a = S[n.inputs[0]]
    if (a && out) macs = outN * (typeof a[a.length - 1] === 'number' ? a[a.length - 1] as number : 1)
  } else if (n.op === 'Gemm') {
    const a = S[n.inputs[0]]
    if (a && out) {
      // TFLite FULLY_CONNECTED weights are [out, in] and flatten the input.
      const k = tflite ? S[n.inputs[1]]?.[1] : n.attrs.transA === 1 ? a[0] : a[1]
      macs = outN * (typeof k === 'number' ? k : 1)
    }
  } else if (n.op === 'Einsum') {
    const eq = String(n.attrs.equation ?? '').replace(/\s/g, '')
    const [lhs, rhs = ''] = eq.split('->')
    const dims: Record<string, number> = {}
    lhs.split(',').forEach((t, k) => [...t].forEach((c, i) => { const d = S[n.inputs[k]]?.[i]; if (typeof d === 'number') dims[c] = Math.max(dims[c] ?? 1, d) }))
    const reduced = Object.keys(dims).filter(c => !rhs.includes(c)).reduce((p, c) => p * dims[c], 1)
    macs = outN * reduced
  }
  return { macs, flops: macs ? 2 * macs : outN }
}

const MB = 1024 * 1024
export const fmtBytes = (b: number) => b >= 1024 * MB ? `${(b / 1024 / MB).toFixed(2)} GB` : b >= MB ? `${(b / MB).toFixed(1)} MB` : b >= 1024 ? `${(b / 1024).toFixed(1)} KB` : `${Math.round(b)} B`

function findings(ir: ModelIR, a: Analysis, opts: Options, pins: Record<string, number>, floatParams: number): Finding[] {
  const f: Finding[] = []
  const applied = new Set(opts.applied)
  if (ir.llm) {
    const L = ir.llm
    f.push({ id: 'llm-path', severity: 'info', title: 'LLM weights: plan the runtime, not just the file', detail: `GGUF is a llama.cpp format. On Snapdragon, NPU execution of LLMs goes through a converted model (QAIRT / Genie, or llama.cpp's experimental Hexagon backend), with static context buckets and a KV cache sized up front. Architecture: ${L.arch}${L.blockCount ? `, ${L.blockCount} blocks` : ''}${L.contextLength ? `, trained context ${L.contextLength}` : ''}.` })
    if (a.dtypes.fp32.params + a.dtypes.fp16.params > a.params * 0.5) f.push({ id: 'llm-float', severity: 'warn', title: 'Mostly float weights', detail: `Decode streams every weight per token, so ${fmtBytes(a.weightBytes)} of float weights caps throughput. A 4-bit build cuts memory traffic about 4x versus fp16.`, fix: 'quantize-int8' })
    if (a.latency.kvCacheBytes) f.push({ id: 'kv', severity: 'info', title: `KV cache: ${fmtBytes(a.latency.kvCacheBytes)} at ${Math.min(L.contextLength ?? 0, pins.ctx ?? 4096)} tokens`, detail: 'fp16 cache, grouped-query heads accounted for. On phones, this plus weights must fit next to everything else in shared memory.' })
    return f
  }
  const dynIn = a.inputs.filter(i => i.dynamic.length)
  if (dynIn.length && !applied.has('pin-dims')) {
    const syms = [...new Set(dynIn.flatMap(i => i.dynamic))]
    f.push({ id: 'dynamic', severity: 'blocker', title: `Dynamic input shapes: ${syms.join(', ')}`, detail: `${dynIn.map(i => i.name).join(', ')} ${dynIn.length > 1 ? 'have' : 'has'} symbolic dims. NPU graphs are compiled for fixed shapes, so every node downstream falls back. Pin them (${syms.map(s => `${s}=${pins[s]}`).join(', ')}) and export one model per shape bucket if you need several.`, fix: 'pin-dims', nodes: a.nodes.filter(n => n.reason.startsWith('Dynamic')).map(n => n.name) })
  }
  const unsupported = a.nodes.filter(n => n.status === 'cpu' && n.support === 'cpu')
  const byOp = new Map<string, string[]>()
  unsupported.forEach(n => byOp.set(n.op, [...(byOp.get(n.op) ?? []), n.name]))
  for (const [op, names] of byOp) {
    const r = rule(op, ir.nodes.find(n => n.op === op)?.domain ?? '')
    f.push({ id: `op-${op}`, severity: r.post ? 'warn' : 'blocker', title: `${op} runs on the CPU (${names.length} node${names.length > 1 ? 's' : ''})`, detail: `${r.note}${r.replace ? ` Fix: ${r.replace}` : ''}${r.post ? ' This is post-processing: cut the graph before it and run it on the CPU once per frame.' : ''}`, nodes: names, fix: r.post ? 'split-post' : r.replace ? 'replace-ops' : undefined })
  }
  const constrainedFail = a.nodes.filter(n => n.status === 'cpu' && n.support === 'constrained' && !n.reason.startsWith('Dynamic') && !n.reason.includes('data-dependent'))
  if (constrainedFail.length) f.push({ id: 'constraints', severity: 'warn', title: `${constrainedFail.length} supported op${constrainedFail.length > 1 ? 's' : ''} outside their constraints`, detail: [...new Set(constrainedFail.map(n => `${n.op}: ${n.reason}`))].join(' '), nodes: constrainedFail.map(n => n.name) })
  if (a.partitions > 1) f.push({ id: 'partitions', severity: a.switches > 2 ? 'blocker' : 'warn', title: `${a.partitions} partitions, ${a.switches} CPU/NPU switch${a.switches === 1 ? '' : 'es'}`, detail: `Each hand-off copies activations through shared memory and adds about ${opts.device.switchMs} ms on ${opts.device.name}. Estimated cost: ${a.latency.switchMs.toFixed(2)} ms per inference.` })
  if (a.quant.verdict === 'float' && !applied.has('quantize-int8')) f.push({ id: 'float', severity: 'warn', title: 'Float model: no QDQ quantization', detail: `${fmtBytes(a.weightBytes)} of float weights. The NPU runs fp32 graphs at fp16 at best (half the int8 rate) and some delegates refuse fp32 outright. Quantize to int8 QDQ with 100 to 500 calibration samples from real inputs; keep LayerNorm and Softmax at int16 or fp16 if accuracy drops.`, fix: 'quantize-int8' })
  else if (a.quant.verdict === 'quantized-per-tensor') f.push({ id: 'per-tensor', severity: 'info', title: 'Per-tensor weight quantization', detail: 'Per-channel weight scales for Conv and MatMul usually recover most of the accuracy lost at int8 at no speed cost.' })
  else if (a.quant.verdict === 'weight-only') f.push({ id: 'weight-only', severity: 'info', title: 'Weights are integer but activations are float', detail: 'Weight-only quantization saves memory but the math still runs in float. Full QDQ (activations too) is what unlocks int8 kernels.', fix: 'quantize-int8' })
  if (floatParams && a.dtypes.fp32.params > 0 && a.quant.verdict !== 'float' && a.dtypes.fp32.params / Math.max(1, a.params) > 0.2) f.push({ id: 'fp32-left', severity: 'info', title: `${Math.round(a.dtypes.fp32.params / a.params * 100)}% of weights still fp32`, detail: 'Mixed graph: check that the fp32 tensors are biases or excluded layers, not a missed quantization pass.' })
  const vtcm = opts.device.vtcmMB * MB
  const big = a.nodes.filter(n => n.status === 'npu').map(n => ({ n, b: numel(n.outShape, pins) * (a.quant.precision === 'int8' ? 1 : 2) })).filter(x => x.b > vtcm)
  if (big.length) f.push({ id: 'activations', severity: 'warn', title: `${big.length} activation${big.length > 1 ? 's' : ''} larger than ${opts.device.vtcmMB} MB of NPU scratch`, detail: `Largest: ${big.sort((x, y) => y.b - x.b)[0].n.name} at ${fmtBytes(big[0].b)}. Tensors that do not fit spill to DRAM and become bandwidth-bound; tile the input or reduce resolution.`, nodes: big.map(x => x.n.name) })
  if (a.opset !== null && a.opset < 13) f.push({ id: 'opset', severity: 'info', title: `Old opset ${a.opset}`, detail: 'Converters are tested most against opset 13 to 20. Re-export with opset_version=17 to avoid legacy op semantics.' })
  if (a.externalBytes) f.push({ id: 'external', severity: 'info', title: `${fmtBytes(a.externalBytes)} of weights in external data`, detail: 'Normal for models above 2 GB. Keep the .data file next to the .onnx when converting; it is not needed for this inspection.' })
  for (const i of ir.issues.filter(x => x.level === 'error').slice(0, 5)) f.push({ id: `issue-${i.where}`, severity: 'warn', title: `Graph issue at ${i.where}`, detail: i.message })
  return f
}

function score(a: Analysis): number {
  if (a.format === 'gguf') return a.dtypes.int4.params + a.dtypes.int8.params > a.params / 2 ? 70 : 45
  const total = Math.max(1, a.npuNodes + a.cpuNodes)
  let s = 100 * a.npuNodes / total
  s -= Math.min(20, Math.max(0, a.partitions - 1) * 5)
  if (a.quant.precision !== 'int8') s -= 15
  if (a.findings.some(f => f.id === 'activations')) s -= 5
  return Math.max(0, Math.min(100, Math.round(s)))
}

// ---------- Reports ----------

const csvCell = (v: string | number) => { const s = String(v); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s }

export function opReportCsv(a: Analysis): string {
  const head = ['op', 'domain', 'count', 'support', 'on_npu', 'on_cpu', 'macs', 'note', 'replacement']
  const lines = a.opRows.map(r => [r.op, r.domain || 'ai.onnx', r.count, r.support, r.npu, r.cpu, Math.round(r.macs), r.note, r.replace ?? ''].map(csvCell).join(','))
  return [head.join(','), ...lines].join('\n') + '\n'
}

export function nodeReportCsv(a: Analysis): string {
  const head = ['index', 'name', 'op', 'placement', 'reason', 'output_shape', 'macs', 'est_ms']
  return [head.join(','), ...a.nodes.map(n => [n.index, n.name, n.op, n.status, n.reason, (n.outShape ?? []).join('x'), Math.round(n.macs), n.ms.toFixed(4)].map(csvCell).join(','))].join('\n') + '\n'
}

// ---------- Version summaries and comparison ----------

export interface Summary {
  params: number; weightBytes: number; macs: number; peakBytes: number; nodeCount: number
  partitions: number; switches: number; cpuNodes: number; npuNodes: number; latencyMs: number; score: number
  precision: string; ops: Record<string, number>; findings: { id: string; title: string; severity: string }[]
  inputs: { name: string; shape: Dim[] | null }[]
}

export function summarize(a: Analysis): Summary {
  return {
    params: a.params, weightBytes: a.weightBytes, macs: a.macs, peakBytes: a.peakBytes, nodeCount: a.nodeCount,
    partitions: a.partitions, switches: a.switches, cpuNodes: a.cpuNodes, npuNodes: a.npuNodes, latencyMs: a.latency.totalMs, score: a.score,
    precision: a.quant.verdict, ops: Object.fromEntries(a.opRows.map(r => [r.op, r.count])),
    findings: a.findings.map(f => ({ id: f.id, title: f.title, severity: f.severity })), inputs: a.inputs.map(i => ({ name: i.name, shape: i.shape })),
  }
}

export interface Diff {
  metrics: { key: keyof Summary; label: string; a: number; b: number; better: boolean | null }[]
  ops: { op: string; a: number; b: number }[]
  fixed: string[]
  introduced: string[]
}

export function compare(a: Summary, b: Summary): Diff {
  const m = (key: keyof Summary, label: string, lowerIsBetter: boolean | null) => {
    const x = a[key] as number, y = b[key] as number
    return { key, label, a: x, b: y, better: lowerIsBetter === null || x === y ? null : lowerIsBetter ? y < x : y > x }
  }
  const metrics = [
    m('score', 'Readiness score', false), m('latencyMs', 'Estimated latency (ms)', true), m('partitions', 'Partitions', true),
    m('switches', 'CPU/NPU switches', true), m('cpuNodes', 'Nodes on CPU', true), m('weightBytes', 'Weight bytes', true),
    m('params', 'Parameters', null), m('macs', 'MACs', null), m('peakBytes', 'Peak activation bytes', true), m('nodeCount', 'Nodes', null),
  ]
  const opsAll = [...new Set([...Object.keys(a.ops), ...Object.keys(b.ops)])].sort()
  const ops = opsAll.map(op => ({ op, a: a.ops[op] ?? 0, b: b.ops[op] ?? 0 })).filter(o => o.a !== o.b)
  const ida = new Set(a.findings.map(f => f.id)), idb = new Set(b.findings.map(f => f.id))
  return { metrics, ops, fixed: a.findings.filter(f => !idb.has(f.id)).map(f => f.title), introduced: b.findings.filter(f => !ida.has(f.id)).map(f => f.title) }
}
