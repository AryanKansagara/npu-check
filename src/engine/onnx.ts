// ONNX ModelProto decoder (field numbers from onnx/onnx.proto) and a matching encoder used for the
// synthetic samples, plus the dynamic-shape fixer that rewrites dims in place.
import { Reader, rewrite, Writer, type Field } from './pb.ts'
import type { AttrValue, Dim, DType, Issue, ModelIR, ModelNode, TensorInfo } from './types.ts'

const ONNX_DTYPES: Record<number, DType> = {
  1: 'fp32', 2: 'uint8', 3: 'int8', 4: 'uint16', 5: 'int16', 6: 'int32', 7: 'int64', 8: 'string', 9: 'bool',
  10: 'fp16', 11: 'fp64', 12: 'uint32', 13: 'uint64', 14: 'complex', 15: 'complex', 16: 'bf16',
  17: 'fp8', 18: 'fp8', 19: 'fp8', 20: 'fp8', 21: 'uint4', 22: 'int4', 23: 'fp8',
}
export const ONNX_DTYPE_CODE: Partial<Record<DType, number>> = Object.fromEntries(
  Object.entries(ONNX_DTYPES).filter(([k]) => !['15', '18', '19', '20', '23'].includes(k)).map(([k, v]) => [v, +k]),
)

/** Integer constants up to this many elements are kept for shape inference. */
const KEEP_VALUES = 64

interface Ctx { tensors: Record<string, TensorInfo>; issues: Issue[] }

export function parseOnnx(buf: Uint8Array): ModelIR {
  const r = new Reader(buf)
  const ir: ModelIR = { format: 'onnx', producer: '', opset: [], graphName: '', inputs: [], outputs: [], tensors: {}, nodes: [], issues: [], meta: {}, fileSize: buf.length }
  let graph: Field | null = null
  let producerVersion = ''
  let sawField = false
  for (const f of r.fields()) {
    sawField = true
    if (f.no === 1 && f.wt === 0) ir.irVersion = f.n
    else if (f.no === 2 && f.wt === 2) ir.producer = r.str(f)
    else if (f.no === 3 && f.wt === 2) producerVersion = r.str(f)
    else if (f.no === 7 && f.wt === 2) graph = f
    else if (f.no === 8 && f.wt === 2) {
      const s = r.sub(f)
      const o = { domain: '', version: 0 }
      for (const g of s.fields()) { if (g.no === 1) o.domain = s.str(g); else if (g.no === 2) o.version = g.n }
      ir.opset.push(o)
    } else if (f.no === 14 && f.wt === 2) {
      const s = r.sub(f)
      let k = '', v = ''
      for (const g of s.fields()) { if (g.no === 1) k = s.str(g); else if (g.no === 2) v = s.str(g) }
      if (k) ir.meta[k] = v
    }
  }
  if (!sawField || ir.irVersion === undefined || !graph) throw new Error('Not an ONNX model: no ir_version or graph field found. Is this the .onnx file (not the .onnx.data weights file)?')
  if (producerVersion) ir.producer = `${ir.producer} ${producerVersion}`.trim()
  const ctx: Ctx = { tensors: ir.tensors, issues: ir.issues }
  const g = r.sub(graph)
  const initNames = new Set<string>()
  const vinfos: TensorInfo[] = []
  for (const f of g.fields()) {
    if (f.no === 1) ir.nodes.push(parseNode(g, f, ctx, ir.nodes.length))
    else if (f.no === 2) ir.graphName = g.str(f)
    else if (f.no === 5) { const t = parseTensor(g, f, ctx); t.init = true; ir.tensors[t.name] = t; initNames.add(t.name) }
    else if (f.no === 15) {
      // SparseTensorProto: values=1 (TensorProto), dims=3
      const s = g.sub(f)
      let t: TensorInfo | null = null
      const dims: number[] = []
      for (const h of s.fields()) { if (h.no === 1) t = parseTensor(s, h, ctx); else if (h.no === 3) s.ints(h, dims) }
      if (t) { ir.tensors[t.name] = { ...t, shape: dims, init: true }; initNames.add(t.name) }
    } else if (f.no === 11) { const v = parseValueInfo(g, f); ir.inputs.push(v.name); vinfos.push(v) }
    else if (f.no === 12) { const v = parseValueInfo(g, f); ir.outputs.push(v.name); vinfos.push(v) }
    else if (f.no === 13) vinfos.push(parseValueInfo(g, f))
  }
  for (const v of vinfos) if (!ir.tensors[v.name]?.init) ir.tensors[v.name] = v
  // Older exporters list initializers as graph inputs too; those are weights, not inputs.
  ir.inputs = ir.inputs.filter(n => !initNames.has(n))
  if (!ir.opset.length) ir.issues.push({ level: 'warn', where: 'model', message: 'No opset_import; assuming the default ONNX domain.' })
  const externals = Object.values(ir.tensors).filter(t => t.external)
  if (externals.length) ir.issues.push({ level: 'info', where: 'initializers', message: `${externals.length} initializers live in external data files (not needed here: sizes come from their dims).` })
  checkGraph(ir)
  return ir
}

function parseNode(r: Reader, f: Field, ctx: Ctx, idx: number): ModelNode {
  const s = r.sub(f)
  const n: ModelNode = { name: '', op: '', domain: '', inputs: [], outputs: [], attrs: {} }
  for (const g of s.fields()) {
    if (g.no === 1) n.inputs.push(s.str(g))
    else if (g.no === 2) n.outputs.push(s.str(g))
    else if (g.no === 3) n.name = s.str(g)
    else if (g.no === 4) n.op = s.str(g)
    else if (g.no === 7) n.domain = s.str(g)
    else if (g.no === 5) {
      const [k, v] = parseAttr(s, g, n, ctx)
      if (k) n.attrs[k] = v
    }
  }
  if (n.domain === 'ai.onnx') n.domain = ''
  if (!n.name) n.name = `${n.op}_${idx}`
  return n
}

function parseAttr(r: Reader, f: Field, node: ModelNode, ctx: Ctx): [string, AttrValue] {
  const s = r.sub(f)
  let name = ''
  let v: AttrValue = 0
  const ints: number[] = [], floats: number[] = [], strs: string[] = []
  let type = 0
  for (const g of s.fields()) {
    if (g.no === 1) name = s.str(g)
    else if (g.no === 20) type = g.n
    else if (g.no === 2) v = s.f32(g)
    else if (g.no === 3) v = g.n
    else if (g.no === 4) v = s.str(g)
    else if (g.no === 7) s.floats(g, floats)
    else if (g.no === 8) s.ints(g, ints)
    else if (g.no === 9) strs.push(s.str(g))
    else if (g.no === 5) {
      // Constant node value: registered under the node output (serializers write outputs before attributes).
      const t = parseTensor(s, g, ctx)
      v = t.values ?? []
      const out = node.outputs[0]
      if (out) ctx.tensors[out] = { ...t, name: out, init: true }
    } else if (g.no === 6 || g.no === 11) v = 'subgraph'
  }
  if (type === 7 || (ints.length && !type)) v = ints
  else if (type === 6 || (floats.length && !type)) v = floats
  else if (type === 8 || (strs.length && !type)) v = strs
  return [name, v]
}

function parseTensor(r: Reader, f: Field, ctx: Ctx): TensorInfo {
  const s = r.sub(f)
  const t: TensorInfo = { name: '', dtype: 'unknown', shape: [] }
  const dims: number[] = []
  let raw: Field | null = null
  const i64: number[] = [], i32: number[] = []
  for (const g of s.fields()) {
    if (g.no === 1) s.ints(g, dims)
    else if (g.no === 2) t.dtype = ONNX_DTYPES[g.n] ?? 'unknown'
    else if (g.no === 8) t.name = s.str(g)
    else if (g.no === 9) raw = g
    else if (g.no === 7) s.ints(g, i64)
    else if (g.no === 5) s.ints(g, i32)
    else if (g.no === 14 && g.n === 1) t.external = true
  }
  t.shape = dims
  const numel = dims.reduce((a, b) => a * b, 1)
  if ((t.dtype === 'int64' || t.dtype === 'int32') && numel <= KEEP_VALUES) {
    if (raw) {
      const dv = new DataView(s.buf.buffer, s.buf.byteOffset + raw.start, raw.end - raw.start)
      const w = t.dtype === 'int64' ? 8 : 4
      t.values = []
      for (let p = 0; p + w <= dv.byteLength; p += w) t.values.push(w === 8 ? Number(dv.getBigInt64(p, true)) : dv.getInt32(p, true))
    } else t.values = t.dtype === 'int64' ? i64 : i32
  }
  if (t.dtype === 'unknown') ctx.issues.push({ level: 'warn', where: `tensor ${t.name}`, message: 'Unknown data type; size assumed 4 bytes per element.' })
  return t
}

function parseValueInfo(r: Reader, f: Field): TensorInfo {
  const s = r.sub(f)
  const t: TensorInfo = { name: '', dtype: 'unknown', shape: null }
  for (const g of s.fields()) {
    if (g.no === 1) t.name = s.str(g)
    else if (g.no === 2) {
      const ty = s.sub(g)
      for (const h of ty.fields()) {
        if (h.no !== 1) continue // only tensor_type
        const tt = ty.sub(h)
        for (const k of tt.fields()) {
          if (k.no === 1) t.dtype = ONNX_DTYPES[k.n] ?? 'unknown'
          else if (k.no === 2) {
            t.shape = []
            const sh = tt.sub(k)
            for (const d of sh.fields()) {
              if (d.no !== 1) continue
              const dm = sh.sub(d)
              let dim: Dim = '?'
              for (const x of dm.fields()) { if (x.no === 1) dim = x.n; else if (x.no === 2) dim = dm.str(x) || '?' }
              t.shape.push(dim)
            }
          }
        }
      }
    }
  }
  return t
}

/** Structural checks: dangling inputs, duplicate outputs. Reports node index so users can find it. */
function checkGraph(ir: ModelIR) {
  const produced = new Set<string>([...ir.inputs, ...Object.values(ir.tensors).filter(t => t.init).map(t => t.name), ''])
  const seen = new Map<string, number>()
  ir.nodes.forEach((n, i) => {
    for (const o of n.outputs) {
      if (seen.has(o)) ir.issues.push({ level: 'error', where: `node ${i} (${n.name})`, message: `Output "${o}" is also produced by node ${seen.get(o)}.` })
      seen.set(o, i)
      produced.add(o)
    }
    if (!n.op) ir.issues.push({ level: 'error', where: `node ${i}`, message: 'Node has no op_type.' })
  })
  ir.nodes.forEach((n, i) => {
    for (const x of n.inputs) if (!produced.has(x)) ir.issues.push({ level: 'error', where: `node ${i} (${n.name})`, message: `Input "${x}" has no producer, initializer or graph input.` })
  })
  for (const o of ir.outputs) if (!produced.has(o)) ir.issues.push({ level: 'error', where: 'graph outputs', message: `Output "${o}" is never produced.` })
}

// ---------- Encoder (synthetic samples and tests) ----------

export interface BuildTensor { name: string; dtype: DType; dims: number[]; data?: Float32Array | number[]; external?: boolean }
export interface BuildValue { name: string; dtype: DType; shape: Dim[] }
export interface BuildNode { op: string; inputs: string[]; outputs: string[]; name?: string; domain?: string; attrs?: Record<string, AttrValue> }
export interface BuildGraph {
  name: string; producer?: string; opset?: number; extraOpsets?: { domain: string; version: number }[]
  inputs: BuildValue[]; outputs: BuildValue[]; nodes: BuildNode[]; inits: BuildTensor[]; valueInfo?: BuildValue[]
  meta?: Record<string, string>
}

function encTensor(t: BuildTensor): Writer {
  const w = new Writer()
  w.packedInts(1, t.dims).int(2, ONNX_DTYPE_CODE[t.dtype] ?? 1).str(8, t.name)
  if (t.external) {
    w.msg(13, new Writer().str(1, 'location').str(2, 'weights.onnx.data')).int(14, 1)
    return w
  }
  const n = t.dims.reduce((a, b) => a * b, 1)
  const width = t.dtype === 'int64' ? 8 : t.dtype === 'fp16' ? 2 : t.dtype === 'int8' || t.dtype === 'uint8' ? 1 : 4
  const raw = new Uint8Array(n * width)
  const dv = new DataView(raw.buffer)
  for (let i = 0; i < n; i++) {
    const v = t.data ? Number(t.data[i] ?? 0) : 0
    if (t.dtype === 'int64') dv.setBigInt64(i * 8, BigInt(Math.round(v)), true)
    else if (t.dtype === 'int32') dv.setInt32(i * 4, v, true)
    else if (width === 1) dv.setInt8(i, v)
    else if (width === 2) dv.setUint16(i * 2, 0, true)
    else dv.setFloat32(i * 4, v, true)
  }
  return w.bytes(9, raw)
}

function encValue(v: BuildValue): Writer {
  const shape = new Writer()
  for (const d of v.shape) shape.msg(1, typeof d === 'number' ? new Writer().int(1, d) : new Writer().str(2, d))
  const tt = new Writer().int(1, ONNX_DTYPE_CODE[v.dtype] ?? 1).msg(2, shape)
  return new Writer().str(1, v.name).msg(2, new Writer().msg(1, tt))
}

function encAttr(name: string, v: AttrValue): Writer {
  const w = new Writer().str(1, name)
  if (typeof v === 'number') return Number.isInteger(v) ? w.int(3, v).int(20, 2) : w.f32(2, v).int(20, 1)
  if (typeof v === 'string') return w.str(4, v).int(20, 3)
  if (v.length && typeof v[0] === 'string') { (v as string[]).forEach(s => w.str(9, s)); return w.int(20, 8) }
  return w.packedInts(8, v as number[]).int(20, 7)
}

export function buildOnnx(g: BuildGraph): Uint8Array {
  const gw = new Writer()
  g.nodes.forEach((n, i) => {
    const nw = new Writer()
    n.inputs.forEach(x => nw.str(1, x))
    n.outputs.forEach(x => nw.str(2, x))
    nw.str(3, n.name ?? `${n.op}_${i}`).str(4, n.op)
    for (const [k, v] of Object.entries(n.attrs ?? {})) nw.msg(5, encAttr(k, v))
    if (n.domain) nw.str(7, n.domain)
    gw.msg(1, nw)
  })
  gw.str(2, g.name)
  g.inits.forEach(t => gw.msg(5, encTensor(t)))
  g.inputs.forEach(v => gw.msg(11, encValue(v)))
  g.outputs.forEach(v => gw.msg(12, encValue(v)))
  ;(g.valueInfo ?? []).forEach(v => gw.msg(13, encValue(v)))
  const m = new Writer().int(1, 9).str(2, g.producer ?? 'npu-check synthetic').str(3, '1.0').msg(7, gw)
  m.msg(8, new Writer().str(1, '').int(2, g.opset ?? 17))
  for (const o of g.extraOpsets ?? []) m.msg(8, new Writer().str(1, o.domain).int(2, o.version))
  for (const [k, v] of Object.entries(g.meta ?? {})) m.msg(14, new Writer().str(1, k).str(2, v))
  return m.finish()
}

// ---------- Dynamic shape fixer ----------

/**
 * Returns a copy of the model where symbolic dims named in `pins` become fixed values in graph inputs,
 * outputs and value_info. Everything else (weights, nodes, unknown fields) is copied byte-for-byte.
 * Same idea as onnxruntime.tools.make_dynamic_shape_fixed.
 */
export function pinDims(buf: Uint8Array, pins: Record<string, number>): Uint8Array {
  const dimFix = (b: Uint8Array, f: Field) => {
    const r = new Reader(b, f.start, f.end)
    let param = ''
    for (const x of r.fields()) if (x.no === 2) param = r.str(x)
    return param in pins ? new Writer().int(1, pins[param]).finish() : null
  }
  const shapeFix = (b: Uint8Array, f: Field) => rewrite(b, f.start, f.end, d => d.no === 1 ? dimFix(b, d) : null)
  const ttFix = (b: Uint8Array, f: Field) => rewrite(b, f.start, f.end, d => d.no === 2 ? shapeFix(b, d) : null)
  const typeFix = (b: Uint8Array, f: Field) => rewrite(b, f.start, f.end, d => d.no === 1 ? ttFix(b, d) : null)
  const vinfoFix = (b: Uint8Array, f: Field) => rewrite(b, f.start, f.end, d => d.no === 2 ? typeFix(b, d) : null)
  const graphFix = (b: Uint8Array, f: Field) => rewrite(b, f.start, f.end, d => d.no === 11 || d.no === 12 || d.no === 13 ? vinfoFix(b, d) : null)
  return rewrite(buf, 0, buf.length, f => f.no === 7 ? graphFix(buf, f) : null)
}
