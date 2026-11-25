// Model files written byte by byte for the tests, independently of the app's own encoder, from the published
// formats: onnx/onnx.proto field numbers, ggml/docs/gguf.md. TFLite files use the test builder in src/engine.
import { buildFlatbuffer, type FbTable } from '../src/engine/tflite.ts'

export type Bytes = Buffer

// ---------- protobuf / ONNX ----------

const varint = (v: number | bigint): Bytes => {
  let b = BigInt.asUintN(64, BigInt(v))
  const out: number[] = []
  while (b >= 0x80n) { out.push(Number(b & 0x7fn) | 0x80); b >>= 7n }
  return Buffer.from([...out, Number(b)])
}
const int = (no: number, v: number) => Buffer.concat([varint(no * 8), varint(v)])
const len = (no: number, body: Bytes | string) => { const b = Buffer.from(body as string); return Buffer.concat([varint(no * 8 + 2), varint(b.length), b]) }
const msg = (no: number, ...parts: Bytes[]) => len(no, Buffer.concat(parts))

const TYPE = { fp32: 1, uint8: 2, int64: 7, fp16: 10 } as const
export type Ty = keyof typeof TYPE
export interface Value { name: string; shape: (number | string)[]; type?: Ty }
export interface Init { name: string; dims: number[]; type?: Ty; ints?: number[]; external?: boolean }
export interface Node { op: string; in: string[]; out: string[]; name?: string; domain?: string; attrs?: Record<string, number | number[] | string> }
export interface Model { inputs: Value[]; outputs: Value[]; nodes: Node[]; inits?: Init[]; valueInfo?: Value[]; opset?: number }

// '' is a dim with neither a value nor a name.
const value = (no: number, v: Value) => msg(no, len(1, v.name), msg(2, msg(1, int(1, TYPE[v.type ?? 'fp32']),
  msg(2, ...v.shape.map(d => msg(1, ...(typeof d === 'number' ? [int(1, d)] : d ? [len(2, d)] : [])))))))

function tensor(t: Init) {
  const type = t.type ?? (t.ints ? 'int64' : 'fp32')
  const head = [msg(1, ...t.dims.map(varint)), int(2, TYPE[type]), len(8, t.name)]
  if (t.external) return msg(5, ...head, msg(13, len(1, 'location'), len(2, 'weights.bin')), int(14, 1))
  const width = { fp32: 4, fp16: 2, uint8: 1, int64: 8 }[type]
  const raw = Buffer.alloc(t.dims.reduce((a, b) => a * b, 1) * width)
  t.ints?.forEach((v, i) => raw.writeBigInt64LE(BigInt(v), i * 8))
  return msg(5, ...head, len(9, raw))
}

const attr = (name: string, v: number | number[] | string) => msg(5, len(1, name),
  ...(typeof v === 'number' ? [int(3, v), int(20, 2)] : typeof v === 'string' ? [len(4, v), int(20, 3)] : [...v.map(x => int(8, x)), int(20, 7)]))

export function onnx(m: Model): Bytes {
  return Buffer.concat([
    int(1, 8), len(2, 'e2e'),
    msg(7,
      ...m.nodes.map(n => msg(1, ...n.in.map(x => len(1, x)), ...n.out.map(x => len(2, x)), ...(n.name ? [len(3, n.name)] : []), len(4, n.op),
        ...Object.entries(n.attrs ?? {}).map(([k, v]) => attr(k, v)), ...(n.domain ? [len(7, n.domain)] : []))),
      len(2, 'g'), ...(m.inits ?? []).map(tensor), ...m.inputs.map(v => value(11, v)), ...m.outputs.map(v => value(12, v)), ...(m.valueInfo ?? []).map(v => value(13, v))),
    msg(8, len(1, ''), int(2, m.opset ?? 17)),
  ])
}

/** The paper example: see the sums in onnx.spec.ts. */
export const handWorked = (nodeName = 'conv') => onnx({
  inputs: [{ name: 'x', shape: [1, 3, 8, 8] }], outputs: [{ name: 'y', shape: [1, 10] }],
  inits: [{ name: 'w', dims: [4, 3, 3, 3] }, { name: 'b', dims: [4] }, { name: 'fw', dims: [10, 256] }, { name: 'fb', dims: [10] }],
  nodes: [
    { op: 'Conv', name: nodeName, in: ['x', 'w', 'b'], out: ['c'], attrs: { kernel_shape: [3, 3], pads: [1, 1, 1, 1] } },
    { op: 'Relu', name: 'act', in: ['c'], out: ['r'] },
    { op: 'Flatten', name: 'flat', in: ['r'], out: ['f'], attrs: { axis: 1 } },
    { op: 'Gemm', name: 'fc', in: ['f', 'fw', 'fb'], out: ['y'], attrs: { transB: 1 } },
  ],
})

/** x[1,16,8,8] through `nodes` to y. */
export const small = (nodes: Node[], extra: Partial<Model> = {}) => onnx({ inputs: [{ name: 'x', shape: [1, 16, 8, 8] }], outputs: [{ name: 'y', shape: [1, 16, 8, 8] }], nodes, ...extra })

// ---------- GGUF v3 ----------

export interface GgufTensor { name: string; dims: number[]; type: number; offset: number }
export function ggufHeader(kv: [string, string | number][], tensors: GgufTensor[]): Bytes {
  const u32 = (v: number) => { const b = Buffer.alloc(4); b.writeUInt32LE(v); return b }
  const u64 = (v: number) => { const b = Buffer.alloc(8); b.writeBigUInt64LE(BigInt(v)); return b }
  const str = (s: string) => Buffer.concat([u64(Buffer.byteLength(s)), Buffer.from(s)])
  return Buffer.concat([Buffer.from('GGUF'), u32(3), u64(tensors.length), u64(kv.length),
    ...kv.map(([k, v]) => Buffer.concat([str(k), ...(typeof v === 'string' ? [u32(8), str(v)] : [u32(4), u32(v)])])),
    ...tensors.map(t => Buffer.concat([str(t.name), u32(t.dims.length), ...t.dims.map(u64), u32(t.type), u64(t.offset)]))])
}
/** Header, padding to the 32-byte alignment, then `data` bytes of tensor data. */
export function gguf(kv: [string, string | number][], tensors: GgufTensor[], data: number): Bytes {
  const h = ggufHeader(kv, tensors)
  return Buffer.concat([h, Buffer.alloc((32 - h.length % 32) % 32 + data)])
}
/** One F16 1024 x 1024 tensor: 2,097,152 bytes of data. */
export const llm = (kv: [string, string | number][], arch = 'llama') => gguf([['general.architecture', arch], ...kv], [{ name: 'blk.0.attn_q.weight', dims: [1024, 1024], type: 1, offset: 0 }], 2_097_152)

// ---------- TFLite ----------

export interface TflTensor { name: string; shape: number[]; type?: number; weight?: number }
/** One operator. Tensor `weight` is the byte size of its constant buffer. */
export function tflite(builtin: number, tensors: TflTensor[], op: { in: number[]; out: number[] }, io: { in: number[]; out: number[] }): Bytes {
  const buffers: FbTable[] = [[]]
  const tbl = tensors.map((t): FbTable => {
    const buf = t.weight ? buffers.push([{ vu8: new Uint8Array(t.weight) }]) - 1 : 0
    return [{ vi32: t.shape }, { u8: t.type ?? 0 }, { u32: buf }, { str: t.name }]
  })
  return Buffer.from(buildFlatbuffer([{ u32: 3 }, { vtbl: [[{ u8: builtin }, undefined, { i32: 1 }, { i32: builtin }]] },
    { vtbl: [[{ vtbl: tbl }, { vi32: io.in }, { vi32: io.out }, { vtbl: [[{ u32: 0 }, { vi32: op.in }, { vi32: op.out }]] }, { str: 'main' }]] },
    { str: 'e2e' }, { vtbl: buffers }]))
}
