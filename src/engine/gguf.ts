// GGUF header reader (v2/v3): metadata key/values and tensor infos. Never reads tensor data,
// so multi-GB files only need their first few MB.
import type { Issue, LlmMeta, ModelIR, TensorInfo } from './types.ts'

/** ggml type id -> [name, block size in elements, bytes per block] */
export const GGML_TYPES: Record<number, [string, number, number]> = {
  0: ['F32', 1, 4], 1: ['F16', 1, 2], 2: ['Q4_0', 32, 18], 3: ['Q4_1', 32, 20], 6: ['Q5_0', 32, 22], 7: ['Q5_1', 32, 24],
  8: ['Q8_0', 32, 34], 9: ['Q8_1', 32, 36], 10: ['Q2_K', 256, 84], 11: ['Q3_K', 256, 110], 12: ['Q4_K', 256, 144],
  13: ['Q5_K', 256, 176], 14: ['Q6_K', 256, 210], 15: ['Q8_K', 256, 292], 16: ['IQ2_XXS', 256, 66], 17: ['IQ2_XS', 256, 74],
  18: ['IQ3_XXS', 256, 98], 19: ['IQ1_S', 256, 50], 20: ['IQ4_NL', 32, 18], 21: ['IQ3_S', 256, 110], 22: ['IQ2_S', 256, 82],
  23: ['IQ4_XS', 256, 136], 24: ['I8', 1, 1], 25: ['I16', 1, 2], 26: ['I32', 1, 4], 27: ['I64', 1, 8], 28: ['F64', 1, 8],
  29: ['IQ1_M', 256, 56], 30: ['BF16', 1, 2], 34: ['TQ1_0', 256, 54], 35: ['TQ2_0', 256, 66],
}

export const isGguf = (b: Uint8Array) => b.length >= 4 && b[0] === 0x47 && b[1] === 0x47 && b[2] === 0x55 && b[3] === 0x46

export class NeedMoreBytes extends Error { constructor() { super('GGUF header continues past the bytes read so far') } }

export function parseGguf(buf: Uint8Array, fileSize = buf.length): ModelIR {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength)
  let p = 4
  const need = (n: number) => { if (p + n > buf.length) throw new NeedMoreBytes() }
  const u32 = () => { need(4); const v = dv.getUint32(p, true); p += 4; return v }
  const u64 = () => { need(8); const v = Number(dv.getBigUint64(p, true)); p += 8; return v }
  const str = () => { const n = u64(); need(n); const s = new TextDecoder().decode(buf.subarray(p, p + n)); p += n; return s }
  const sizes: Record<number, number> = { 0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8 }
  const scalar = (t: number): string | number | boolean => {
    need(sizes[t] ?? 0)
    let v: number | boolean
    switch (t) {
      case 0: v = dv.getUint8(p); break
      case 1: v = dv.getInt8(p); break
      case 2: v = dv.getUint16(p, true); break
      case 3: v = dv.getInt16(p, true); break
      case 4: v = dv.getUint32(p, true); break
      case 5: v = dv.getInt32(p, true); break
      case 6: v = dv.getFloat32(p, true); break
      case 7: v = dv.getUint8(p) !== 0; break
      case 10: v = Number(dv.getBigUint64(p, true)); break
      case 11: v = Number(dv.getBigInt64(p, true)); break
      case 12: v = dv.getFloat64(p, true); break
      default: throw new Error(`Unknown GGUF value type ${t} at byte ${p}`)
    }
    p += sizes[t]
    return v
  }
  const value = (t: number): unknown => {
    if (t === 8) return str()
    if (t === 9) {
      const et = u32(), n = u64()
      // Skip big arrays (tokenizer vocab) without materialising them; keep the length.
      if (et !== 8 && et !== 9) { need((sizes[et] ?? 0) * n); p += (sizes[et] ?? 0) * n; return { array: n } }
      for (let i = 0; i < n; i++) value(et)
      return { array: n }
    }
    return scalar(t)
  }

  const version = u32()
  if (version < 2 || version > 3) throw new Error(`GGUF version ${version} is not supported (expected 2 or 3).`)
  const nTensors = u64(), nKv = u64()
  const kv: Record<string, unknown> = {}
  for (let i = 0; i < nKv; i++) { const k = str(); kv[k] = value(u32()) }
  const tensors: Record<string, TensorInfo> = {}
  const issues: Issue[] = []
  for (let i = 0; i < nTensors; i++) {
    const name = str()
    const nd = u32()
    const dims: number[] = []
    for (let d = 0; d < nd; d++) dims.push(u64())
    const type = u32()
    u64() // offset
    const t = GGML_TYPES[type]
    if (!t) issues.push({ level: 'warn', where: `tensor ${name}`, message: `Unknown ggml type ${type}; size estimated as 1 byte per weight.` })
    const numel = dims.reduce((a, b) => a * b, 1)
    tensors[name] = { name, dtype: `ggml:${t?.[0] ?? type}`, shape: dims.slice().reverse(), init: true, bytes: t ? Math.ceil(numel / t[1]) * t[2] : numel }
  }
  const arch = String(kv['general.architecture'] ?? 'unknown')
  const num = (k: string) => typeof kv[`${arch}.${k}`] === 'number' ? kv[`${arch}.${k}`] as number : undefined
  const llm: LlmMeta = {
    arch, name: kv['general.name'] as string | undefined,
    contextLength: num('context_length'), blockCount: num('block_count'), embeddingLength: num('embedding_length'),
    headCount: num('attention.head_count'), headCountKv: num('attention.head_count_kv'),
    fileType: typeof kv['general.file_type'] === 'number' ? FILE_TYPES[kv['general.file_type'] as number] : undefined,
  }
  const meta: Record<string, string> = {}
  for (const [k, v] of Object.entries(kv)) if (typeof v !== 'object') meta[k] = String(v)
  return { format: 'gguf', producer: `GGUF v${version}`, opset: [], graphName: llm.name ?? arch, inputs: [], outputs: [], tensors, nodes: [], issues, meta, fileSize, llm }
}

const FILE_TYPES: Record<number, string> = {
  0: 'ALL_F32', 1: 'MOSTLY_F16', 2: 'MOSTLY_Q4_0', 3: 'MOSTLY_Q4_1', 7: 'MOSTLY_Q8_0', 8: 'MOSTLY_Q5_0', 9: 'MOSTLY_Q5_1',
  10: 'MOSTLY_Q2_K', 11: 'MOSTLY_Q3_K_S', 12: 'MOSTLY_Q3_K_M', 13: 'MOSTLY_Q3_K_L', 14: 'MOSTLY_Q4_K_S', 15: 'MOSTLY_Q4_K_M',
  16: 'MOSTLY_Q5_K_S', 17: 'MOSTLY_Q5_K_M', 18: 'MOSTLY_Q6_K', 32: 'MOSTLY_BF16',
}

// ---------- Builder (tests) ----------

export function buildGguf(kv: [string, 'u32' | 'str' | 'f32', number | string][], tensors: { name: string; dims: number[]; type: number }[]): Uint8Array {
  const out: number[] = [0x47, 0x47, 0x55, 0x46]
  const u32 = (v: number) => { const b = new Uint8Array(4); new DataView(b.buffer).setUint32(0, v, true); out.push(...b) }
  const u64 = (v: number) => { const b = new Uint8Array(8); new DataView(b.buffer).setBigUint64(0, BigInt(v), true); out.push(...b) }
  const str = (s: string) => { const b = new TextEncoder().encode(s); u64(b.length); out.push(...b) }
  u32(3); u64(tensors.length); u64(kv.length)
  for (const [k, t, v] of kv) {
    str(k)
    if (t === 'u32') { u32(4); u32(v as number) }
    else if (t === 'f32') { u32(6); const b = new Uint8Array(4); new DataView(b.buffer).setFloat32(0, v as number, true); out.push(...b) }
    else { u32(8); str(v as string) }
  }
  let off = 0
  for (const t of tensors) { str(t.name); u32(t.dims.length); t.dims.forEach(u64); u32(t.type); u64(off); off += 1024 }
  return Uint8Array.from(out)
}
