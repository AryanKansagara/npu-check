// Minimal protobuf wire-format reader and writer. Enough for ONNX: varints, fixed32/64,
// length-delimited fields, packed repeated scalars, and byte-exact field rewriting.

export const WT = { Varint: 0, I64: 1, Len: 2, I32: 5 } as const

export interface Field {
  no: number
  wt: number
  /** varint value (as number; int64 above 2^53 loses precision, fine for dims and enums) */
  n: number
  /** start/end of the payload for Len fields, of the value for fixed fields */
  start: number
  end: number
  /** start of the tag, so a whole field can be copied byte-for-byte */
  tagStart: number
}

export class Reader {
  pos: number
  buf: Uint8Array
  end: number
  dv: DataView
  constructor(buf: Uint8Array, start = 0, end = buf.length) {
    this.buf = buf
    this.pos = start
    this.end = end
    this.dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength)
  }
  varint(): number {
    const s = this.pos
    let v = 0, mul = 1, b: number
    do {
      if (this.pos >= this.end) throw new Error(`Truncated varint at byte ${s}`)
      b = this.buf[this.pos++]
      v += (b & 0x7f) * mul
      mul *= 128
    } while (b & 0x80)
    if (this.pos - s > 7) {
      // Past 49 bits: recompute exactly so negative int64 (ten-byte encodings) come out right.
      let big = 0n
      for (let i = this.pos - 1; i >= s; i--) big = (big << 7n) | BigInt(this.buf[i] & 0x7f)
      return Number(BigInt.asIntN(64, big))
    }
    return v
  }
  /** Iterates the fields of the message in [pos, end). */
  *fields(): Generator<Field> {
    while (this.pos < this.end) {
      const tagStart = this.pos
      const tag = this.varint()
      const no = tag >>> 3, wt = tag & 7
      if (no === 0) throw new Error(`Invalid field number 0 at byte ${tagStart}`)
      let n = 0, start = this.pos, end = this.pos
      if (wt === WT.Varint) { n = this.varint(); end = this.pos }
      else if (wt === WT.I64) { start = this.pos; this.pos += 8; end = this.pos }
      else if (wt === WT.I32) { start = this.pos; this.pos += 4; end = this.pos }
      else if (wt === WT.Len) {
        const len = this.varint()
        start = this.pos
        end = start + len
        if (end > this.end) throw new Error(`Field ${no} at byte ${tagStart} runs past the end of its message (${len} bytes declared)`)
        this.pos = end
      } else throw new Error(`Unsupported wire type ${wt} at byte ${tagStart}`)
      yield { no, wt, n, start, end, tagStart }
    }
  }
  sub(f: Field) { return new Reader(this.buf, f.start, f.end) }
  str(f: Field) { return dec.decode(this.buf.subarray(f.start, f.end)) }
  f32(f: Field) { return this.dv.getFloat32(f.start, true) }
  f64(f: Field) { return this.dv.getFloat64(f.start, true) }
  /** Repeated varint field that may be packed (Len) or not (Varint). */
  ints(f: Field, out: number[]) {
    if (f.wt === WT.Varint) { out.push(f.n); return }
    const r = this.sub(f)
    while (r.pos < r.end) out.push(r.varint())
  }
  floats(f: Field, out: number[]) {
    if (f.wt === WT.I32) { out.push(this.f32(f)); return }
    for (let p = f.start; p + 4 <= f.end; p += 4) out.push(this.dv.getFloat32(p, true))
  }
}

const dec = new TextDecoder()
const enc = new TextEncoder()

export class Writer {
  private chunks: number[] = []
  private parts: Uint8Array[] = []
  private flush() { if (this.chunks.length) { this.parts.push(Uint8Array.from(this.chunks)); this.chunks = [] } }
  varintRaw(v: number) {
    if (v < 0) {
      // Two's complement 64-bit, ten bytes.
      let big = BigInt.asUintN(64, BigInt(v))
      for (let i = 0; i < 10; i++) { const b = Number(big & 0x7fn); big >>= 7n; this.chunks.push(i === 9 ? b : b | 0x80) }
      return this
    }
    while (v >= 0x80) { this.chunks.push((v % 128) | 0x80); v = Math.floor(v / 128) }
    this.chunks.push(v)
    return this
  }
  tag(no: number, wt: number) { return this.varintRaw(no * 8 + wt) }
  int(no: number, v: number) { return this.tag(no, WT.Varint).varintRaw(v) }
  bytes(no: number, b: Uint8Array) { this.tag(no, WT.Len).varintRaw(b.length); this.flush(); this.parts.push(b); return this }
  str(no: number, s: string) { return this.bytes(no, enc.encode(s)) }
  msg(no: number, w: Writer) { return this.bytes(no, w.finish()) }
  f32(no: number, v: number) { this.tag(no, WT.I32); const b = new Uint8Array(4); new DataView(b.buffer).setFloat32(0, v, true); this.flush(); this.parts.push(b); return this }
  packedInts(no: number, vs: number[]) { const w = new Writer(); vs.forEach(v => w.varintRaw(v)); return this.bytes(no, w.finish()) }
  raw(b: Uint8Array) { this.flush(); this.parts.push(b); return this }
  finish(): Uint8Array {
    this.flush()
    const len = this.parts.reduce((s, p) => s + p.length, 0)
    const out = new Uint8Array(len)
    let o = 0
    for (const p of this.parts) { out.set(p, o); o += p.length }
    return out
  }
}

/**
 * Copies a message, replacing the payload of selected length-delimited fields.
 * `fn` returns new payload bytes for a field, or null to keep it as is. Unknown fields are kept byte-exact.
 */
export function rewrite(buf: Uint8Array, start: number, end: number, fn: (f: Field, r: Reader) => Uint8Array | null): Uint8Array {
  const r = new Reader(buf, start, end)
  const w = new Writer()
  for (const f of r.fields()) {
    const next = f.wt === WT.Len ? fn(f, r) : null
    if (next) w.bytes(f.no, next)
    else w.raw(buf.subarray(f.tagStart, f.end))
  }
  return w.finish()
}
