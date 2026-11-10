import { test } from 'node:test'
import assert from 'node:assert/strict'
import { Reader, rewrite, Writer } from './pb.ts'
import { buildOnnx, parseOnnx, pinDims, type BuildGraph } from './onnx.ts'
import { buildFlatbuffer, parseTflite } from './tflite.ts'
import { buildGguf, NeedMoreBytes, parseGguf } from './gguf.ts'
import { parseModel, detectFormat } from './load.ts'
import { analyze, compare, DEVICES, inferShapes, opReportCsv, suggestPin, summarize, topoOrder, type FixId } from './analyze.ts'
import { cnnClassifier, detector, transformerBlock } from './samples.ts'

const dev = DEVICES[0]
const run = (bytes: Uint8Array, applied: FixId[] = [], pins: Record<string, number> = {}, device = dev) => analyze(parseOnnx(bytes), { pins, applied, device })

function tiny(nodes: BuildGraph['nodes'], extra: Partial<BuildGraph> = {}): Uint8Array {
  return buildOnnx({ name: 't', inputs: [{ name: 'x', dtype: 'fp32', shape: [1, 8] }], outputs: [{ name: 'y', dtype: 'fp32', shape: [1, 8] }], inits: [], nodes, ...extra })
}

// ---------- protobuf ----------

test('varints round-trip, including negative int64 and values past 2^32', () => {
  const vals = [0, 1, 127, 128, 300, 2 ** 31, 2 ** 40 + 5, -1, -12345]
  const w = new Writer()
  vals.forEach(v => w.int(1, v))
  const r = new Reader(w.finish())
  assert.deepEqual([...r.fields()].map(f => f.n), vals)
})

test('repeated ints decode in both packed and unpacked form', () => {
  const b = new Writer().packedInts(1, [3, 4, 5]).int(1, 6).int(1, 7).finish()
  const r = new Reader(b)
  const out: number[] = []
  for (const f of r.fields()) r.ints(f, out)
  assert.deepEqual(out, [3, 4, 5, 6, 7])
})

test('rewrite replaces one field and keeps others byte-exact', () => {
  const b = new Writer().int(1, 42).str(2, 'old').str(3, 'keep').f32(4, 1.5).finish()
  const out = rewrite(b, 0, b.length, f => f.no === 2 ? new TextEncoder().encode('new value') : null)
  const r = new Reader(out)
  const got = [...r.fields()].map(f => f.wt === 2 ? r.str(f) : f.wt === 5 ? r.f32(f) : f.n)
  assert.deepEqual(got, [42, 'new value', 'keep', 1.5])
})

test('truncated message reports the byte offset', () => {
  const b = new Writer().str(2, 'hello').finish().slice(0, 4)
  assert.throws(() => [...new Reader(b).fields()], /byte 0/)
})

// ---------- ONNX ----------

test('CNN sample parses: opset, io, initializers, producer, metadata', () => {
  const ir = parseOnnx(cnnClassifier())
  assert.equal(ir.nodes.length, 12)
  assert.deepEqual(ir.opset, [{ domain: '', version: 17 }])
  assert.deepEqual(ir.inputs, ['image'])
  assert.deepEqual(ir.outputs, ['probs'])
  assert.equal(Object.values(ir.tensors).filter(t => t.init).length, 8)
  assert.match(ir.producer, /synthetic/)
  assert.equal(ir.meta.synthetic, 'true')
  assert.deepEqual(ir.issues, [])
})

test('shape inference through conv, pool, global pool, flatten and gemm', () => {
  const ir = parseOnnx(cnnClassifier())
  const s = inferShapes(ir)
  assert.deepEqual(s.c1, [1, 16, 64, 64])
  assert.deepEqual(s.p1, [1, 16, 32, 32])
  assert.deepEqual(s.c3, [1, 64, 16, 16])
  assert.deepEqual(s.g, [1, 64, 1, 1])
  assert.deepEqual(s.f, [1, 64])
  assert.deepEqual(s.probs, [1, 10])
})

test('CNN MACs and parameter count match hand calculation', () => {
  const a = run(cnnClassifier())
  const conv = (h: number, co: number, ci: number) => h * h * co * ci * 9
  assert.equal(a.macs, conv(64, 16, 3) + conv(32, 32, 16) + conv(16, 64, 32) + 64 * 10)
  assert.equal(a.params, (16 * 27 + 16) + (32 * 144 + 32) + (64 * 288 + 64) + (640 + 10))
  assert.equal(a.weightBytes, a.params * 4)
  assert.equal(a.dtypes.fp32.params, a.params)
})

test('fully supported float CNN: one NPU partition, no switches, quantize suggested', () => {
  const a = run(cnnClassifier())
  assert.equal(a.cpuNodes, 0)
  assert.equal(a.partitions, 1)
  assert.equal(a.switches, 0)
  assert.equal(a.quant.verdict, 'float')
  assert.equal(a.quant.precision, 'fp16')
  assert.deepEqual(a.available, ['quantize-int8'])
  const q = run(cnnClassifier(), ['quantize-int8'])
  assert.equal(q.quant.precision, 'int8')
  assert.ok(q.latency.totalMs < a.latency.totalMs)
  assert.ok(q.score > a.score)
})

test('peak activation memory follows liveness, not the sum of all tensors', () => {
  const a = run(cnnClassifier())
  // Largest moment: relu1 reads conv1's output while writing its own (16x64x64 fp32 each); the image is already freed.
  const c1 = 16 * 64 * 64 * 4
  assert.equal(a.peakBytes, 2 * c1)
  assert.ok(a.peakBytes < a.nodes.reduce((s, n) => s + (n.outShape ? n.outShape.reduce<number>((p, d) => p * (d as number), 4) : 0), 0))
  assert.equal(a.largest?.name, 'c1')
})

test('symbolic dims propagate through reshape, transpose and attention matmuls', () => {
  const s = inferShapes(parseOnnx(transformerBlock()))
  assert.deepEqual(s.q4, ['batch', 'seq', 4, 32])
  assert.deepEqual(s.kt, ['batch', 4, 32, 'seq'])
  assert.deepEqual(s.s, ['batch', 4, 'seq', 'seq'])
  assert.deepEqual(s.m, ['batch', 'seq', 128])
  assert.deepEqual(s.y, ['batch', 'seq', 128])
})

test('dynamic sequence length sends the transformer to the CPU until dims are pinned', () => {
  const a = run(transformerBlock())
  assert.equal(a.npuNodes, 0)
  assert.equal(a.findings[0].id, 'dynamic')
  assert.equal(a.findings[0].severity, 'blocker')
  assert.deepEqual(a.symbols, ['batch', 'seq'])
  const p = run(transformerBlock(), ['pin-dims'], { seq: 64 })
  assert.equal(p.cpuNodes, 0)
  assert.equal(p.partitions, 1)
  // Attention scores scale with seq^2: MACs at seq=64 = 4 heads * 64 * 64 * 32 for QK^T.
  assert.equal(p.nodes.find(n => n.name === 'scores')!.macs, 4 * 64 * 64 * 32)
})

test('pinDims writes a valid model with fixed input dims and untouched weights', () => {
  const src = transformerBlock()
  const fixed = pinDims(src, { batch: 1, seq: 128 })
  const ir = parseOnnx(fixed), orig = parseOnnx(src)
  assert.deepEqual(ir.tensors.x.shape, [1, 128, 128])
  assert.deepEqual(ir.tensors.y.shape, [1, 128, 128])
  assert.equal(ir.nodes.length, orig.nodes.length)
  assert.deepEqual(Object.keys(ir.tensors).filter(k => ir.tensors[k].init).sort(), Object.keys(orig.tensors).filter(k => orig.tensors[k].init).sort())
  assert.equal(analyze(ir, { pins: {}, applied: [], device: dev }).cpuNodes, 0)
  assert.ok(Math.abs(fixed.length - src.length) < 32)
})

test('detector: unsupported ops create 5 partitions; each fix lowers the count', () => {
  const base = run(detector())
  assert.equal(base.partitions, 5)
  assert.equal(base.switches, 6)
  assert.deepEqual(base.nodes.filter(n => n.status === 'cpu' && n.support === 'cpu').map(n => n.op).sort(), ['Einsum', 'Mod', 'NonMaxSuppression'])
  const r = run(detector(), ['replace-ops'])
  assert.equal(r.partitions, 2)
  const s = run(detector(), ['replace-ops', 'split-post'])
  assert.equal(s.partitions, 1)
  assert.equal(s.cpuNodes, 0)
  assert.ok(s.latency.totalMs < r.latency.totalMs && r.latency.totalMs < base.latency.totalMs)
})

test('Einsum shape and MACs come from the equation', () => {
  const a = run(detector())
  const e = a.nodes.find(n => n.op === 'Einsum')!
  assert.deepEqual(e.outShape, [1, 64, 256])
  assert.equal(e.macs, 64 * 256 * 64)
})

test('topological order fixes nodes listed out of order', () => {
  const ir = parseOnnx(tiny([{ op: 'Relu', inputs: ['a'], outputs: ['y'] }, { op: 'Sigmoid', inputs: ['x'], outputs: ['a'] }]))
  assert.deepEqual(topoOrder(ir), [1, 0])
  assert.deepEqual(inferShapes(ir).y, [1, 8])
})

test('external data: weights sized from dims, flagged, not required', () => {
  const b = tiny([{ op: 'MatMul', inputs: ['x', 'W'], outputs: ['y'] }], { inits: [{ name: 'W', dtype: 'fp32', dims: [8, 8], external: true }] })
  const ir = parseOnnx(b)
  assert.equal(ir.tensors.W.external, true)
  const a = analyze(ir, { pins: {}, applied: [], device: dev })
  assert.equal(a.externalBytes, 256)
  assert.ok(a.findings.some(f => f.id === 'external'))
})

test('initializers repeated as graph inputs (old exporters) are not model inputs', () => {
  const b = buildOnnx({ name: 't', inputs: [{ name: 'x', dtype: 'fp32', shape: [1, 4] }, { name: 'W', dtype: 'fp32', shape: [4, 4] }], outputs: [{ name: 'y', dtype: 'fp32', shape: [1, 4] }], inits: [{ name: 'W', dtype: 'fp32', dims: [4, 4] }], nodes: [{ op: 'MatMul', inputs: ['x', 'W'], outputs: ['y'] }] })
  assert.deepEqual(parseOnnx(b).inputs, ['x'])
})

test('validation reports dangling inputs with the node index', () => {
  const ir = parseOnnx(tiny([{ op: 'Relu', inputs: ['x'], outputs: ['a'] }, { op: 'Add', name: 'bad_add', inputs: ['a', 'ghost'], outputs: ['y'] }]))
  const e = ir.issues.find(i => i.level === 'error')!
  assert.equal(e.where, 'node 1 (bad_add)')
  assert.match(e.message, /ghost/)
})

test('format detection and helpful errors for the wrong file', () => {
  assert.equal(detectFormat(cnnClassifier(), 'm.onnx'), 'onnx')
  assert.throws(() => parseModel(new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9]), 'model.onnx.data'), /external weights/)
  assert.throws(() => parseModel(new Uint8Array([0x80, 1, 2, 3, 4, 5, 6, 7, 8]), 'model.pt'), /Export to ONNX/)
  assert.throws(() => parseOnnx(new Writer().str(2, 'x').finish()), /Not an ONNX model/)
})

test('QDQ model with per-channel scales counts as int8 and per-channel', () => {
  const b = tiny([
    { op: 'DequantizeLinear', inputs: ['Wq', 'Ws', 'Wz'], outputs: ['W'], attrs: { axis: 0 } },
    { op: 'QuantizeLinear', inputs: ['x', 'xs', 'xz'], outputs: ['xq'] },
    { op: 'DequantizeLinear', inputs: ['xq', 'xs', 'xz'], outputs: ['xd'] },
    { op: 'MatMul', inputs: ['xd', 'W'], outputs: ['y'] },
  ], { inits: [{ name: 'Wq', dtype: 'int8', dims: [8, 8] }, { name: 'Ws', dtype: 'fp32', dims: [8] }, { name: 'Wz', dtype: 'int8', dims: [8] }, { name: 'xs', dtype: 'fp32', dims: [] }, { name: 'xz', dtype: 'int8', dims: [] }] })
  const a = run(b)
  assert.equal(a.quant.qdq, 3)
  assert.equal(a.quant.perChannel, true)
  assert.equal(a.quant.verdict, 'quantized-per-channel')
  assert.equal(a.quant.precision, 'int8')
  assert.equal(a.dtypes.int8.params, 64 + 8 + 1)
})

test('constraints: cubic Resize and custom-domain ops fall back with a reason', () => {
  const a = run(tiny([
    { op: 'Resize', inputs: ['x'], outputs: ['r'], attrs: { mode: 'cubic' } },
    { op: 'MyOp', domain: 'com.acme', inputs: ['r'], outputs: ['y'] },
  ], { extraOpsets: [{ domain: 'com.acme', version: 1 }] }))
  assert.equal(a.nodes[0].status, 'cpu')
  assert.match(a.nodes[0].reason, /Cubic/)
  assert.match(a.nodes[1].reason, /com\.acme/)
})

test('roofline: a big MatMul is compute-bound and matches the formula', () => {
  const N = 1024
  const b = buildOnnx({ name: 'mm', inputs: [{ name: 'x', dtype: 'fp32', shape: [N, N] }], outputs: [{ name: 'y', dtype: 'fp32', shape: [N, N] }], inits: [{ name: 'W', dtype: 'fp32', dims: [N, N], external: true }], nodes: [{ op: 'MatMul', inputs: ['x', 'W'], outputs: ['y'] }] })
  const d = { ...dev, npuTops: 10, util: 1, memGBs: 1000 }
  const a = run(b, [], {}, d)
  const expected = 2 * N ** 3 / (10e12 * 0.5) * 1000
  assert.ok(Math.abs(a.latency.totalMs - expected) / expected < 1e-9)
  assert.ok(a.latency.computeShare > 0.99)
})

// ---------- TFLite ----------

function tfliteModel(): Uint8Array {
  const tensor = (name: string, shape: number[], type: number, buffer: number, quant?: number[], sig?: number[]) =>
    ({ tbl: [{ vi32: shape }, { u8: type }, { u32: buffer }, { str: name }, quant ? { tbl: [undefined, undefined, { vf32: quant }, undefined, undefined, undefined, { i32: 0 }] } : undefined, undefined, undefined, sig ? { vi32: sig } : undefined] })
  return buildFlatbuffer([
    { u32: 3 },
    { vtbl: [[{ u8: 3 }, undefined, { i32: 1 }, { i32: 3 }], [{ u8: 9 }, undefined, { i32: 1 }, { i32: 9 }], [{ u8: 32 }, { str: 'MyPostProcess' }, { i32: 1 }, { i32: 32 }]] },
    { vtbl: [[
      { vtbl: [tensor('input', [1, 32, 32, 3], 9, 0, [0.02], [-1, 32, 32, 3]).tbl, tensor('conv_w', [8, 3, 3, 3], 9, 1, Array(8).fill(0.01)).tbl, tensor('conv_out', [1, 32, 32, 8], 9, 0, [0.05]).tbl, tensor('fc_w', [10, 8192], 9, 2, [0.01]).tbl, tensor('logits', [1, 10], 9, 0, [0.1]).tbl, tensor('final', [1, 10], 0, 0).tbl] },
      { vi32: [0] }, { vi32: [5] },
      { vtbl: [[{ u32: 0 }, { vi32: [0, 1] }, { vi32: [2] }], [{ u32: 1 }, { vi32: [2, 3] }, { vi32: [4] }], [{ u32: 2 }, { vi32: [4] }, { vi32: [5] }]] },
      { str: 'main' },
    ]] },
    { str: 'synthetic test model' },
    { vtbl: [[], [{ vu8: new Uint8Array(216) }], [{ vu8: new Uint8Array(81920) }]] },
  ])
}

test('TFLite flatbuffer: ops mapped, weights sized from buffers, per-channel quant detected', () => {
  const ir = parseTflite(tfliteModel())
  assert.equal(detectFormat(tfliteModel()), 'tflite')
  assert.deepEqual(ir.nodes.map(n => n.op), ['Conv', 'Gemm', 'MyPostProcess'])
  assert.equal(ir.nodes[2].domain, 'custom')
  assert.equal(ir.tensors.conv_w.bytes, 216)
  assert.equal(ir.tensors.fc_w.init, true)
  assert.deepEqual(ir.tensors.input.shape, ['batch', 32, 32, 3])
  assert.equal(ir.meta.description, 'synthetic test model')
  const a = analyze(ir, { pins: {}, applied: ['pin-dims'], device: dev })
  assert.equal(a.quant.perChannel, true)
  assert.equal(a.quant.precision, 'int8')
  assert.equal(a.nodes[2].status, 'cpu')
})

test('TFLite NHWC conv and fully-connected MACs', () => {
  const a = analyze(parseTflite(tfliteModel()), { pins: {}, applied: ['pin-dims'], device: dev })
  assert.equal(a.nodes[0].macs, 32 * 32 * 8 * 3 * 3 * 3)
  assert.equal(a.nodes[1].macs, 10 * 8192)
})

// ---------- GGUF ----------

function ggufModel() {
  return buildGguf(
    [['general.architecture', 'str', 'llama'], ['general.name', 'str', 'Synthetic 1B'], ['llama.block_count', 'u32', 16], ['llama.context_length', 'u32', 8192],
      ['llama.embedding_length', 'u32', 2048], ['llama.attention.head_count', 'u32', 32], ['llama.attention.head_count_kv', 'u32', 8], ['general.file_type', 'u32', 15]],
    [{ name: 'token_embd.weight', dims: [2048, 32000], type: 12 }, { name: 'blk.0.attn_q.weight', dims: [2048, 2048], type: 12 }, { name: 'output_norm.weight', dims: [2048], type: 0 }],
  )
}

test('GGUF header: metadata, k-quant block sizes and LLM estimates', () => {
  const ir = parseGguf(ggufModel())
  assert.equal(ir.llm?.arch, 'llama')
  assert.equal(ir.llm?.headCountKv, 8)
  assert.equal(ir.llm?.fileType, 'MOSTLY_Q4_K_M')
  assert.equal(ir.tensors['token_embd.weight'].bytes, 2048 * 32000 / 256 * 144)
  assert.deepEqual(ir.tensors['token_embd.weight'].shape, [32000, 2048])
  const a = analyze(ir, { pins: {}, applied: [], device: dev })
  assert.equal(a.dtypes.int4.params, 2048 * 32000 + 2048 * 2048)
  assert.equal(a.dtypes.fp32.bytes, 2048 * 4)
  // KV cache: 2 (k,v) * 16 blocks * 4096 tokens * 2048 * (8/32) * 2 bytes
  assert.equal(a.latency.kvCacheBytes, 2 * 16 * 4096 * 2048 * 0.25 * 2)
  assert.ok(Math.abs(a.latency.tokensPerSec! - dev.memGBs * 1e9 * dev.util * 2 / a.weightBytes) < 1e-6)
})

test('GGUF reader asks for more bytes when the header is cut off', () => {
  assert.throws(() => parseGguf(ggufModel().slice(0, 60)), NeedMoreBytes)
})

// ---------- Reports and comparison ----------

test('op report CSV has one row per op type and escapes commas', () => {
  const csv = opReportCsv(run(detector()))
  const lines = csv.trim().split('\n')
  assert.equal(lines[0], 'op,domain,count,support,on_npu,on_cpu,macs,note,replacement')
  assert.equal(lines.length, 1 + new Set(parseOnnx(detector()).nodes.map(n => n.op)).size)
  assert.ok(lines.some(l => l.startsWith('Einsum,ai.onnx,1,cpu,0,1,1048576,')))
  assert.match(csv, /"Express as Sub\(a, Mul/)
})

test('version comparison reports fixed findings and which metrics improved', () => {
  const before = summarize(run(detector()))
  const after = summarize(run(detector(), ['replace-ops', 'split-post']))
  const d = compare(before, after)
  assert.ok(d.fixed.some(t => t.startsWith('Einsum')))
  assert.deepEqual(d.introduced, [])
  assert.equal(d.metrics.find(m => m.key === 'partitions')!.better, true)
  assert.equal(d.metrics.find(m => m.key === 'params')!.better, null)
})

test('suggested pins: batch 1, sequence 128, image 224', () => {
  assert.equal(suggestPin('batch_size'), 1)
  assert.equal(suggestPin('sequence_length'), 128)
  assert.equal(suggestPin('height'), 224)
  assert.equal(suggestPin('unk__42'), 1)
})
