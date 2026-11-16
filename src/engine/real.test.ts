// Checks against real public model files (see public/samples and the Docs page for sources and licences)
// plus regressions found while running public ONNX / TFLite / GGUF models through the engine.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { buildOnnx, parseOnnx, pinDims, type BuildGraph } from './onnx.ts'
import { buildGguf, parseGguf } from './gguf.ts'
import { TFL_NAMES } from './tflite.ts'
import { parseModel } from './load.ts'
import { analyze, defaultPin, DEVICES, inferShapes, type FixId } from './analyze.ts'
import { QNN_EP_LISTED, rule } from './support.ts'

const dev = DEVICES[0]
const load = (rel: string) => parseModel(new Uint8Array(readFileSync(new URL(rel, import.meta.url))), rel)
const sample = (name: string) => load(`../../public/samples/${name}`)
const run = (ir: ReturnType<typeof parseOnnx>, applied: FixId[] = [], pins: Record<string, number> = {}) => analyze(ir, { pins, applied, device: dev })
const i64 = (name: string, data: number[], dims = [data.length]) => ({ name, dtype: 'int64' as const, dims, data })

test('MNIST-12 (ONNX Model Zoo): 5,998 parameters, static shapes, one NPU partition', () => {
  const ir = sample('mnist-12.onnx')
  const a = run(ir)
  assert.equal(ir.producer, 'CNTK 2.5.1')
  assert.equal(a.opset, 12)
  assert.equal(a.nodeCount, 12)
  assert.equal(a.params, 5998)
  assert.deepEqual(a.inputs[0].shape, [1, 1, 28, 28])
  assert.deepEqual(a.outputs[0].shape, [1, 10])
  // Conv1 8x1x5x5 on 28x28 (SAME), Conv2 16x8x5x5 on 14x14, MatMul 256 -> 10.
  assert.equal(a.macs, 28 * 28 * 8 * 25 + 14 * 14 * 16 * 8 * 25 + 256 * 10)
  assert.equal(a.partitions, 1)
  assert.equal(a.cpuNodes, 0)
  assert.equal(ir.issues.length, 0)
})

test('Qualcomm AI Hub MobileNet-v2 w8a8: published 3.49 M parameters and 300 M MACs are reproduced', () => {
  const ir = sample('mobilenet_v2_w8a8.onnx')
  const a = run(ir)
  assert.ok(ir.producer.startsWith('Qualcomm AI Hub'))
  assert.ok(Math.abs(a.params - 3.49e6) / 3.49e6 < 0.02, `params ${a.params}`)
  assert.ok(Math.abs(a.macs - 300e6) / 300e6 < 0.01, `macs ${a.macs}`)
  assert.deepEqual(a.inputs[0].shape, [1, 3, 224, 224])
  assert.equal(a.inputs[0].dtype, 'uint8')
  assert.ok(a.externalBytes > 13e6)
})

test('QDQ export: weight Q/DQ chains fold, activations count at int8, the graph is one partition', () => {
  const a = run(sample('mobilenet_v2_w8a8.onnx'))
  assert.equal(a.quant.verdict, 'quantized-per-channel')
  assert.equal(a.partitions, 1)
  assert.equal(a.cpuNodes, 0)
  assert.ok(a.npuNodes < a.nodeCount, 'folded weight DequantizeLinear nodes are not counted as runtime nodes')
  // Largest live set in MobileNet-v2 at int8 is the 96x112x112 expansion plus its neighbours: a few MB, not tens.
  assert.ok(a.peakBytes > 1.2e6 && a.peakBytes < 4e6, `peak ${a.peakBytes}`)
  assert.ok(a.findings.some(f => f.id === 'fake-quant'))
  assert.ok(!a.findings.some(f => f.id === 'fp32-left' || f.id === 'float'))
})

test('roofline estimate is within 2.5x of the published Snapdragon X Elite measurement (0.454 ms, w8a8)', () => {
  const a = run(sample('mobilenet_v2_w8a8.onnx'))
  const measured = 0.454
  assert.ok(a.latency.totalMs > measured / 2.5 && a.latency.totalMs < measured * 2.5, `estimate ${a.latency.totalMs}`)
})

test('BlazeFace TFLite: fp16 weights behind DEQUANTIZE are a float model, not a quantized one', () => {
  const ir = sample('face_detection_short_range.tflite')
  const a = run(ir)
  assert.equal(ir.format, 'tflite')
  assert.deepEqual(a.inputs[0].shape, [1, 128, 128, 3])
  assert.deepEqual(a.outputs.map(o => o.shape), [[1, 896, 16], [1, 896, 1]])
  assert.equal(a.quant.verdict, 'float')
  assert.equal(a.quant.qdq, 0)
  assert.ok(a.dtypes.fp16.params > 100_000)
  assert.equal(a.partitions, 1)
  assert.ok(!ir.nodes.some(n => /^[A-Z_0-9]+$/.test(n.op)), 'every builtin in the file maps to a known op')
})

test('TinyStories 260K GGUF: header fields, parameter count and grouped-query KV cache', () => {
  const ir = sample('stories260K.gguf')
  const a = run(ir)
  assert.deepEqual([ir.llm!.arch, ir.llm!.blockCount, ir.llm!.embeddingLength, ir.llm!.headCount, ir.llm!.headCountKv, ir.llm!.contextLength], ['llama', 5, 64, 8, 4, 2048])
  assert.equal(a.params, 292_800)
  assert.equal(a.dtypes.fp32.params, a.params)
  // 2 (K and V) x 5 layers x 2048 tokens x 64 wide x 4/8 KV heads x 2 bytes
  assert.equal(a.latency.kvCacheBytes, 2 * 5 * 2048 * 64 * 0.5 * 2)
  assert.ok(a.weightBytes <= ir.fileSize && a.weightBytes > ir.fileSize * 0.95)
})

test('Phi-3 mini int4 graph (ONNX Runtime GenAI): unnamed optional outputs are not duplicate-output errors', () => {
  const ir = load('./fixtures/phi3-mini-4k-int4-graph.onnx')
  assert.equal(ir.issues.filter(i => i.level === 'error').length, 0)
  assert.ok(ir.nodes.some(n => n.outputs.includes('')))
})

test('Phi-3 mini int4 graph: MatMulNBits blobs count as K x N int4 weights (about 3.8 B parameters)', () => {
  const a = run(load('./fixtures/phi3-mini-4k-int4-graph.onnx'))
  assert.ok(a.params > 3.7e9 && a.params < 4.0e9, `params ${a.params}`)
  assert.ok(a.dtypes.int4.params / a.params > 0.9)
  assert.equal(a.quant.verdict, 'weight-only')
  assert.ok(a.opRows.find(r => r.op === 'GroupQueryAttention')!.cpu === 32)
  assert.ok(a.macs > 1e9)
})

// Exported transformers compute Reshape targets at runtime: Shape -> Gather -> Unsqueeze -> Concat -> Reshape.
function shapeMath(): BuildGraph {
  return {
    name: 'shape_math', opset: 17,
    inputs: [{ name: 'x', dtype: 'fp32', shape: ['batch', 'seq', 64] }], outputs: [{ name: 'y', dtype: 'fp32', shape: ['batch', 'seq', 64] }],
    inits: [i64('i0', [0], []), i64('i1', [1], []), i64('ax0', [0]), i64('heads', [4]), i64('hd', [16]), i64('merge', [0, 0, 64]),
      { name: 'w', dtype: 'fp32', dims: [64, 64] }],
    nodes: [
      { op: 'Shape', inputs: ['x'], outputs: ['s'] },
      { op: 'Gather', inputs: ['s', 'i0'], outputs: ['b'], attrs: { axis: 0 } },
      { op: 'Gather', inputs: ['s', 'i1'], outputs: ['t'], attrs: { axis: 0 } },
      { op: 'Unsqueeze', inputs: ['b', 'ax0'], outputs: ['b1'] },
      { op: 'Unsqueeze', inputs: ['t', 'ax0'], outputs: ['t1'] },
      { op: 'Concat', inputs: ['b1', 't1', 'heads', 'hd'], outputs: ['target'], attrs: { axis: 0 } },
      { op: 'MatMul', name: 'proj', inputs: ['x', 'w'], outputs: ['p'] },
      { op: 'Reshape', name: 'split_heads', inputs: ['p', 'target'], outputs: ['h'] },
      { op: 'Reshape', name: 'merge_heads', inputs: ['h', 'merge'], outputs: ['y'] },
    ],
  }
}

test('shape arithmetic in the graph resolves: symbolic without pins, numeric with them', () => {
  const ir = parseOnnx(buildOnnx(shapeMath()))
  assert.deepEqual(inferShapes(ir).h, ['batch', 'seq', 4, 16])
  assert.deepEqual(inferShapes(ir, { batch: 2, seq: 128 }).h, [2, 128, 4, 16])
})

test('pinned shapes fold the Shape chain away: everything left runs on the NPU in one partition', () => {
  const ir = parseOnnx(buildOnnx(shapeMath()))
  const dyn = run(ir)
  assert.ok(dyn.cpuNodes > 0)
  assert.match(dyn.nodes.find(n => n.op === 'Shape')!.reason, /dynamic/)
  const pinned = run(ir, ['pin-dims'], { batch: 1, seq: 128 })
  assert.equal(pinned.cpuNodes, 0)
  assert.equal(pinned.partitions, 1)
  assert.equal(pinned.npuNodes, 3, 'MatMul and two Reshapes remain; six shape nodes are folded')
  assert.equal(pinned.macs, 128 * 64 * 64)
})

test('unnamed input dims: default to 224 in image slots and are pinned in the rewritten file', () => {
  const g: BuildGraph = {
    name: 'anon', opset: 11, inputs: [{ name: 'img', dtype: 'fp32', shape: ['N', 3, '?', '?'] }], outputs: [{ name: 'o', dtype: 'fp32', shape: ['N', 8, '?', '?'] }],
    inits: [{ name: 'w', dtype: 'fp32', dims: [8, 3, 3, 3] }, { name: 'scales', dtype: 'fp32', dims: [4], data: [1, 1, 2, 2] }, { name: 'roi', dtype: 'fp32', dims: [0] }],
    nodes: [
      { op: 'Conv', inputs: ['img', 'w'], outputs: ['c'], attrs: { kernel_shape: [3, 3], strides: [2, 2], pads: [1, 1, 1, 1] } },
      { op: 'Resize', inputs: ['c', 'roi', 'scales'], outputs: ['o'], attrs: { mode: 'nearest' } },
    ],
  }
  const bytes = buildOnnx(g)
  const ir = parseOnnx(bytes)
  assert.deepEqual(ir.tensors.img.shape, ['N', 3, '?', '?'])
  assert.equal(defaultPin(ir, '?'), 224)
  assert.equal(defaultPin(ir, 'N'), 1)
  const a = run(ir)
  assert.equal(a.macs, 112 * 112 * 8 * 27)
  const pinned = parseOnnx(pinDims(bytes, { N: 1, '?': 416 }))
  assert.deepEqual(pinned.tensors.img.shape, [1, 3, 416, 416])
  // Resize scales (a tiny float constant) give the upsampled shape: 416 / 2 * 2.
  assert.deepEqual(inferShapes(pinned).c, [1, 8, 208, 208])
  assert.deepEqual(run(pinned).nodes[1].outShape, [1, 8, 416, 416])
})

test('QOperator and dynamically quantized graphs are flagged for QDQ re-quantization', () => {
  for (const op of ['QLinearConv', 'QLinearMatMul', 'MatMulInteger', 'DynamicQuantizeLinear']) {
    const r = rule(op, '')
    assert.equal(r.status, 'cpu', op)
    assert.match(r.replace ?? '', /QDQ/, op)
  }
  assert.equal(rule('QLinearAdd', 'com.microsoft').status, 'cpu')
  assert.match(rule('QLinearAdd', 'com.microsoft').note, /QOperator/)
  const g: BuildGraph = {
    name: 'dynq', opset: 13, inputs: [{ name: 'x', dtype: 'fp32', shape: [1, 16] }], outputs: [{ name: 'y', dtype: 'int32', shape: [1, 8] }],
    inits: [{ name: 'w', dtype: 'int8', dims: [16, 8] }, { name: 'wzp', dtype: 'int8', dims: [] }],
    nodes: [
      { op: 'DynamicQuantizeLinear', inputs: ['x'], outputs: ['xq', 'xs', 'xzp'] },
      { op: 'MatMulInteger', inputs: ['xq', 'w', 'xzp', 'wzp'], outputs: ['y'] },
    ],
  }
  const a = run(parseOnnx(buildOnnx(g)))
  assert.ok(a.findings.some(f => f.id === 'dyn-quant' && f.severity === 'blocker'))
  assert.equal(a.cpuNodes, 2)
  assert.equal(a.macs, 8 * 16)
})

test('support table tracks the ONNX Runtime QNN EP operator list', () => {
  assert.equal(QNN_EP_LISTED.size, 77)
  for (const op of ['Conv', 'MatMul', 'LayerNormalization', 'Gelu', 'LRN', 'GridSample']) assert.ok(QNN_EP_LISTED.has(op), op)
  assert.equal(rule('LRN', '').status, 'npu')
  // In the curated table but not in the published list: kept, with lower confidence.
  assert.equal(rule('GlobalMaxPool', '').status, 'constrained')
  assert.match(rule('GlobalMaxPool', '').note, /QNN EP operator table/)
  assert.equal(rule('Reshape', '').status, 'npu')
  assert.equal(rule('TFLite_Detection_PostProcess', 'custom').post, true)
  assert.equal(rule('SomeVendorOp', 'custom').status, 'cpu')
})

test('TFLite builtin names follow schema.fbs enum values', () => {
  assert.deepEqual([TFL_NAMES[0], TFL_NAMES[3], TFL_NAMES[32], TFL_NAMES[126], TFL_NAMES[150], TFL_NAMES[209]], ['ADD', 'CONV_2D', 'CUSTOM', 'BATCH_MATMUL', 'GELU', 'STABLEHLO_CASE'])
  assert.equal(TFL_NAMES.length, 210)
  assert.match(rule('HASHTABLE_LOOKUP', '').note, /TFLite builtin/)
})

test('GGUF encoder-only models report one pass: no decode speed, no KV cache', () => {
  const ir = parseGguf(buildGguf(
    [['general.architecture', 'str', 'nomic-bert'], ['nomic-bert.block_count', 'u32', 12], ['nomic-bert.context_length', 'u32', 2048], ['nomic-bert.embedding_length', 'u32', 768], ['nomic-bert.attention.head_count', 'u32', 12], ['nomic-bert.pooling_type', 'u32', 1]],
    [{ name: 'token_embd.weight', dims: [768, 30522], type: 1 }, { name: 'blk.0.attn_q.weight', dims: [768, 768], type: 12 }],
  ))
  assert.equal(ir.llm!.encoder, true)
  const a = run(ir)
  assert.equal(a.latency.tokensPerSec, undefined)
  assert.equal(a.latency.kvCacheBytes, undefined)
  assert.ok(a.latency.totalMs > 0)
  assert.ok(!a.findings.some(f => f.id === 'kv'))
  assert.match(a.findings[0].title, /Encoder-only/)
  const dec = parseGguf(buildGguf([['general.architecture', 'str', 'llama'], ['llama.block_count', 'u32', 2], ['llama.context_length', 'u32', 512], ['llama.embedding_length', 'u32', 64]], [{ name: 'blk.0.attn_q.weight', dims: [64, 64], type: 0 }]))
  assert.equal(dec.llm!.encoder, undefined)
  assert.ok(run(dec).latency.tokensPerSec! > 0)
})
