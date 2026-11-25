// Nodes whose shapes cannot be worked out (issue #5).
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { buildOnnx, parseOnnx, type BuildGraph } from './onnx.ts'
import { parseModel } from './load.ts'
import { analyze, DEVICES, nodeReportCsv, type FixId } from './analyze.ts'
import { cnnClassifier, detector, transformerBlock } from './samples.ts'

const run = (ir: ReturnType<typeof parseOnnx>, applied: FixId[] = ['quantize-int8']) => analyze(ir, { pins: {}, applied, device: DEVICES[0] })
const g = (nodes: BuildGraph['nodes'], extra: Partial<BuildGraph> = {}) => parseOnnx(buildOnnx({ name: 't', inputs: [{ name: 'x', dtype: 'fp32', shape: [1, 16, 8, 8] }], outputs: [], inits: [{ name: 'w', dtype: 'fp32', dims: [64, 16, 3, 3] }], nodes, ...extra }))
const conv = { op: 'Conv', name: 'conv', inputs: ['h', 'w'], outputs: ['y'], attrs: { pads: [1, 1, 1, 1] } }

test('everything after an op without a shape rule is reported, not confirmed and not priced', () => {
  const a = run(g([{ op: 'GridSample', name: 'gs', inputs: ['x', 'x'], outputs: ['h'] }, conv]))
  assert.equal(a.unknownNodes, 2)
  assert.match(a.findings.find(f => f.id === 'unknown-shapes')!.detail, /gs \(GridSample\).*lower bounds/)
  assert.equal(a.score, 0)
  assert.ok(nodeReportCsv(a).trim().split('\n').slice(1).every(l => l.endsWith(',,,')))
})

test('a declared shape is enough: 8 x 8 x 64 outputs x 16 x 9 weights = 589,824 MACs', () => {
  const a = run(g([{ op: 'GridSample', inputs: ['x', 'x'], outputs: ['h'] }, conv], { valueInfo: [{ name: 'h', dtype: 'fp32', shape: [1, 16, 8, 8] }] }))
  assert.equal(a.unknownNodes, 0)
  assert.equal(a.macs, 589_824)
  assert.equal(a.score, 100)
})

test('NaN dims count as unknown; a skipped optional output does not', () => {
  const i64 = (name: string, data: number[]) => ({ name, dtype: 'int64' as const, dims: [data.length], data })
  const pad = g([{ op: 'Pad', inputs: ['x', 'p', '', 'ax'], outputs: ['y'] }], { inits: [i64('p', [1]), i64('ax', [2])] })
  assert.equal(run(pad).unknownNodes, 1)
  assert.equal(run(g([{ op: 'LSTM', inputs: ['x'], outputs: ['', 'y'] }], { valueInfo: [{ name: 'y', dtype: 'fp32', shape: [1, 1, 8] }] })).unknownNodes, 0)
})

test('samples and bundled models have no unknown nodes', () => {
  const file = (n: string) => parseModel(new Uint8Array(readFileSync(new URL(`../../public/samples/${n}`, import.meta.url))), n)
  for (const ir of [cnnClassifier(), detector(), detector(true), transformerBlock()].map(parseOnnx).concat(['mnist-12.onnx', 'mobilenet_v2_w8a8.onnx', 'face_detection_short_range.tflite', 'stories260K.gguf'].map(file))) {
    for (const fixes of [[], ['pin-dims']] as FixId[][]) assert.equal(run(ir, fixes).unknownNodes, 0, ir.graphName)
  }
})
