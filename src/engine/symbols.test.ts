// Symbols that shape inference tools write into value_info (issue #6).
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { buildOnnx, parseOnnx, type BuildValue } from './onnx.ts'
import { analyze, DEVICES, inferShapes } from './analyze.ts'

const model = (valueInfo: BuildValue[], out: BuildValue['shape']) => parseOnnx(buildOnnx({
  name: 't', inputs: [{ name: 'x', dtype: 'fp32', shape: ['batch', 3, 'height', 'width'] }], outputs: [{ name: 'y', dtype: 'fp32', shape: out }],
  inits: [{ name: 'w', dtype: 'fp32', dims: [16, 3, 3, 3] }, { name: 'w2', dtype: 'fp32', dims: [32, 16, 3, 3] }], valueInfo,
  nodes: [{ op: 'Conv', inputs: ['x', 'w'], outputs: ['c'], attrs: { strides: [2, 2], pads: [1, 1, 1, 1] } }, { op: 'Conv', inputs: ['c', 'w2'], outputs: ['y'], attrs: { pads: [1, 1, 1, 1] } }],
}))
const vi = (shape: BuildValue['shape']) => [{ name: 'c', dtype: 'fp32' as const, shape }]
const gen = model(vi(['batch', 16, 'unk__0', 'unk__1']), ['batch', 32, 'unk__2', 'unk__3'])

test('generated symbols take the size that follows from the pinned inputs', () => {
  // 224 / 2 = 112: 112*112*16*27 + 112*112*32*144.
  assert.equal(analyze(gen, { pins: {}, applied: [], device: DEVICES[0] }).macs, 63_221_760)
  const pins = { batch: 2, height: 32, width: 64 }
  assert.deepEqual(inferShapes(gen, pins).c, [2, 16, 16, 32])
  assert.deepEqual(inferShapes(gen, pins).y, [2, 32, 16, 32])
})

test('without value_info the result is the same', () => {
  assert.deepEqual(inferShapes(model([], ['batch', 32, '?', '?']), { batch: 1, height: 224, width: 224 }).y, [1, 32, 112, 112])
})

test('unpinned shapes keep the declared names, and static declared dims win', () => {
  assert.deepEqual(inferShapes(gen).c, ['batch', 16, 'unk__0', 'unk__1'])
  assert.deepEqual(inferShapes(model(vi(['batch', 16, 100, 'unk__1']), ['batch', 32, '?', '?']), { batch: 1, height: 224, width: 224 }).c, [1, 16, 100, 112])
})

test('a wrong rank or an op without a rule keeps the old fallback', () => {
  const ir = (op: string) => parseOnnx(buildOnnx({ name: 't', inputs: [{ name: 'x', dtype: 'fp32', shape: ['batch', 8] }], outputs: [{ name: 'y', dtype: 'fp32', shape: ['batch', 'n', 4] }], inits: [], nodes: [{ op, inputs: ['x'], outputs: ['y'] }] }))
  for (const op of ['Relu', 'GridSample']) assert.deepEqual(inferShapes(ir(op), { batch: 3 }).y, [3, 1, 4])
})
