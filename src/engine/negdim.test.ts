// Negative dims in value_info are unknown sizes (issue #15).
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { analyze, DEVICES, symbols, type FixId } from './analyze.ts'
import { buildOnnx, parseOnnx, pinDims } from './onnx.ts'
import type { Dim } from './types.ts'

const model = (x: Dim[], y: Dim[] = ['?', 64, 8, 8]) => buildOnnx({
  name: 't', inputs: [{ name: 'x', dtype: 'fp32', shape: x }], outputs: [{ name: 'y', dtype: 'fp32', shape: y }],
  inits: [{ name: 'w', dtype: 'fp32', dims: [64, 16, 3, 3] }],
  nodes: [{ op: 'Conv', name: 'conv', inputs: ['x', 'w'], outputs: ['y'], attrs: { kernel_shape: [3, 3], pads: [1, 1, 1, 1] } }],
})
const run = (b: Uint8Array, applied: FixId[] = []) => analyze(parseOnnx(b), { pins: {}, applied, device: DEVICES[0] })

test('-1 is dynamic, offered for pinning, and its MACs use the pin: 8*8*64 * 16*9', () => {
  const a = run(model([-1, 16, 8, 8]))
  assert.deepEqual(a.symbols, ['?'])
  assert.equal(a.macs, 589_824)
  assert.ok(a.findings.some(f => f.id === 'dynamic' && f.severity === 'blocker'))
  assert.ok(a.available.includes('pin-dims'))
  assert.equal(a.score, 0)
})

test('it analyses exactly like a dim with neither value nor name', () => {
  for (const fix of [[], ['pin-dims']] as FixId[][]) assert.deepEqual(run(model([-1, 16, 8, 8]), fix), run(model(['?', 16, 8, 8]), fix))
  assert.deepEqual(run(model([-1, 16, 8, 8], [-1, 64, 8, 8])), run(model(['?', 16, 8, 8])))
})

test('the pinned download fixes it and leaves outputs alone', () => {
  const ir = parseOnnx(pinDims(model([-1, 16, -1, 8], [-1, 64, -1, 8]), { '?': 4 }))
  assert.deepEqual(ir.tensors.x.shape, [4, 16, 4, 8])
  assert.deepEqual(ir.tensors.y.shape, ['?', 64, '?', 8])
  assert.deepEqual(symbols(ir), [])
})

test('static sizes, zero and names are untouched', () => {
  assert.deepEqual(parseOnnx(model([1, 0, 'h', 8])).tensors.x.shape, [1, 0, 'h', 8])
  const b = model([1, 16, 8, 8], [1, 64, 8, 8])
  assert.deepEqual(pinDims(b, { '?': 4 }), b)
})
