// TFLite convolution layouts (issue #7): numbers worked by hand from the op definitions.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { cost } from './analyze.ts'
import type { Dim, ModelIR, ModelNode } from './types.ts'

const macs = (op: string, shapes: Record<string, Dim[]>, inputs: string[], attrs: ModelNode['attrs'] = {}) =>
  cost({ name: '', op, domain: '', inputs, outputs: ['y'], attrs: { layout: 'NHWC', ...attrs } }, shapes, {} as ModelIR).macs

test('TRANSPOSE_CONV: 8*8*16 inputs x 3*3*32 weights each, with or without a bias', () => {
  const s = { os: [4], w: [32, 3, 3, 16], x: [1, 8, 8, 16], b: [32], y: [1, 16, 16, 32] }
  assert.equal(macs('ConvTranspose', s, ['os', 'w', 'x']), 294_912)
  assert.equal(macs('ConvTranspose', s, ['os', 'w', 'x', 'b']), 294_912)
})

test('CONV_3D: 4*4*4*5 outputs x 3*3*3*2 weights each', () => {
  assert.equal(macs('Conv', { x: [1, 4, 4, 4, 2], w: [3, 3, 3, 2, 5], y: [1, 4, 4, 4, 5] }, ['x', 'w']), 17_280)
})

test('CONV_2D, DEPTHWISE_CONV_2D and ONNX ConvTranspose are unchanged', () => {
  assert.equal(macs('Conv', { x: [1, 8, 8, 3], w: [4, 3, 3, 3], y: [1, 8, 8, 4] }, ['x', 'w']), 8 * 8 * 4 * 27)
  assert.equal(macs('Conv', { x: [1, 8, 8, 6], w: [1, 3, 3, 6], y: [1, 8, 8, 6] }, ['x', 'w'], { depthwise: 1 }), 8 * 8 * 6 * 9)
  assert.equal(macs('ConvTranspose', { x: [1, 16, 8, 8], w: [16, 32, 3, 3], y: [1, 32, 16, 16] }, ['x', 'w'], { layout: 'NCHW' }), 294_912)
})

test('a weight tensor without channels does not divide by zero', () => {
  assert.ok(Number.isFinite(macs('ConvTranspose', { os: [4], w: [1, 1, 1, 0], x: [1, 8, 8, 0], y: [1, 8, 8, 1] }, ['os', 'w', 'x'])))
})
