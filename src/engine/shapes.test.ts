// Shape inference against onnx/docs/Operators.md (issue #10).
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { analyze, DEVICES, inferShapes } from './analyze.ts'
import { buildOnnx, parseOnnx, type BuildNode, type BuildTensor } from './onnx.ts'
import type { Dim, ModelIR } from './types.ts'

const X: Dim[] = [1, 16, 8, 8]
const ints = (name: string, data: number[]): BuildTensor => ({ name, dtype: 'int64', dims: [data.length], data })
const ir = (nodes: BuildNode[], inits: BuildTensor[] = [], x = X): ModelIR => parseOnnx(buildOnnx({ name: 't', inputs: [{ name: 'x', dtype: 'fp32', shape: x }], outputs: [], nodes, inits }))
const shapes = (node: Omit<BuildNode, 'outputs'>, inits: BuildTensor[] = [], x = X, outs = ['y']) => { const s = inferShapes(ir([{ ...node, outputs: outs }], inits, x)); return outs.map(o => s[o]) }

const deconv = (attrs: BuildNode['attrs'], x = X) => shapes({ op: 'ConvTranspose', inputs: ['x', 'w'], attrs }, [{ name: 'w', dtype: 'fp32', dims: [16, 4, 3, 3] }], x)[0]
test('ConvTranspose: output_padding, output_shape and SAME', () => {
  assert.deepEqual(deconv({ strides: [2, 2], pads: [1, 1, 1, 1], output_padding: [1, 1] }), [1, 4, 16, 16])
  assert.deepEqual(deconv({ strides: [2, 2], pads: [1, 1, 1, 1] }), [1, 4, 15, 15])
  assert.deepEqual(deconv({ strides: [2, 2], output_shape: [16, 17] }), [1, 4, 16, 17])
  assert.deepEqual(deconv({ strides: [2, 2], auto_pad: 'SAME_UPPER' }), [1, 4, 16, 16])
  assert.deepEqual(deconv({ pads: [1, 1, 1, 1] }, [1, 16, 'h', 8]), [1, 4, 'h', 8])
  assert.deepEqual(deconv({ pads: [1, 1, 1, 1], dilations: [1, 1], output_padding: [1, 0] }, [1, 16, 'h', 8]), [1, 4, '?', 8])
})

const slice = (start: number, end: number, step?: number, x = X) =>
  shapes({ op: 'Slice', inputs: ['x', 's', 'e', 'a', ...(step === undefined ? [] : ['p'])] }, [ints('s', [start]), ints('e', [end]), ints('a', [1]), ...(step === undefined ? [] : [ints('p', [step])])], x)[0]![1]
test('Slice agrees with Python slicing in both directions', () => {
  assert.equal(slice(-1, -9e15, -1), 16)
  for (const step of [1, 2, 3, -1, -2, -5]) for (let s = -20; s <= 20; s++) for (let e = -20; e <= 20; e++) {
    // The spec clamps, so s and e walk the same positions as a plain loop over 0..15.
    const at = (v: number) => v < 0 ? v + 16 : v
    let n = 0
    if (step > 0) for (let i = Math.max(at(s), 0); i < Math.min(at(e), 16); i += step) n++
    else for (let i = Math.min(Math.max(at(s), 0), 15); i > Math.max(at(e), -1); i += step) n++
    assert.equal(slice(s, e, step), n, `[${s}:${e}:${step}]`)
  }
  assert.equal(slice(0, 4), 4)
  assert.equal(slice(0, 4, 0), '?')
  assert.equal(slice(-1, -9e15, -1, [1, 0, 8, 8]), 0)
})

test('Slice-1 reads starts, ends and axes from attributes and stays on the NPU', () => {
  assert.deepEqual(shapes({ op: 'Slice', inputs: ['x'], attrs: { starts: [0], ends: [4], axes: [1] } }), [[1, 4, 8, 8]])
  assert.deepEqual(shapes({ op: 'Slice', inputs: ['x'], attrs: { starts: [0, 2], ends: [1, 9] } }), [[1, 7, 8, 8]])
  const a = analyze(ir([{ op: 'Slice', inputs: ['x'], outputs: ['y'], attrs: { starts: [0], ends: [4], axes: [1] } }]), { pins: {}, applied: [], device: DEVICES[0] })
  assert.equal(a.nodes[0].status, 'npu')
})

test('Slice with axes or steps computed at runtime has no known shape', () => {
  assert.deepEqual(shapes({ op: 'Slice', inputs: ['x', 's', 'e', 'x'] }, [ints('s', [0]), ints('e', [4])]), [['?', '?', '?', '?']])
  assert.deepEqual(shapes({ op: 'Slice', inputs: ['x', 's', 'e', '', 'x'] }, [ints('s', [0]), ints('e', [4])]), [['?', '?', '?', '?']])
})

const pad = (pads: number[], axes?: number[], x = X) => shapes({ op: 'Pad', inputs: ['x', 'p', ...(axes ? ['', 'ax'] : [])] }, [ints('p', pads), ...(axes ? [ints('ax', axes)] : [])], x)[0]
test('Pad-18 axes', () => {
  assert.deepEqual(pad([1, 1, 1, 1], [2, 3]), [1, 16, 10, 10])
  assert.deepEqual(pad([1, 2, 3, 4], [-1, 2], ['batch', 16, 8, 8]), ['batch', 16, 8 + 2 + 4, 8 + 1 + 3])
  assert.deepEqual(pad([0, 0, 1, 1, 0, 0, 1, 1]), [1, 16, 10, 10])
  assert.equal(pad([1, 1, 1, 1]), null)
})

const split = (attrs: BuildNode['attrs'], outs: string[], x = X) => shapes({ op: 'Split', inputs: ['x'], attrs: { axis: 1, ...attrs } }, [], x, outs).map(s => s![1])
test('Split-18 num_outputs: the last chunk is smaller', () => {
  assert.deepEqual(split({ num_outputs: 3 }, ['a', 'b', 'c']), [6, 6, 4])
  assert.deepEqual(split({ num_outputs: 2 }, ['a', 'b']), [8, 8])
  assert.deepEqual(split({}, ['a', 'b']), [8, 8])
  assert.deepEqual(split({ split: [10, 6] }, ['a', 'b']), [10, 6])
  // 5 over 4 outputs would need a last chunk of -1.
  assert.equal(split({ num_outputs: 4 }, ['a', 'b', 'c', 'd'], [1, 5, 8, 8])[3], '?')
  assert.equal(split({ num_outputs: 2 }, ['a', 'b'], [1, 'c', 8, 8])[1], '?')
})

test('Constant value_ints and value_int feed shapes; value_floats and value_float are known constants', () => {
  const g = ir([{ op: 'Constant', inputs: [], outputs: ['sh'], attrs: { value_ints: [1, 16, 64] } }, { op: 'Reshape', inputs: ['x', 'sh'], outputs: ['y'] },
    { op: 'Constant', inputs: [], outputs: ['i'], attrs: { value_int: 3 } }, { op: 'Gather', inputs: ['x', 'i'], outputs: ['g'], attrs: { axis: 1 } },
    { op: 'Constant', inputs: [], outputs: ['f'], attrs: { value_float: 0.5 } }, { op: 'Mul', inputs: ['x', 'f'], outputs: ['m'] }])
  const s = inferShapes(g)
  assert.deepEqual(s.y, [1, 16, 64])
  assert.deepEqual(s.g, [1, 8, 8])
  assert.deepEqual(s.m, X)
  assert.deepEqual([g.tensors.sh.dtype, g.tensors.f.dtype], ['int64', 'fp32'])
  assert.deepEqual(g.issues, [])
})
