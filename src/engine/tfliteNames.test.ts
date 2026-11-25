// TFLite tensors are addressed by index; names may repeat (issue #14).
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { analyze, DEVICES } from './analyze.ts'
import { buildFlatbuffer, parseTflite, type FbTable } from './tflite.ts'

/** [name, constant bytes]; every tensor is fp32 [1000]. Ops are [builtin, inputs, outputs]. */
function model(tensors: [string, number][], ops: [number, number[], number[]][], ins: number[], outs: number[]) {
  const buffers: FbTable[] = [[]]
  const tbl = tensors.map(([name, w]): FbTable => [{ vi32: [1000] }, { u8: 0 }, { u32: w ? buffers.push([{ vu8: new Uint8Array(w) }]) - 1 : 0 }, name ? { str: name } : undefined])
  return parseTflite(buildFlatbuffer([{ u32: 3 }, { vtbl: ops.map(([b]): FbTable => [{ u8: b }, undefined, { i32: 1 }, { i32: b }]) },
    { vtbl: [[{ vtbl: tbl }, { vi32: ins }, { vi32: outs }, { vtbl: ops.map(([, i, o], k): FbTable => [{ u32: k }, { vi32: i }, { vi32: o }]) }, { str: 'main' }]] }, { str: 't' }, { vtbl: buffers }]))
}
const run = (ir: ReturnType<typeof model>) => analyze(ir, { pins: {}, applied: [], device: DEVICES[0] })

test('two weights called w: 2,000 parameters, each input its own tensor', () => {
  const ir = model([['w', 4000], ['w', 4000], ['y', 0]], [[0, [0, 1], [2]]], [], [2])
  assert.equal(run(ir).params, 2000)
  assert.equal(run(ir).weightBytes, 8000)
  assert.deepEqual(ir.nodes[0].inputs, ['w', 'w#1'])
  assert.deepEqual(ir.nodes[0].outputs, ['y'])
})

test('edges follow indexes: a chain whose tensors share a name is a chain', () => {
  const ir = model([['t', 0], ['t', 0], ['t', 0]], [[19, [0], [1]], [19, [1], [2]]], [0], [2])
  assert.deepEqual(ir.nodes.map(n => [n.inputs, n.outputs]), [[['t'], ['t#1']], [['t#1'], ['t#2']]])
  assert.deepEqual([ir.inputs, ir.outputs], [['t'], ['t#2']])
  assert.equal(run(ir).peakBytes, 8000)
})

test('the added suffix and default names never collide with a real name', () => {
  for (const names of [['w', 'w', 'w#1'], ['w#1', 'w', 'w'], ['w#2', 'w', 'w', 'w#2#3'], ['tensor_1', ''], ['', 'tensor_0'], ['', '']]) {
    const ir = model([...names.map((n): [string, number] => [n, 4000]), ['y', 0]], [[0, names.map((_, i) => i), [names.length]]], [], [names.length])
    assert.equal(run(ir).params, names.length * 1000, names.join())
    assert.equal(new Set(ir.nodes[0].inputs).size, names.length)
  }
})

test('unique names are kept as they are', () => {
  assert.deepEqual(Object.keys(model([['a', 4000], ['b', 0], ['', 0]], [[0, [0, 1], [2]]], [1], [2]).tensors), ['a', 'b', 'tensor_2'])
})
