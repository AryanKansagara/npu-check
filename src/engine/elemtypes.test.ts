// Element types of tensors the file does not declare (issue #13).
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { analyze, DEVICES, topoOrder, type FixId } from './analyze.ts'
import { inferTypes } from './elemtypes.ts'
import { buildOnnx, parseOnnx, type BuildNode, type BuildTensor, type BuildValue } from './onnx.ts'
import type { DType } from './types.ts'

const build = (nodes: BuildNode[], type: DType = 'fp16', extra: { inits?: BuildTensor[]; inputs?: BuildValue[]; outputs?: BuildValue[]; valueInfo?: BuildValue[] } = {}) =>
  parseOnnx(buildOnnx({ name: 't', inputs: [{ name: 'x', dtype: type, shape: [1, 16, 8, 8] }], outputs: [], nodes, inits: [], ...extra }))
const types = (...a: Parameters<typeof build>) => { const ir = build(...a); return inferTypes(ir, topoOrder(ir)) }
const run = (ir: ReturnType<typeof build>, applied: FixId[] = []) => analyze(ir, { pins: {}, applied, device: DEVICES[0] })
const n = (op: string, inputs: string[], outputs: string[], attrs: BuildNode['attrs'] = {}, domain?: string): BuildNode => ({ op, inputs, outputs, attrs, domain })
const two = (type: DType) => build([n('Relu', ['x'], ['a']), n('Relu', ['a'], ['y'])], type, { outputs: [{ name: 'y', dtype: type, shape: [1, 16, 8, 8] }] })

test('fp16 Relu, Relu: two live tensors of 1,024 * 2 B', () => {
  assert.equal(run(two('fp16')).peakBytes, 4096)
  assert.equal(run(two('fp32')).peakBytes, 8192)
  assert.equal(run(two('fp64')).peakBytes, 16384)
})

test('memory time follows the same widths', () => {
  const ms = (t: DType) => analyze(two(t), { pins: {}, applied: [], device: { ...DEVICES[0], memGBs: 0.002 } }).latency.npuMs
  assert.equal(ms('fp32') / ms('fp16'), 2)
})

test('simulated int8 quantization leaves one byte per float and does not touch integers', () => {
  for (const t of ['fp16', 'fp32', 'bf16'] as const) assert.equal(run(two(t), ['quantize-int8']).peakBytes, 2048, t)
  assert.equal(run(two('int64'), ['quantize-int8']).peakBytes, 16384)
})

test('Cast, comparisons, Shape, TopK and QuantizeLinear have their own output types', () => {
  const t = types([
    n('Cast', ['x'], ['c'], { to: 7 }), n('Neg', ['c'], ['cn']), n('Greater', ['x', 'x'], ['g']), n('Not', ['g'], ['gn']), n('Where', ['g', 'c', 'c'], ['w']),
    n('Shape', ['x'], ['s']), n('TopK', ['x', 'k'], ['v', 'i']), n('ArgMax', ['x'], ['am']), n('MaxPool', ['x'], ['mp', 'mi']), n('CastLike', ['x', 'c'], ['cl']),
    n('QuantizeLinear', ['x', 'sc'], ['q8']), n('QuantizeLinear', ['x', 'sc', 'zp'], ['qi']), n('DequantizeLinear', ['qi', 'sc', 'zp'], ['dq']), n('OneHot', ['c', 'k', 'x'], ['oh']),
    n('MatMulInteger', ['q8', 'q8'], ['mi32']), n('Dropout', ['x'], ['d', 'mask']), n('Split', ['x'], ['s1', 's2']),
  ], 'fp16', { inits: [{ name: 'k', dtype: 'int64', dims: [1], data: [1] }, { name: 'sc', dtype: 'fp16', dims: [] }, { name: 'zp', dtype: 'int8', dims: [] }] })
  const want = { c: 'int64', cn: 'int64', g: 'bool', gn: 'bool', w: 'int64', s: 'int64', v: 'fp16', i: 'int64', am: 'int64', mp: 'fp16', mi: 'int64', cl: 'int64', q8: 'uint8', qi: 'int8', dq: 'fp16', oh: 'fp16', mi32: 'int32', d: 'fp16', mask: 'bool', s1: 'fp16', s2: 'fp16' }
  assert.deepEqual(Object.fromEntries(Object.keys(want).map(k => [k, t[k]])), want)
})

test('a declared type wins, also as the source for what follows', () => {
  const t = types([n('Relu', ['x'], ['a']), n('Relu', ['a'], ['b'])], 'fp16', { valueInfo: [{ name: 'a', dtype: 'fp32', shape: [1, 16, 8, 8] }] })
  assert.deepEqual([t.a, t.b], ['fp32', 'fp32'])
})

test('types are not guessed for control flow, custom ops or bad Cast targets', () => {
  const t = types([n('If', ['x'], ['a']), n('Magic', ['x'], ['b'], {}, 'com.acme'), n('Cast', ['x'], ['c'], { to: 99 }), n('Cast', ['x'], ['d']), n('Relu', ['b'], ['e']), n('Gelu', ['x'], ['f'], {}, 'com.microsoft')])
  assert.deepEqual([t.a, t.b, t.c, t.d, t.e, t.f], [undefined, undefined, undefined, undefined, undefined, 'fp16'])
})

test('an fp16 QDQ graph still counts undeclared activations at the quantized width', () => {
  const ir = build([n('QuantizeLinear', ['x', 'sc', 'zp'], ['q']), n('DequantizeLinear', ['q', 'sc', 'zp'], ['d']), n('Relu', ['d'], ['y'])], 'fp16',
    { inits: [{ name: 'sc', dtype: 'fp16', dims: [] }, { name: 'zp', dtype: 'uint8', dims: [] }] })
  // x 2,048 + q 1,024 at the start; d and y are 1,024 each.
  assert.equal(run(ir).peakBytes, 3072)
})
