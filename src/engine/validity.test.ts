// Structurally invalid graphs (issue #4).
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { buildOnnx, parseOnnx, type BuildNode } from './onnx.ts'
import { parseModel } from './load.ts'
import { analyze, DEVICES } from './analyze.ts'
import { cnnClassifier, detector, transformerBlock } from './samples.ts'

const g = (nodes: BuildNode[]) => parseOnnx(buildOnnx({ name: 't', inputs: [{ name: 'x', dtype: 'fp32', shape: [1, 8] }], outputs: [{ name: 'y', dtype: 'fp32', shape: [1, 8] }], inits: [], nodes }))
const run = (ir: ReturnType<typeof g>) => analyze(ir, { pins: {}, applied: ['quantize-int8'], device: DEVICES[0] })
const relu = (i: string, o: string): BuildNode => ({ op: 'Relu', name: `relu_${o}`, inputs: [i], outputs: [o] })

test('missing input: a blocker, and the score stays below 60 even with every fix simulated', () => {
  const a = run(g([{ op: 'Add', inputs: ['x', 'ghost'], outputs: ['y'] }]))
  assert.equal(a.findings.find(f => f.id.startsWith('issue-'))!.severity, 'blocker')
  assert.equal(a.score, 40)
})

test('cycles are reported once, naming their nodes', () => {
  const ir = g([{ op: 'Add', name: 'add', inputs: ['x', 'b'], outputs: ['a'] }, relu('a', 'b'), relu('b', 'y')])
  assert.deepEqual(ir.issues.map(i => [i.level, i.where]), [['error', 'node 0 (add)']])
  assert.match(ir.issues[0].message, /add -> relu_b -> add/)
  assert.ok(run(ir).score < 60)
  assert.match(g([{ op: 'Add', name: 'self', inputs: ['x', 'y'], outputs: ['y'] }]).issues[0].message, /self -> self/)
})

test('valid graphs are not cycles: out of order, diamonds, and a 50,000 node chain', () => {
  assert.deepEqual(g([relu('a', 'y'), relu('x', 'a')]).issues, [])
  assert.deepEqual(g([relu('x', 'a'), relu('a', 'b'), { op: 'Add', inputs: ['a', 'b'], outputs: ['c'] }, { op: 'Add', inputs: ['a', 'c'], outputs: ['y'] }]).issues, [])
  const N = 50_000
  assert.deepEqual(g(Array.from({ length: N }, (_, i) => relu(i === N - 1 ? 'x' : `t${i + 1}`, i === 0 ? 'y' : `t${i}`))).issues, [])
})

test('bundled models report no structural errors', () => {
  const file = (rel: string) => parseModel(new Uint8Array(readFileSync(new URL(rel, import.meta.url))), rel)
  for (const ir of [parseOnnx(cnnClassifier()), parseOnnx(detector()), parseOnnx(detector(true)), parseOnnx(transformerBlock()), file('../../public/samples/mnist-12.onnx'), file('../../public/samples/mobilenet_v2_w8a8.onnx'), file('./fixtures/phi3-mini-4k-int4-graph.onnx')]) {
    assert.deepEqual(ir.issues.filter(i => i.level === 'error'), [])
  }
})
