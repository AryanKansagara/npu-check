// CSV exports of untrusted names (issue #8).
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { analyze, DEVICES, nodeReportCsv, opReportCsv } from './analyze.ts'
import { buildOnnx, parseOnnx } from './onnx.ts'
import { cnnClassifier } from './samples.ts'

const run = (name: string, op = 'Relu', domain = '', dim: string | number = 1) => analyze(parseOnnx(buildOnnx({ name: 't', inputs: [{ name: 'x', dtype: 'fp32', shape: [dim, 8] }], outputs: [{ name: 'y', dtype: 'fp32', shape: [dim, 8] }], inits: [], nodes: [{ op, name, domain, inputs: ['x'], outputs: ['y'] }] })), { pins: {}, applied: [], device: DEVICES[0] })
const row = (csv: string) => csv.slice(csv.indexOf('\n') + 1)

test('text starting like a formula gets a leading quote', () => {
  for (const c of ['=', '+', '-', '@', '＝', '＋', '－', '＠', '\t']) assert.ok(row(nodeReportCsv(run(`${c}1+2`))).startsWith(`0,'${c}1+2,Relu,`), JSON.stringify(c))
  assert.ok(row(nodeReportCsv(run('=HYPERLINK("http://evil.example","x")'))).startsWith(`0,"'=HYPERLINK(""http://evil.example"",""x"")",Relu,`))
})

test('carriage returns are quoted, at the start or inside', () => {
  assert.ok(row(nodeReportCsv(run('a\rb'))).startsWith('0,"a\rb",Relu,'))
  assert.ok(row(nodeReportCsv(run('\r=1'))).startsWith(`0,"'\r=1",Relu,`))
})

test('op type, domain and symbolic shapes are covered too', () => {
  assert.ok(row(opReportCsv(run('n', '=1+2', '@evil'))).startsWith(`'=1+2,'@evil,1,`))
  assert.match(nodeReportCsv(run('n', 'Relu', '', '=cmd')), /,'=cmdx8,/)
})

test('ordinary names and numbers are unchanged', () => {
  const a = run('a-b=c')
  assert.ok(row(nodeReportCsv(a)).startsWith('0,a-b=c,Relu,npu,'))
  assert.match(nodeReportCsv(a), /,1x8,0,0\.\d{4}\n$/)
  const csv = nodeReportCsv(analyze(parseOnnx(cnnClassifier()), { pins: {}, applied: [], device: DEVICES[0] })) + opReportCsv(a)
  assert.ok(!csv.includes("'"))
})
