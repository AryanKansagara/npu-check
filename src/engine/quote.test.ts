// Names from model files in copied commands (issue #3). Where bash or python3 exist, they are the judge.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { py, sh, shOpt, shPath } from './quote.ts'

const NASTY = ['seq; touch /tmp/x #', '$(id)', '`id`', "it's", `a"b'c`, 'model (1).onnx', 'a\nid\n', '\\', "\\'", 'tab\there', '\u001b[201~id', 'x\u2028y', '日本 seq', '!!', '~', '*', 'a&b|c>d', '', '\u0085', '\u0001f']

test('ordinary names are unchanged', () => {
  for (const s of ['batch', 'sequence_length', 'unk__12', 'model.onnx', 'a-b/c.1']) { assert.equal(sh(s), s); assert.equal(py(s), `"${s}"`) }
})

test('every shell word expands back to exactly the name', { skip: spawnSync('bash', ['-c', 'true']).status !== 0 }, () => {
  for (const s of NASTY) assert.equal(spawnSync('bash', ['-c', `printf %s ${sh(s)}`], { encoding: 'utf8', env: { ...process.env, LC_ALL: 'C.UTF-8' } }).stdout, s, JSON.stringify(s))
})

test('every Python literal evaluates to exactly the name', { skip: spawnSync('python3', ['-c', '']).status !== 0 }, () => {
  for (const s of NASTY) assert.equal(JSON.parse(spawnSync('python3', ['-I', '-c', `import json\nprint(json.dumps(${py(s)}))`], { encoding: 'utf8' }).stdout), s)
})

test('no raw control characters reach the clipboard, and dashes are not options', () => {
  for (const s of NASTY) assert.doesNotMatch(sh(s) + py(s), /[\x00-\x1f\x7f-\x9f\u2028\u2029]/)
  assert.equal(shPath('-rf.onnx'), './-rf.onnx')
  assert.equal(shOpt('--dim_param', '-x y'), "--dim_param='-x y'")
  assert.equal(shOpt('--dim_param', 'seq'), '--dim_param seq')
})
