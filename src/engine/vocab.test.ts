// GGUF files without weights (issue #12).
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { analyze, compare, DEVICES, summarize } from './analyze.ts'
import { buildGguf, parseGguf } from './gguf.ts'

const run = (arch: string, tensors: Parameters<typeof buildGguf>[1] = []) => analyze(parseGguf(buildGguf([['general.architecture', 'str', arch],
  ...['block_count', 'context_length', 'embedding_length'].map((k): [string, 'u32', number] => [`${arch}.${k}`, 'u32', 1024])], tensors)), { pins: {}, applied: ['quantize-int8'], device: DEVICES[0] })

test('no weights: no speed, latency or readiness, and a finding says why', () => {
  for (const arch of ['llama', 'bert']) for (const a of [run(arch), run(arch, [{ name: 'empty', dims: [0], type: 0 }])]) {
    assert.deepEqual([a.latency.tokensPerSec, a.latency.prefillMs, a.latency.kvCacheBytes, a.score], [undefined, undefined, undefined, 0])
    assert.ok(!Number.isFinite(a.latency.totalMs))
    assert.deepEqual(a.findings.map(f => [f.id, f.severity]), [['no-weights', 'blocker']])
  }
})

test('comparing with a model that has weights judges no latency, also after a JSON round trip', () => {
  const none = summarize(run('llama')), some = summarize(run('llama', [{ name: 'w', dims: [64, 64], type: 1 }]))
  for (const v of [none, JSON.parse(JSON.stringify(none))]) for (const d of [compare(v, some), compare(some, v)]) assert.equal(d.metrics.find(m => m.key === 'latencyMs')!.better, null)
})

test('a file with weights keeps its estimates', () => {
  const a = run('llama', [{ name: 'w', dims: [64, 64], type: 1 }])
  assert.ok(Number.isFinite(a.latency.tokensPerSec) && a.latency.totalMs > 0 && a.score === 45)
})
