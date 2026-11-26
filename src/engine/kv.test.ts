// KV cache size (issue #11): head sizes from the GGUF spec's attention.key_length / value_length.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { analyze, DEVICES } from './analyze.ts'
import { buildGguf, parseGguf } from './gguf.ts'

const kv = (extra: Record<string, number>) => {
  const keys = { block_count: 28, context_length: 40960, embedding_length: 1024, ...extra }
  const ir = parseGguf(buildGguf([['general.architecture', 'str', 'qwen3'], ...Object.entries(keys).map(([k, v]): [string, 'u32', number] => [`qwen3.${k}`, 'u32', v])], [{ name: 'w', dims: [64, 64], type: 1 }]))
  return analyze(ir, { pins: {}, applied: [], device: DEVICES[0] }).latency.kvCacheBytes
}
const heads = { 'attention.head_count': 16, 'attention.head_count_kv': 8 }

test('declared head sizes: 28 * 4,096 * (8*128 + 8*128) * 2 B', () => {
  assert.equal(kv({ ...heads, 'attention.key_length': 128, 'attention.value_length': 128 }), 469_762_048)
})

test('key and value sizes are independent, and a missing one is width / heads', () => {
  assert.equal(kv({ ...heads, 'attention.key_length': 32, 'attention.value_length': 16 }), 28 * 4096 * (8 * 32 + 8 * 16) * 2)
  assert.equal(kv({ ...heads, 'attention.key_length': 128 }), 28 * 4096 * (8 * 128 + 8 * 64) * 2)
})

test('head_count_kv defaults to head_count', () => {
  assert.equal(kv({ 'attention.head_count': 16, 'attention.key_length': 128, 'attention.value_length': 128 }), 28 * 4096 * 2 * 16 * 128 * 2)
})

test('headers without the keys are unchanged', () => {
  assert.equal(kv(heads), 234_881_024)
  assert.equal(kv({ 'attention.head_count': 16 }), 469_762_048)
  assert.equal(kv({}), 469_762_048)
})
