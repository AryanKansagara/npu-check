// Damaged, cut and mislabelled files (issue #2).
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { parseModel } from './load.ts'
import { buildGguf, NeedMoreBytes } from './gguf.ts'
import { buildFlatbuffer, parseTflite } from './tflite.ts'
import { cnnClassifier, detector, transformerBlock } from './samples.ts'

const sample = (n: string) => new Uint8Array(readFileSync(new URL(`../../public/samples/${n}`, import.meta.url)))
const txt = (s: string) => new TextEncoder().encode(s)

test('GGUF: every cut of the sample is refused, the whole file is not', () => {
  const b = sample('stories260K.gguf')
  assert.equal(parseModel(b, 'a.gguf').requiredSize, b.length)
  for (const n of [8, 60, 5000, 20_000, 600_000, b.length - 1]) assert.throws(() => parseModel(b.slice(0, n), 'a.gguf'), /cut short/)
  // A partial read of a longer file asks for more instead.
  assert.throws(() => parseModel(b.slice(0, 5000), 'a.gguf', b.length), NeedMoreBytes)
})

test('GGUF: data starts at the declared alignment', () => {
  const t = [{ name: 'a', dims: [8], type: 0 }]
  for (const align of [32, 64, 4096]) {
    const h = buildGguf(align === 32 ? [] : [['general.alignment', 'u32', align]], t)
    // One F32 tensor of 8 weights: 32 bytes after the aligned header.
    const need = Math.ceil(h.length / align) * align + 32
    assert.equal(parseModel(h, 'a.gguf', need).requiredSize, need)
    assert.throws(() => parseModel(h, 'a.gguf', need - 1), new RegExp(need.toLocaleString('en-US')))
  }
})

test('ONNX and TFLite: every cut of the bundled models is refused', () => {
  for (const [name, b] of [['mnist.onnx', sample('mnist-12.onnx')], ['cnn.onnx', cnnClassifier()], ['t.onnx', transformerBlock()], ['d.onnx', detector()], ['f.tflite', sample('face_detection_short_range.tflite')]] as const) {
    const step = Math.max(1, Math.floor(b.length / 400))
    const cuts = new Set([...Array(60).keys()].map(k => b.length - 1 - k).concat([...Array(400).keys()].map(k => k * step)))
    const whole = parseModel(b, name)
    for (const n of cuts) {
      let cut
      try { cut = parseModel(b.slice(0, n), name) } catch { continue }
      // Only a cut exactly between trailing metadata fields may pass: the model itself is then complete.
      assert.deepEqual([cut.nodes, cut.tensors, cut.opset, cut.outputs], [whole.nodes, whole.tensors, whole.opset, whole.outputs], `${name} cut to ${n}`)
      assert.notDeepEqual(cut.meta, whole.meta, `${name} cut to ${n}`)
    }
  }
})

test('TFLite: a buffer that claims more bytes than the file has is not counted', () => {
  const b = buildFlatbuffer([{ u32: 3 }, { vtbl: [] }, { vtbl: [[{ vtbl: [[{ vi32: [4] }, { u8: 0 }, { u32: 1 }, { str: 'w' }]] }, { vi32: [] }, { vi32: [] }, { vtbl: [] }]] }, { str: 'd' }, { vtbl: [[], [{ vu8: new Uint8Array([9, 9, 9, 9]) }]] }])
  assert.equal(parseTflite(b).tensors.w.bytes, 4)
  const at = b.lastIndexOf(9) - 3 - 4
  new DataView(b.buffer).setUint32(at, 1e9, true)
  assert.throws(() => parseModel(b, 'a.tflite'), /cut short, damaged/)
})

test('files that only carry a model name say what they are', () => {
  assert.throws(() => parseModel(txt('version https://git-lfs.github.com/spec/v1\n'), 'm.onnx'), /Git LFS/)
  assert.throws(() => parseModel(txt('\n<!DOCTYPE html><html>'), 'm.onnx'), /web page/)
  assert.throws(() => parseModel(Uint8Array.from([0x50, 0x4b, 3, 4, 20, 0, 0, 0, 8]), 'm.onnx'), /ZIP archive/)
  assert.throws(() => parseModel(Uint8Array.from([0x08, 8, 0x3d, 1]), 'm.onnx'), /cut short, damaged/)
})
