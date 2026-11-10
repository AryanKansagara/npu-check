// Format detection and parsing entry point.
import { isGguf, parseGguf } from './gguf.ts'
import { parseOnnx } from './onnx.ts'
import { isTflite, parseTflite } from './tflite.ts'
import type { ModelIR } from './types.ts'

export type Format = ModelIR['format']

export function detectFormat(head: Uint8Array, fileName = ''): Format | null {
  if (isGguf(head)) return 'gguf'
  if (isTflite(head)) return 'tflite'
  if (/\.onnx$/i.test(fileName) || head[0] === 0x08) return 'onnx'
  return null
}

export function parseModel(buf: Uint8Array, fileName = '', fileSize = buf.length): ModelIR {
  const fmt = detectFormat(buf, fileName)
  if (fmt === 'gguf') return parseGguf(buf, fileSize)
  if (fmt === 'tflite') return parseTflite(buf)
  if (fmt === 'onnx') return parseOnnx(buf)
  if (/\.(onnx_data|data|bin)$/i.test(fileName)) throw new Error('This looks like an external weights file. Drop the .onnx graph file instead; weights are not needed.')
  if (/\.(pt|pth|safetensors|h5|pb)$/i.test(fileName)) throw new Error('Export to ONNX or TFLite first (see Docs). PyTorch, safetensors, Keras and TF graphs are not read directly.')
  throw new Error('Unrecognised file: expected ONNX (.onnx), TFLite (.tflite, "TFL3" identifier) or GGUF ("GGUF" magic).')
}
