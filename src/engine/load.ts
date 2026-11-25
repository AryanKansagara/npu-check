// Format detection and parsing entry point.
import { isGguf, NeedMoreBytes, parseGguf } from './gguf.ts'
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
  if (fmt === 'gguf') return loadGguf(buf, fileSize)
  if (fmt === 'tflite') return explain('TFLite', () => parseTflite(buf))
  const other = notAModel(buf)
  if (other) throw new Error(other)
  if (fmt === 'onnx') return explain('ONNX', () => parseOnnx(buf))
  if (/\.(onnx_data|data|bin)$/i.test(fileName)) throw new Error('This looks like an external weights file. Drop the .onnx graph file instead; weights are not needed.')
  if (/\.(pt|pth|safetensors|h5|pb)$/i.test(fileName)) throw new Error('Export to ONNX or TFLite first (see Docs). PyTorch, safetensors, Keras and TF graphs are not read directly.')
  throw new Error('Unrecognised file: expected ONNX (.onnx), TFLite (.tflite, "TFL3" identifier) or GGUF ("GGUF" magic).')
}

function loadGguf(buf: Uint8Array, fileSize: number): ModelIR {
  try {
    const ir = parseGguf(buf, fileSize)
    if (ir.requiredSize! > fileSize) throw new Error(`This GGUF is cut short: its tensor table needs ${ir.requiredSize!.toLocaleString('en-US')} bytes and the file has ${fileSize.toLocaleString('en-US')}. Download it again.`)
    return ir
  } catch (e) {
    // The caller can read on only when there is more file.
    if (e instanceof NeedMoreBytes && buf.length >= fileSize) throw new Error('This GGUF is cut short or damaged: the file ends inside its header. Download it again.')
    throw e
  }
}

/** Parser errors are byte offsets; say what they usually mean. */
function explain(kind: string, parse: () => ModelIR): ModelIR {
  try { return parse() } catch (e) {
    const detail = e instanceof Error ? e.message : String(e)
    throw new Error(/^Not an ONNX model/.test(detail) ? detail : `Could not read this file as ${kind}: it is cut short, damaged or not a ${kind} model. Detail: ${detail}`)
  }
}

/** Common files that end up with a model's name. */
function notAModel(b: Uint8Array): string | null {
  const head = new TextDecoder().decode(b.subarray(0, 200)).trimStart()
  if (/^version https:\/\/git-lfs/.test(head)) return 'This is a Git LFS pointer, not the model. Run "git lfs pull", or download the file itself.'
  if (/^<(!doctype|html|\?xml)/i.test(head)) return 'This is a web page, not a model file. The download probably returned an error or sign-in page.'
  if (/^PK\x03\x04/.test(head)) return 'This is a ZIP archive or a PyTorch/Keras checkpoint. Unzip it, or export the model to ONNX or TFLite first (see Docs).'
  return null
}
