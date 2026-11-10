// Format-neutral model description. Small enough to store and sync: no weight bytes.

/** number = static, string = symbolic (dim_param) or '?' when unknown */
export type Dim = number | string

export type DType =
  | 'fp32' | 'fp16' | 'bf16' | 'fp64' | 'fp8'
  | 'int8' | 'uint8' | 'int16' | 'uint16' | 'int32' | 'uint32' | 'int64' | 'uint64'
  | 'int4' | 'uint4' | 'bool' | 'string' | 'complex' | 'unknown'
  | `ggml:${string}`

export interface TensorInfo {
  name: string
  dtype: DType
  shape: Dim[] | null
  /** Initializer / constant weight */
  init?: boolean
  /** Stored outside the model file (ONNX external data); size is computed from dims */
  external?: boolean
  /** Byte size for initializers when known exactly (GGUF block formats) */
  bytes?: number
  /** Small integer constants (Reshape shapes, axes), kept so shapes can be inferred */
  values?: number[]
  /** Quantization metadata (TFLite tensors) */
  quant?: { scales: number; axis?: number }
}

export type AttrValue = number | string | number[] | string[]

export interface ModelNode {
  name: string
  op: string
  domain: string
  inputs: string[]
  outputs: string[]
  attrs: Record<string, AttrValue>
}

export interface Issue {
  level: 'error' | 'warn' | 'info'
  where: string
  message: string
}

export interface LlmMeta {
  arch: string
  name?: string
  contextLength?: number
  blockCount?: number
  embeddingLength?: number
  headCount?: number
  headCountKv?: number
  fileType?: string
}

export interface ModelIR {
  format: 'onnx' | 'tflite' | 'gguf'
  producer: string
  irVersion?: number
  opset: { domain: string; version: number }[]
  graphName: string
  inputs: string[]
  outputs: string[]
  tensors: Record<string, TensorInfo>
  nodes: ModelNode[]
  issues: Issue[]
  meta: Record<string, string>
  fileSize: number
  llm?: LlmMeta
}

export const DTYPE_BYTES: Record<string, number> = {
  fp32: 4, fp16: 2, bf16: 2, fp64: 8, fp8: 1, int8: 1, uint8: 1, int16: 2, uint16: 2, int32: 4, uint32: 4,
  int64: 8, uint64: 8, int4: 0.5, uint4: 0.5, bool: 1, string: 0, complex: 8, unknown: 4,
}

export const isStatic = (s: Dim[] | null): s is number[] => !!s && s.every(d => typeof d === 'number' && d >= 0)
