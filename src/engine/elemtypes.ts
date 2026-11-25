// Element type of every tensor: declared types first, then what onnx/docs/Operators.md says the op produces.
import { ONNX_DTYPES } from './onnx.ts'
import type { DType, ModelIR, ModelNode } from './types.ts'

const BOOL = new Set(['Equal', 'Greater', 'GreaterOrEqual', 'Less', 'LessOrEqual', 'And', 'Or', 'Xor', 'Not', 'IsNaN', 'IsInf', 'RegexFullMatch'])
const INT64 = new Set(['Shape', 'Size', 'NonZero', 'NonMaxSuppression', 'ArgMax', 'ArgMin', 'SequenceLength'])
/** Second and later outputs that do not share the first one's type. */
const EXTRA: Record<string, DType> = { TopK: 'int64', MaxPool: 'int64', Unique: 'int64', Dropout: 'bool' }
/** Input whose type the output has, when it is not the first one. -1: not decided by one tensor input. */
const FROM: Record<string, number> = {
  CastLike: 1, Where: 1, OneHot: 2, DequantizeLinear: 1, QLinearConv: 7, QLinearMatMul: 7, MatMulIntegerToFloat: 2, DynamicQuantizeMatMul: 2,
  GatherBlockQuantized: 1, If: -1, Loop: -1, Scan: -1, SequenceEmpty: -1, SequenceConstruct: -1, SequenceAt: -1, SequenceInsert: -1, SequenceErase: -1,
  SequenceMap: -1, SplitToSequence: -1, ConcatFromSequence: -1, Optional: -1, OptionalGetElement: -1, OptionalHasElement: -1,
  StringSplit: -1, TfIdfVectorizer: -1, ImageDecoder: -1, HannWindow: -1, HammingWindow: -1, BlackmanWindow: -1, MelWeightMatrix: -1, Constant: -1,
}
/** Contributed ops whose outputs have the type of one input; anything else outside ai.onnx is not guessed. */
const CONTRIB = new Set(['Gelu', 'FastGelu', 'QuickGelu', 'LayerNormalization', 'SkipLayerNormalization', 'SimplifiedLayerNormalization', 'SkipSimplifiedLayerNormalization',
  'Attention', 'MultiHeadAttention', 'GroupQueryAttention', 'RotaryEmbedding', 'MatMulNBits', 'FusedMatMul', 'FusedConv', 'FusedGemm', 'QuantizeLinear', 'DequantizeLinear',
  'DynamicQuantizeMatMul', 'MatMulIntegerToFloat', 'GatherBlockQuantized', 'QLinearAdd', 'QLinearMul', 'QLinearSigmoid', 'QLinearLeakyRelu', 'QLinearSoftmax', 'QLinearAveragePool', 'QLinearGlobalAveragePool'])

const attrType = (n: ModelNode, key: string): DType | undefined => typeof n.attrs[key] === 'number' && n.attrs[key] ? ONNX_DTYPES[n.attrs[key] as number] : undefined

function outputTypes(n: ModelNode, of: (name: string) => DType | undefined): (DType | undefined)[] {
  const all = (t: DType | undefined) => n.outputs.map(() => t)
  if (n.domain && !(n.domain === 'com.microsoft' && CONTRIB.has(n.op))) return []
  if (BOOL.has(n.op)) return all('bool')
  if (INT64.has(n.op)) return all('int64')
  switch (n.op) {
    case 'Cast': return all(attrType(n, 'to'))
    case 'QuantizeLinear': return all((n.inputs[2] && of(n.inputs[2])) || attrType(n, 'output_dtype') || 'uint8')
    case 'DequantizeLinear': return all(attrType(n, 'output_dtype') ?? of(n.inputs[1]))
    case 'DynamicQuantizeLinear': return ['uint8', 'fp32', 'uint8']
    case 'MatMulInteger': case 'ConvInteger': return all('int32')
    case 'ConstantOfShape': return all(undefined) // its type is in a tensor attribute
    case 'RandomNormal': case 'RandomUniform': return all(attrType(n, 'dtype') ?? 'fp32')
    case 'Multinomial': return all(attrType(n, 'dtype') ?? 'int32')
    case 'RandomNormalLike': case 'RandomUniformLike': case 'Bernoulli': case 'EyeLike': return all(attrType(n, 'dtype') ?? of(n.inputs[0]))
  }
  const first = FROM[n.op] === -1 ? undefined : of(n.inputs[FROM[n.op] ?? 0] ?? '')
  return n.outputs.map((_, k) => k && EXTRA[n.op] ? EXTRA[n.op] : first)
}

export function inferTypes(ir: ModelIR, order: number[]): Record<string, DType | undefined> {
  const T: Record<string, DType | undefined> = {}
  for (const t of Object.values(ir.tensors)) if (t.dtype !== 'unknown') T[t.name] = t.dtype
  if (ir.format !== 'onnx') return T
  for (const i of order) {
    const n = ir.nodes[i], out = outputTypes(n, x => T[x])
    n.outputs.forEach((o, k) => { if (o && !T[o]) T[o] = out[k] })
  }
  return T
}
