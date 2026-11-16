// Curated op support table for NPU-style delegates (ONNX Runtime QNN EP / QNN HTP backend style).
// This is a first-pass heuristic assembled from public documentation, not an official compatibility list.
// Support changes with every QAIRT and ONNX Runtime release: confirm with the converter or EP partition logs.

export type Support = 'npu' | 'constrained' | 'cpu'

export interface OpRule {
  status: Support
  /** What the constraint is, or why it falls back */
  note: string
  /** Replacement for ops that fall back mid-graph */
  replace?: string
  /** Typically belongs in post-processing on the CPU (detection heads) */
  post?: boolean
}

export const SOURCES = [
  { name: 'ONNX Runtime QNN Execution Provider, supported operators', url: 'https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html', checked: '2026-11-15' },
  { name: 'Qualcomm AI Engine Direct (QNN) SDK docs, HTP backend op definitions', url: 'https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/HtpOpDefSupplement.html', checked: '2026-11-10' },
  { name: 'LiteRT (TFLite) delegates overview', url: 'https://ai.google.dev/edge/litert/performance/delegates', checked: '2026-11-10' },
] as const

export const TABLE_DATE = '2026-11-15'

/**
 * ai.onnx operators in the "Supported ONNX operators" table of the ONNX Runtime QNN EP page, as published on 2026-11-15.
 * The page is a subset of what the HTP backend runs (Reshape, for one, is missing), so absence only lowers confidence.
 */
export const QNN_EP_LISTED = new Set(`Abs Add And ArgMax ArgMin Asin Atan AveragePool BatchNormalization Cast Clip Concat Conv ConvTranspose Cos
DepthToSpace DequantizeLinear Div Elu Equal Exp Expand Flatten Floor Gather Gelu Gemm GlobalAveragePool Greater GreaterOrEqual GridSample
HardSwish InstanceNormalization LRN LayerNormalization LeakyRelu Less LessOrEqual Log LogSoftmax LpNormalization MatMul Max MaxPool Min Mul
Neg Not Or Pad Pow PRelu QuantizeLinear ReduceMax ReduceMean ReduceMin ReduceProd ReduceSum Relu Resize Round Sigmoid Sign Sin Slice Softmax
SpaceToDepth Split Sqrt Squeeze Sub Tanh Tile TopK Transpose Unsqueeze Where`.split(/\s+/))
/** Structural ops every converter folds or handles; not worth flagging as unlisted. */
const STRUCTURAL = new Set(['Reshape', 'Identity', 'Dropout', 'Constant', 'Shape', 'ConstantOfShape'])
const UNLISTED = ' Not in the ONNX Runtime QNN EP operator table; confirm in the partition log.'
const QOP = 'QOperator format (QLinear ops). The QNN EP expects QDQ models.'
const QOP_FIX = 'Re-quantize with quantize_static(..., quant_format=QuantFormat.QDQ) from the float model.'
const DYNQ = 'Dynamic quantization computes scales at runtime, which the NPU cannot do.'
const DYNQ_FIX = 'Quantize statically instead: quantize_static with calibration data and QuantFormat.QDQ.'

const npu = (note = 'Runs on the NPU with static shapes.'): OpRule => ({ status: 'npu', note })
const con = (note: string): OpRule => ({ status: 'constrained', note })
const cpu = (note: string, extra: Partial<OpRule> = {}): OpRule => ({ status: 'cpu', note, ...extra })

const ELEMENTWISE = ['Asin', 'Atan', 'Add', 'Sub', 'Mul', 'Div', 'Abs', 'Neg', 'Exp', 'Log', 'Sqrt', 'Rsqrt', 'Pow', 'Max', 'Min', 'Floor', 'Ceil', 'Round', 'Sign', 'Sin', 'Cos', 'Equal', 'Greater', 'GreaterOrEqual', 'Less', 'LessOrEqual', 'And', 'Or', 'Not', 'Where', 'Erf']
const ACTIVATIONS = ['Relu', 'Sigmoid', 'Tanh', 'LeakyRelu', 'PRelu', 'Elu', 'HardSigmoid', 'HardSwish', 'Gelu', 'Softmax', 'LogSoftmax', 'Clip']
const LAYOUT = ['Reshape', 'Transpose', 'Flatten', 'Squeeze', 'Unsqueeze', 'Concat', 'Split', 'Slice', 'Tile', 'Expand', 'Pad', 'DepthToSpace', 'SpaceToDepth', 'Identity', 'Dropout']

export const OPS: Record<string, OpRule> = {
  ...Object.fromEntries(ELEMENTWISE.map(o => [o, npu('Element-wise; broadcasting supported up to rank 5.')])),
  ...Object.fromEntries(ACTIVATIONS.map(o => [o, npu('Activation; fused into the preceding op where the converter can.')])),
  ...Object.fromEntries(LAYOUT.map(o => [o, npu('Data movement; free when folded, costs a copy otherwise.')])),
  Conv: con('1D, 2D and 3D (3D since ONNX Runtime 1.18) convolution, including depthwise and grouped. Weights must be constant.'),
  ConvTranspose: con('2D transposed convolution with constant weights; output_padding supported, dilations > 1 are not.'),
  MatMul: npu('Batched MatMul. Both-activation MatMul (attention) is supported; on HTP the quantized input pairs are (uint8, uint8), (uint8, uint16) and (uint16, uint8).'),
  Gemm: npu('Lowered to FullyConnected; constant B preferred.'),
  MaxPool: npu('2D pooling.'),
  AveragePool: npu('2D pooling; count_include_pad supported.'),
  GlobalAveragePool: npu('Lowered to a mean over H,W.'),
  GlobalMaxPool: npu('Lowered to a max over H,W.'),
  BatchNormalization: npu('Folded into the preceding Conv at conversion when inference-mode.'),
  LayerNormalization: npu('Last-axis normalisation; fp16 or int16 activations recommended for accuracy.'),
  InstanceNormalization: npu(),
  LpNormalization: con('p = 2 only.'),
  ReduceMean: npu('Constant axes required.'),
  ReduceSum: npu('Constant axes required.'),
  ReduceMax: npu('Constant axes required.'),
  ReduceMin: npu('Constant axes required.'),
  ReduceProd: con('Constant axes; precision loss at int8.'),
  ArgMax: con('Single axis, keepdims; output int32.'),
  ArgMin: con('Single axis, keepdims; output int32.'),
  LRN: npu('Local response normalisation.'),
  Gather: con('Positive indices only; constant or int32 indices; large embedding tables may be placed on the CPU by some delegates.'),
  GatherElements: con('Static shapes only.'),
  GatherND: con('Static shapes; batch_dims = 0.'),
  Resize: con('Nearest and bilinear (linear) modes; cubic falls back.'),
  Upsample: con('Deprecated since opset 10; converters treat it as Resize.'),
  TopK: con('Constant k, last axis. Often better in post-processing.'),
  Cast: npu('Between supported types; int64 is narrowed to int32 by the converter.'),
  QuantizeLinear: npu('QDQ node: absorbed into int8/int16 kernels.'),
  DequantizeLinear: npu('QDQ node: absorbed into int8/int16 kernels.'),
  ...Object.fromEntries(['QLinearConv', 'QLinearMatMul', 'QLinearAdd', 'QLinearMul', 'QLinearGlobalAveragePool', 'QLinearAveragePool', 'QLinearConcat', 'QLinearSigmoid', 'QLinearLeakyRelu', 'QLinearSoftmax', 'QGemm'].map(o => [o, cpu(QOP, { replace: QOP_FIX })])),
  ...Object.fromEntries(['DynamicQuantizeLinear', 'MatMulInteger', 'ConvInteger', 'DynamicQuantizeMatMul'].map(o => [o, cpu(DYNQ, { replace: DYNQ_FIX })])),
  MatMulNBits: cpu('Block-quantized int4 weights unpacked at runtime (ONNX Runtime GenAI layout); not an NPU kernel.'),
  TFLite_Detection_PostProcess: cpu('Box decoding and NMS; data-dependent, runs on the CPU.', { post: true }),
  ScatterND: con('Constant indices and static shapes; reduction modes other than none fall back.'),
  GridSample: con('Bilinear / nearest, align_corners supported; 4D only.'),
  LSTM: con('Single-layer, static sequence length; unrolled by the converter. Prefer exporting unrolled.'),
  GRU: con('Static sequence length; unrolled by the converter.'),
  Shape: con('Constant-folded when shapes are static, otherwise needs the CPU.'),
  ConstantOfShape: con('Constant-folded when shapes are static.'),
  Constant: npu('Folded into the graph.'),
  CumSum: con('Constant axis.'),
  OneHot: con('Constant depth.'),
  Einsum: cpu('Only some equations lower; the rest fall back.', { replace: 'Rewrite as Transpose + MatMul (for example bhqd,bhkd->bhqk becomes MatMul with the second input transposed).' }),
  Mod: cpu('Integer remainder has no NPU kernel.', { replace: 'Express as Sub(a, Mul(Floor(Div(a, b)), b)) or move the index math to pre-processing.' }),
  NonZero: cpu('Output shape depends on the data.', { replace: 'Replace with a fixed-size mask (Where / Mul by a 0-1 mask) so shapes stay static.', post: true }),
  NonMaxSuppression: cpu('Data-dependent output; runs on the CPU.', { post: true }),
  Unique: cpu('Data-dependent output.', { post: true }),
  Range: cpu('Needs constant inputs to fold; otherwise CPU.', { replace: 'Precompute the range as a constant initializer for the pinned shape.' }),
  Loop: cpu('Control flow is not offloaded.', { replace: 'Unroll the loop at export time (torch.onnx.export with a fixed trip count).' }),
  If: cpu('Control flow is not offloaded.', { replace: 'Specialise the branch at export time.' }),
  Scan: cpu('Control flow is not offloaded.', { replace: 'Unroll at export time.' }),
  RoiAlign: con('Static number of boxes.'),
  Bernoulli: cpu('Random ops are not supported.'),
  RandomNormal: cpu('Random ops are not supported.'),
  StringNormalizer: cpu('String ops run on the CPU.'),
}

for (const [op, r] of Object.entries(OPS)) {
  if (r.status === 'cpu' || QNN_EP_LISTED.has(op) || STRUCTURAL.has(op) || /^(Q|Dequ)/.test(op)) continue
  OPS[op] = { ...r, status: 'constrained', note: r.note + UNLISTED }
}
OPS.Erf = con('Erf-based GELU is fused into Gelu by ONNX Runtime graph optimisations; a bare Erf is not in the QNN EP operator table.')

export function rule(op: string, domain: string): OpRule {
  // ORT contrib copies of standard quantized ops and TFLite custom ops share the main table.
  if (domain && OPS[op] && (domain === 'custom' || domain === 'com.microsoft') && (OPS[op].post || /^(QLinear|QGemm|Dynamic|MatMulN|QuantizeLinear|DequantizeLinear)/.test(op))) return OPS[op]
  if (domain && domain !== 'ai.onnx' && domain !== 'com.microsoft') return cpu(`Custom op in domain "${domain}"; needs a custom op package for the NPU.`)
  if (domain === 'com.microsoft') {
    const ms: Record<string, OpRule> = {
      FusedConv: npu('ORT fused Conv.'), FusedMatMul: npu('ORT fused MatMul.'), Gelu: npu('ORT Gelu.'), FastGelu: npu('ORT FastGelu.'),
      QuickGelu: npu(), SkipLayerNormalization: con('Lowered to Add + LayerNormalization.'),
      MultiHeadAttention: con('Fused attention: offloaded only with static sequence length; otherwise CPU.'),
      Attention: con('Fused attention: static sequence length required.'),
      GroupQueryAttention: cpu('Uses a KV-cache with dynamic past length; run via the Genie / QNN LLM path instead.'),
      RotaryEmbedding: con('Static positions required.'),
    }
    return ms[op] ?? cpu(`com.microsoft contrib op ${op} is ORT-specific; unlikely to offload.`)
  }
  return OPS[op] ?? (/^[A-Z][A-Z0-9_]+$/.test(op) ? cpu('TFLite builtin not in the curated table.') : cpu('Not in the curated support table; assumed to fall back to the CPU.'))
}
