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
  { name: 'ONNX Runtime QNN Execution Provider, supported operators', url: 'https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html', checked: '2026-11-10' },
  { name: 'Qualcomm AI Engine Direct (QNN) SDK docs, HTP backend op definitions', url: 'https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/HtpOpDefSupplement.html', checked: '2026-11-10' },
  { name: 'LiteRT (TFLite) delegates overview', url: 'https://ai.google.dev/edge/litert/performance/delegates', checked: '2026-11-10' },
] as const

export const TABLE_DATE = '2026-11-10'

const npu = (note = 'Runs on the NPU with static shapes.'): OpRule => ({ status: 'npu', note })
const con = (note: string): OpRule => ({ status: 'constrained', note })
const cpu = (note: string, extra: Partial<OpRule> = {}): OpRule => ({ status: 'cpu', note, ...extra })

const ELEMENTWISE = ['Add', 'Sub', 'Mul', 'Div', 'Abs', 'Neg', 'Exp', 'Log', 'Sqrt', 'Rsqrt', 'Pow', 'Max', 'Min', 'Floor', 'Ceil', 'Round', 'Sign', 'Sin', 'Cos', 'Equal', 'Greater', 'GreaterOrEqual', 'Less', 'LessOrEqual', 'And', 'Or', 'Not', 'Where', 'Erf']
const ACTIVATIONS = ['Relu', 'Sigmoid', 'Tanh', 'LeakyRelu', 'PRelu', 'Elu', 'HardSigmoid', 'HardSwish', 'Gelu', 'Softmax', 'LogSoftmax', 'Clip']
const LAYOUT = ['Reshape', 'Transpose', 'Flatten', 'Squeeze', 'Unsqueeze', 'Concat', 'Split', 'Slice', 'Tile', 'Expand', 'Pad', 'DepthToSpace', 'SpaceToDepth', 'Identity', 'Dropout']

export const OPS: Record<string, OpRule> = {
  ...Object.fromEntries(ELEMENTWISE.map(o => [o, npu('Element-wise; broadcasting supported up to rank 5.')])),
  ...Object.fromEntries(ACTIVATIONS.map(o => [o, npu('Activation; fused into the preceding op where the converter can.')])),
  ...Object.fromEntries(LAYOUT.map(o => [o, npu('Data movement; free when folded, costs a copy otherwise.')])),
  Conv: con('2D (and 1D via reshape) convolution, including depthwise and grouped. 3D convolution falls back. Weights must be constant.'),
  ConvTranspose: con('2D transposed convolution with constant weights; output_padding supported, dilations > 1 are not.'),
  MatMul: npu('Batched MatMul. Both-activation MatMul (attention) is supported; int8 x int8 or fp16.'),
  Gemm: npu('Lowered to FullyConnected; constant B preferred.'),
  MaxPool: npu('2D pooling.'),
  AveragePool: npu('2D pooling; count_include_pad supported.'),
  GlobalAveragePool: npu('Lowered to a mean over H,W.'),
  GlobalMaxPool: npu('Lowered to a max over H,W.'),
  BatchNormalization: npu('Folded into the preceding Conv at conversion when inference-mode.'),
  LayerNormalization: npu('Last-axis normalisation; fp16 or int16 activations recommended for accuracy.'),
  InstanceNormalization: npu(),
  LpNormalization: npu(),
  ReduceMean: npu('Constant axes required.'),
  ReduceSum: npu('Constant axes required.'),
  ReduceMax: npu('Constant axes required.'),
  ReduceMin: npu('Constant axes required.'),
  ReduceProd: con('Constant axes; precision loss at int8.'),
  ArgMax: con('Single axis, keepdims; output int32.'),
  ArgMin: con('Single axis, keepdims; output int32.'),
  Gather: con('Constant or int32 indices; large embedding tables may be placed on the CPU by some delegates.'),
  GatherElements: con('Static shapes only.'),
  GatherND: con('Static shapes; batch_dims = 0.'),
  Resize: con('Nearest and bilinear (linear) modes; cubic falls back.'),
  TopK: con('Constant k, last axis. Often better in post-processing.'),
  Cast: npu('Between supported types; int64 is narrowed to int32 by the converter.'),
  QuantizeLinear: npu('QDQ node: absorbed into int8/int16 kernels.'),
  DequantizeLinear: npu('QDQ node: absorbed into int8/int16 kernels.'),
  QLinearConv: npu('Quantized Conv.'),
  QLinearMatMul: npu('Quantized MatMul.'),
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

export function rule(op: string, domain: string): OpRule {
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
  return OPS[op] ?? (op.startsWith('TFL_') ? cpu('TFLite builtin not in the curated table.') : cpu('Not in the curated support table; assumed to fall back to the CPU.'))
}
