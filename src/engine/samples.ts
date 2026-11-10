// Synthetic ONNX models built from scratch (random weights, no real training). Labelled synthetic everywhere.
import { buildOnnx, type BuildGraph, type BuildTensor } from './onnx.ts'

export interface SampleDef { id: string; name: string; file: string; blurb: string; build: () => Uint8Array; v2?: { file: string; build: () => Uint8Array } }

function rng(seed: number) {
  let s = seed >>> 0
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 2 ** 32 - 0.5 }
}

function w(name: string, dims: number[], seed: number, scale = 0.1): BuildTensor {
  const r = rng(seed)
  const n = dims.reduce((a, b) => a * b, 1)
  const data = new Float32Array(n)
  for (let i = 0; i < n; i++) data[i] = r() * scale
  return { name, dtype: 'fp32', dims, data }
}
const i64 = (name: string, vals: number[], dims = [vals.length]): BuildTensor => ({ name, dtype: 'int64', dims, data: vals })
const f32 = (name: string, vals: number[], dims = [vals.length]): BuildTensor => ({ name, dtype: 'fp32', dims, data: vals })

export function cnnClassifier(): Uint8Array {
  const g: BuildGraph = {
    name: 'tiny_cnn_classifier', producer: 'npu-check synthetic', opset: 17, meta: { synthetic: 'true', note: 'Random weights. Not a trained model.' },
    inputs: [{ name: 'image', dtype: 'fp32', shape: [1, 3, 64, 64] }],
    outputs: [{ name: 'probs', dtype: 'fp32', shape: [1, 10] }],
    inits: [w('conv1.w', [16, 3, 3, 3], 1), w('conv1.b', [16], 2), w('conv2.w', [32, 16, 3, 3], 3), w('conv2.b', [32], 4),
      w('conv3.w', [64, 32, 3, 3], 5), w('conv3.b', [64], 6), w('fc.w', [10, 64], 7), w('fc.b', [10], 8)],
    nodes: [
      { op: 'Conv', name: 'conv1', inputs: ['image', 'conv1.w', 'conv1.b'], outputs: ['c1'], attrs: { kernel_shape: [3, 3], pads: [1, 1, 1, 1] } },
      { op: 'Relu', name: 'relu1', inputs: ['c1'], outputs: ['r1'] },
      { op: 'MaxPool', name: 'pool1', inputs: ['r1'], outputs: ['p1'], attrs: { kernel_shape: [2, 2], strides: [2, 2] } },
      { op: 'Conv', name: 'conv2', inputs: ['p1', 'conv2.w', 'conv2.b'], outputs: ['c2'], attrs: { kernel_shape: [3, 3], pads: [1, 1, 1, 1] } },
      { op: 'Relu', name: 'relu2', inputs: ['c2'], outputs: ['r2'] },
      { op: 'MaxPool', name: 'pool2', inputs: ['r2'], outputs: ['p2'], attrs: { kernel_shape: [2, 2], strides: [2, 2] } },
      { op: 'Conv', name: 'conv3', inputs: ['p2', 'conv3.w', 'conv3.b'], outputs: ['c3'], attrs: { kernel_shape: [3, 3], pads: [1, 1, 1, 1] } },
      { op: 'Relu', name: 'relu3', inputs: ['c3'], outputs: ['r3'] },
      { op: 'GlobalAveragePool', name: 'gap', inputs: ['r3'], outputs: ['g'] },
      { op: 'Flatten', name: 'flatten', inputs: ['g'], outputs: ['f'] },
      { op: 'Gemm', name: 'fc', inputs: ['f', 'fc.w', 'fc.b'], outputs: ['logits'], attrs: { transB: 1 } },
      { op: 'Softmax', name: 'softmax', inputs: ['logits'], outputs: ['probs'], attrs: { axis: -1 } },
    ],
  }
  return buildOnnx(g)
}

export function transformerBlock(): Uint8Array {
  const D = 128, H = 4, Dh = 32, F = 512
  const g: BuildGraph = {
    name: 'tiny_transformer_block', producer: 'npu-check synthetic', opset: 17, meta: { synthetic: 'true', note: 'Random weights, dynamic batch and sequence length.' },
    inputs: [{ name: 'x', dtype: 'fp32', shape: ['batch', 'seq', D] }],
    outputs: [{ name: 'y', dtype: 'fp32', shape: ['batch', 'seq', D] }],
    inits: [
      f32('ln1.g', Array(D).fill(1)), f32('ln1.b', Array(D).fill(0)), w('qkv.w', [D, 3 * D], 11), i64('split', [D, D, D]),
      i64('heads_shape', [0, 0, H, Dh]), i64('merge_shape', [0, 0, D]), f32('scale', [Math.sqrt(Dh)], []), w('o.w', [D, D], 12),
      f32('ln2.g', Array(D).fill(1)), f32('ln2.b', Array(D).fill(0)), w('ff1.w', [D, F], 13), w('ff1.b', [F], 14), w('ff2.w', [F, D], 15),
      f32('sqrt2', [Math.SQRT2], []), f32('one', [1], []), f32('half', [0.5], []),
    ],
    nodes: [
      { op: 'LayerNormalization', name: 'ln1', inputs: ['x', 'ln1.g', 'ln1.b'], outputs: ['h'], attrs: { axis: -1 } },
      { op: 'MatMul', name: 'qkv_proj', inputs: ['h', 'qkv.w'], outputs: ['qkv'] },
      { op: 'Split', name: 'split_qkv', inputs: ['qkv', 'split'], outputs: ['q', 'k', 'v'], attrs: { axis: -1 } },
      { op: 'Reshape', name: 'q_heads', inputs: ['q', 'heads_shape'], outputs: ['q4'] },
      { op: 'Reshape', name: 'k_heads', inputs: ['k', 'heads_shape'], outputs: ['k4'] },
      { op: 'Reshape', name: 'v_heads', inputs: ['v', 'heads_shape'], outputs: ['v4'] },
      { op: 'Transpose', name: 'q_t', inputs: ['q4'], outputs: ['qt'], attrs: { perm: [0, 2, 1, 3] } },
      { op: 'Transpose', name: 'k_t', inputs: ['k4'], outputs: ['kt'], attrs: { perm: [0, 2, 3, 1] } },
      { op: 'Transpose', name: 'v_t', inputs: ['v4'], outputs: ['vt'], attrs: { perm: [0, 2, 1, 3] } },
      { op: 'MatMul', name: 'scores', inputs: ['qt', 'kt'], outputs: ['s'] },
      { op: 'Div', name: 'scale_scores', inputs: ['s', 'scale'], outputs: ['ss'] },
      { op: 'Softmax', name: 'attn_softmax', inputs: ['ss'], outputs: ['p'], attrs: { axis: -1 } },
      { op: 'MatMul', name: 'attn_v', inputs: ['p', 'vt'], outputs: ['ctx'] },
      { op: 'Transpose', name: 'ctx_t', inputs: ['ctx'], outputs: ['ctxt'], attrs: { perm: [0, 2, 1, 3] } },
      { op: 'Reshape', name: 'merge_heads', inputs: ['ctxt', 'merge_shape'], outputs: ['m'] },
      { op: 'MatMul', name: 'o_proj', inputs: ['m', 'o.w'], outputs: ['o'] },
      { op: 'Add', name: 'residual1', inputs: ['x', 'o'], outputs: ['r1'] },
      { op: 'LayerNormalization', name: 'ln2', inputs: ['r1', 'ln2.g', 'ln2.b'], outputs: ['h2'], attrs: { axis: -1 } },
      { op: 'MatMul', name: 'ff1', inputs: ['h2', 'ff1.w'], outputs: ['f1'] },
      { op: 'Add', name: 'ff1_bias', inputs: ['f1', 'ff1.b'], outputs: ['f1b'] },
      { op: 'Div', name: 'gelu_div', inputs: ['f1b', 'sqrt2'], outputs: ['gd'] },
      { op: 'Erf', name: 'gelu_erf', inputs: ['gd'], outputs: ['ge'] },
      { op: 'Add', name: 'gelu_add', inputs: ['ge', 'one'], outputs: ['ga'] },
      { op: 'Mul', name: 'gelu_mul', inputs: ['f1b', 'ga'], outputs: ['gm'] },
      { op: 'Mul', name: 'gelu_half', inputs: ['gm', 'half'], outputs: ['gelu'] },
      { op: 'MatMul', name: 'ff2', inputs: ['gelu', 'ff2.w'], outputs: ['f2'] },
      { op: 'Add', name: 'residual2', inputs: ['r1', 'f2'], outputs: ['y'] },
    ],
  }
  return buildOnnx(g)
}

export function detector(fixed = false): Uint8Array {
  const g: BuildGraph = {
    name: 'tiny_detector_with_unsupported_ops', producer: 'npu-check synthetic', opset: 17, meta: { synthetic: 'true', note: 'Random weights; Einsum, Mod and NonMaxSuppression included on purpose.' },
    inputs: [{ name: 'image', dtype: 'fp32', shape: [1, 3, 128, 128] }],
    outputs: [{ name: 'detections', dtype: 'fp32', shape: ['?', 4] }],
    inits: [
      w('b1.w', [16, 3, 3, 3], 21), w('b2.w', [32, 16, 3, 3], 22), w('b3.w', [64, 32, 3, 3], 23), w('mix.w', [64, 64], 24),
      w('head.w', [6, 64, 1, 1], 25), i64('flat_shape', [1, 64, 256]), i64('map_shape', [1, 64, 16, 16]), i64('head_shape', [1, 6, 256]),
      i64('s0', [0]), i64('s4', [4]), i64('s6', [6]), i64('ax2', [2]), { name: 'cell_idx', dtype: 'int64', dims: [256], data: Array.from({ length: 256 }, (_, i) => i) },
      i64('grid_w', [16], []), i64('grid_shape', [1, 256, 1]), i64('box_flat', [-1, 4]),
      i64('max_out', [100]), f32('iou', [0.5]), f32('score_th', [0.25]), i64('col', [2], []),
    ],
    nodes: [
      { op: 'Conv', name: 'backbone1', inputs: ['image', 'b1.w'], outputs: ['x1'], attrs: { kernel_shape: [3, 3], strides: [2, 2], pads: [1, 1, 1, 1] } },
      { op: 'Relu', name: 'act1', inputs: ['x1'], outputs: ['a1'] },
      { op: 'Conv', name: 'backbone2', inputs: ['a1', 'b2.w'], outputs: ['x2'], attrs: { kernel_shape: [3, 3], strides: [2, 2], pads: [1, 1, 1, 1] } },
      { op: 'Relu', name: 'act2', inputs: ['x2'], outputs: ['a2'] },
      { op: 'Conv', name: 'backbone3', inputs: ['a2', 'b3.w'], outputs: ['x3'], attrs: { kernel_shape: [3, 3], strides: [2, 2], pads: [1, 1, 1, 1] } },
      { op: 'Relu', name: 'act3', inputs: ['x3'], outputs: ['a3'] },
      { op: 'Reshape', name: 'to_tokens', inputs: ['a3', 'flat_shape'], outputs: ['tok'] },
      { op: 'Einsum', name: 'channel_mix', inputs: ['tok', 'mix.w'], outputs: ['mixed'], attrs: { equation: 'bcn,cd->bdn' } },
      { op: 'Reshape', name: 'to_map', inputs: ['mixed', 'map_shape'], outputs: ['fm'] },
      { op: 'Conv', name: 'det_head', inputs: ['fm', 'head.w'], outputs: ['hd'], attrs: { kernel_shape: [1, 1] } },
      { op: 'Reshape', name: 'head_flat', inputs: ['hd', 'head_shape'], outputs: ['hf'] },
      { op: 'Transpose', name: 'head_t', inputs: ['hf'], outputs: ['ht'], attrs: { perm: [0, 2, 1] } },
      { op: 'Slice', name: 'boxes_raw', inputs: ['ht', 's0', 's4', 'ax2'], outputs: ['br'] },
      { op: 'Slice', name: 'scores_raw', inputs: ['ht', 's4', 's6', 'ax2'], outputs: ['sr'] },
      { op: 'Sigmoid', name: 'scores_sig', inputs: ['sr'], outputs: ['sc'] },
      { op: 'Transpose', name: 'scores_t', inputs: ['sc'], outputs: ['sct'], attrs: { perm: [0, 2, 1] } },
      { op: 'Mod', name: 'grid_x', inputs: ['cell_idx', 'grid_w'], outputs: ['gx'] },
      { op: 'Cast', name: 'grid_float', inputs: ['gx'], outputs: ['gf'], attrs: { to: 1 } },
      { op: 'Reshape', name: 'grid_bcast', inputs: ['gf', 'grid_shape'], outputs: ['gb'] },
      { op: 'Add', name: 'decode_boxes', inputs: ['br', 'gb'], outputs: ['boxes'] },
      { op: 'NonMaxSuppression', name: 'nms', inputs: ['boxes', 'sct', 'max_out', 'iou', 'score_th'], outputs: ['sel'] },
      { op: 'Gather', name: 'sel_index', inputs: ['sel', 'col'], outputs: ['idx'], attrs: { axis: 1 } },
      { op: 'Reshape', name: 'boxes_flat', inputs: ['boxes', 'box_flat'], outputs: ['bf'] },
      { op: 'Gather', name: 'gather_boxes', inputs: ['bf', 'idx'], outputs: ['detections'], attrs: { axis: 0 } },
    ],
  }
  if (fixed) {
    // v2: Einsum rewritten as MatMul on transposed tokens, grid precomputed as a constant, NMS cut out to the app.
    g.name = 'tiny_detector_v2'
    g.inits.push({ name: 'grid_const', dtype: 'fp32', dims: [1, 256, 1], data: Array.from({ length: 256 }, (_, i) => i % 16) })
    const keep = g.nodes.filter(n => !['channel_mix', 'grid_x', 'grid_float', 'grid_bcast', 'nms', 'sel_index', 'boxes_flat', 'gather_boxes'].includes(n.name!))
    const at = keep.findIndex(n => n.name === 'to_map')
    keep.splice(at, 0,
      { op: 'Transpose', name: 'tok_t', inputs: ['tok'], outputs: ['tokt'], attrs: { perm: [0, 2, 1] } },
      { op: 'MatMul', name: 'channel_mix', inputs: ['tokt', 'mix.w'], outputs: ['mixt'] },
      { op: 'Transpose', name: 'mix_back', inputs: ['mixt'], outputs: ['mixed'], attrs: { perm: [0, 2, 1] } })
    keep.find(n => n.name === 'decode_boxes')!.inputs = ['br', 'grid_const']
    g.nodes = keep
    g.outputs = [{ name: 'boxes', dtype: 'fp32', shape: [1, 256, 4] }, { name: 'sct', dtype: 'fp32', shape: [1, 2, 256] }]
    g.meta = { synthetic: 'true', note: 'v2: NPU-friendly rewrite of the synthetic detector; NMS runs in the app.' }
  }
  return buildOnnx(g)
}

export const SAMPLES: SampleDef[] = [
  { id: 'detector', name: 'Detector with unsupported ops', file: 'synthetic_detector.onnx', blurb: 'Einsum, Mod and NMS split the graph into five partitions.', build: () => detector(), v2: { file: 'synthetic_detector_v2.onnx', build: () => detector(true) } },
  { id: 'transformer', name: 'Transformer block, dynamic sequence', file: 'synthetic_transformer_block.onnx', blurb: 'Symbolic batch and seq dims push every node to the CPU until pinned.', build: transformerBlock },
  { id: 'cnn', name: 'Tiny CNN classifier', file: 'synthetic_cnn.onnx', blurb: 'Fully supported fp32 graph; the remaining win is int8 QDQ.', build: cnnClassifier },
]
