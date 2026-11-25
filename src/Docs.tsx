import { DEVICES } from './engine/analyze.ts'
import { REAL_SAMPLES } from './engine/samples.ts'
import { OPS, QNN_EP_LISTED, SOURCES, TABLE_DATE } from './engine/support.ts'
import { Code } from './ui.tsx'

const TOC = [['export', 'Exporting your model'], ['formats', 'What is read from each format'], ['support', 'Op support table'], ['estimates', 'How estimates work'], ['reference', 'Checked against measurements'], ['devices', 'Device profiles'], ['samples', 'Sample models'], ['privacy', 'Privacy and sync']] as const

export default function Docs() {
  const groups = (['npu', 'constrained', 'cpu'] as const).map(s => [s, Object.entries(OPS).filter(([, r]) => r.status === s)] as const)
  return (
    <section className="page docs-page">
      <div className="docs-grid">
        <nav className="toc" aria-label="On this page">{TOC.map(([id, l]) => <a key={id} href={`#${id}`}>{l}</a>)}</nav>
        <div className="docs">
          <div className="eyebrow">Docs</div>
          <h1>Getting a model into NPU Check</h1>
          <p className="lede">NPU Check reads the file you are about to convert. Below: exactly how to produce it from each toolchain, what is read, and every assumption behind the numbers.</p>

          <h2 id="export">Exporting your model</h2>
          <h3>PyTorch to ONNX</h3>
          <p>Export with a recent opset. Leave the batch and sequence dims dynamic if that is how your model is exported today: NPU Check shows what that costs, and generates the pinned file for you.</p>
          <Code label="Python" code={`import torch\n\nmodel.eval()\nexample = torch.randn(1, 3, 224, 224)\ntorch.onnx.export(\n    model, example, "model.onnx",\n    opset_version=17,\n    input_names=["image"], output_names=["logits"],\n    # dynamic_axes={"image": {0: "batch"}},  # remove for a static NPU graph\n)`} />
          <p>Models over 2 GB are written with external data: a small <code>model.onnx</code> plus <code>model.onnx.data</code>. Drop only the <code>.onnx</code>; tensor sizes come from their dims, so the weights are never needed.</p>
          <p>Optional: running ONNX shape inference first stores intermediate shapes in the file, which NPU Check uses directly instead of its own inference.</p>
          <Code label="Python" code={`import onnx\nfrom onnx import shape_inference\n\nm = shape_inference.infer_shapes(onnx.load("model.onnx", load_external_data=False))\nonnx.save(m, "model_shapes.onnx")`} />
          <h3>Hugging Face models</h3>
          <Code label="Shell" code={`pip install optimum[exporters]\noptimum-cli export onnx --model <model-id> --task <task> out_dir/`} />
          <h3>TensorFlow / Keras to TFLite (LiteRT)</h3>
          <Code label="Python" code={`import tensorflow as tf\n\nconverter = tf.lite.TFLiteConverter.from_keras_model(model)\nopen("model.tflite", "wb").write(converter.convert())`} />
          <h3>GGUF</h3>
          <p>Any <code>.gguf</code> from llama.cpp tooling. Only the header (metadata and tensor table) is read, in growing chunks, so a 7 GB file costs a few MB of reads.</p>

          <h2 id="formats">What is read from each format</h2>
          <ul>
            <li><b>ONNX</b>: ModelProto, GraphProto, NodeProto, TensorProto and ValueInfoProto through a hand-written protobuf decoder. Weight bytes are skipped, not decoded. Small integer constants (Reshape shapes, axes) and tiny float constants (Resize scales) are kept, and shape arithmetic done in the graph (Shape, Gather, Concat, Slice feeding a Reshape) is followed so exported transformers resolve to concrete shapes once dims are pinned.</li>
            <li><b>TFLite</b>: the FlatBuffer Model table: operator codes, the main subgraph's tensors, operators and buffers, plus quantization scales (per-channel when more than one scale) and <code>shape_signature</code> for dynamic dims. All {`209`} builtin operator codes are named from the LiteRT <code>schema.fbs</code>; the common ones are mapped to their ONNX equivalents so one support table covers both. <code>DEQUANTIZE</code> on fp16 constants is treated as a weight cast, not as quantization.</li>
            <li><b>GGUF</b> v2 and v3: metadata key/values and tensor infos. k-quant and i-quant block sizes give exact byte sizes per tensor. Encoder-only architectures (BERT-style embedding models) are reported as a single pass with no KV cache.</li>
          </ul>
          <p>The import step validates the graph and reports problems with their location (node index and name): inputs with no producer, outputs produced twice, nodes without an op type, unknown data types.</p>

          <h2 id="support">Op support table</h2>
          <p>Curated by hand, reviewed <b>{TABLE_DATE}</b>, from public documentation for NPU-style delegates:</p>
          <ul>{SOURCES.map(s => <li key={s.url}><a href={s.url} target="_blank" rel="noreferrer">{s.name}</a> (checked {s.checked})</li>)}</ul>
          <p>The ONNX Runtime page lists {QNN_EP_LISTED.size} standard operators for the QNN EP. Ops in this table that the page does not list (it omits even Reshape) are shown as supported with constraints and say so in their note. The same page states two model requirements that drive most findings here: dynamic shapes must be fixed, and the HTP backend expects quantized QDQ models, so QOperator graphs (QLinearConv and friends) and dynamically quantized graphs (MatMulInteger, DynamicQuantizeLinear) are flagged for re-quantization.</p>
          <p>It is a first-pass heuristic, not an official compatibility list. Support moves with every QAIRT and ONNX Runtime release, and with the SoC generation. The ground truth is the converter or a session with CPU fallback disabled (see the Fixes tab). On top of the table, any node whose output shape is symbolic or data-dependent is placed on the CPU, because NPU graphs are compiled for fixed shapes.</p>
          {groups.map(([s, rows]) => (
            <details key={s}>
              <summary><b>{s === 'npu' ? 'Supported' : s === 'constrained' ? 'Supported with constraints' : 'Falls back to the CPU'}</b> ({rows.length})</summary>
              <table className="compact-table"><tbody>{rows.map(([op, r]) => <tr key={op}><td className="mono">{op}</td><td className="small">{r.note}{r.replace ? ` Fix: ${r.replace}` : ''}</td></tr>)}</tbody></table>
            </details>
          ))}

          <h2 id="estimates">How estimates work</h2>
          <ul>
            <li><b>MACs</b>: Conv = output elements × (Cin / groups) × kH × kW; MatMul = output elements × K; Gemm the same with transA/transB; Einsum from its equation. Other ops count one FLOP per output element.</li>
            <li><b>Peak activation memory</b>: nodes in topological order; a tensor is allocated when produced and freed after its last consumer (graph outputs live to the end). Weights are excluded. A tensor the file gives no type for has the type its operator produces (the input's type for element-wise ops). The peak is the largest live total.</li>
            <li><b>Partitions</b>: connected groups of nodes placed on the same unit. <b>Switches</b>: distinct tensors that cross between NPU and CPU.</li>
            <li><b>Latency</b>: per node, the larger of compute time (FLOPs over TOPS × utilisation; fp16 at half the int8 rate; CPU nodes at CPU GFLOPS) and memory time (bytes read and written over DRAM bandwidth). Each switch adds the profile's fixed cost plus the tensor copy. fp32 graphs are assumed to run as fp16 on the NPU.</li>
            <li><b>LLMs (GGUF)</b>: decode is memory-bound, so tokens per second ≈ bandwidth × utilisation × 2 / weight bytes; prefill assumes compute-bound NPU matmuls; KV cache = layers × tokens × KV heads × (key length + value length) × 2 bytes, where each length is the file's <code>attention.key_length</code> / <code>value_length</code>, or width / heads without one.</li>
            <li><b>Dynamic dims</b> use the values you map at import (batch 1, sequence 128 and image 224 are suggested; unnamed dims in the spatial slots of an image input default to 224).</li>

          <li><b>Constant folding</b>: Constant nodes, nodes fed only by initializers (weight DequantizeLinear in QDQ files) and Shape of a static tensor are folded. They cost nothing and never form a partition.</li>
            <li><b>Quantized graphs</b>: in QDQ and QOperator files, tensors between the quantize nodes are counted at the activation zero-point width (1 or 2 bytes), and Q/DQ nodes themselves are free. fp32 weights that only feed a QuantizeLinear count as int8.</li>
          </ul>

          <h2 id="reference">Checked against measurements</h2>
          <p>The estimate is a roofline, so it is only worth showing if it lands near real devices. Qualcomm AI Hub publishes measured on-device inference times for the models it exports. For MobileNet-v2 (224 x 224, 3.49 M parameters, 300.8 M MACs by NPU Check's count on Qualcomm's own export), ONNX Runtime with QAIRT 2.50:</p>
          <table className="compact-table">
            <thead><tr><th>Chipset</th><th className="num">float, measured</th><th className="num">w8a8, measured</th></tr></thead>
            <tbody>
              <tr><td>Snapdragon X Elite</td><td className="num">0.626 ms</td><td className="num">0.454 ms</td></tr>
              <tr><td>Snapdragon 8 Elite</td><td className="num">0.335 ms</td><td className="num">0.283 ms</td></tr>
              <tr><td>Snapdragon 8 Gen 3</td><td className="num">0.420 ms</td><td className="num">0.326 ms</td></tr>
              <tr><td>Dragonwing QCS6490</td><td className="num">not listed</td><td className="num">1.064 ms</td></tr>
            </tbody>
          </table>
          <p>On the default 45 TOPS laptop profile NPU Check estimates 0.90 ms for the float export and 0.23 ms for the w8a8 export of the same model: about 1.4x above and 2x below the Snapdragon X Elite measurements. Read every latency figure here as right to within a factor of about two for graphs that stay on the NPU, and edit the profile if you have your own measurements. Source: <a href="https://huggingface.co/qualcomm/MobileNet-v2" target="_blank" rel="noreferrer">qualcomm/MobileNet-v2 performance summary</a>, retrieved 2026-11-13. Published parameter and MAC counts for MobileNet-v1, MobileNet-v2, SqueezeNet 1.1 and EfficientNet-Lite0 were also reproduced to within 1%.</p>

          <h2 id="devices">Device profiles</h2>
          <p>Profiles are editable per project and in Settings. Defaults, and what they are based on:</p>
          <table className="compact-table">
            <thead><tr><th>Profile</th><th className="num">TOPS</th><th className="num">GB/s</th><th className="num">CPU GFLOPS</th><th>Basis</th></tr></thead>
            <tbody>{DEVICES.map(d => <tr key={d.id}><td>{d.name}</td><td className="num">{d.npuTops}</td><td className="num">{d.memGBs}</td><td className="num">{d.cpuGflops}</td><td className="small">{d.note}</td></tr>)}</tbody>
          </table>
          <p className="muted small">Figures as of {TABLE_DATE}. Measure your own device with a microbenchmark and enter the numbers for estimates you can quote.</p>

          <h2 id="samples">Sample models</h2>
          <p>The three workspace samples are synthetic graphs built in code with random weights, and are labelled synthetic. The import screen also offers real public models, served unmodified:</p>
          <table className="compact-table">
            <thead><tr><th>File</th><th>Source</th><th>Licence</th></tr></thead>
            <tbody>{REAL_SAMPLES.map(s => <tr key={s.file}><td className="mono small">{s.file}</td><td className="small"><a href={s.url} target="_blank" rel="noreferrer">{s.source}</a>: {s.what}</td><td className="small">{s.licence}</td></tr>)}</tbody>
          </table>

          <h2 id="privacy">Privacy and sync</h2>
          <p>Files are parsed in your browser. Projects, versions and parsed graphs (without weights) are stored in IndexedDB on this device. Signing in is optional: it syncs project settings, version summaries and parsed graphs under 1.5 MB. Model files are never uploaded. Export everything as JSON from the account menu at any time.</p>
        </div>
      </div>
    </section>
  )
}
