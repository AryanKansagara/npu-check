import { DEVICES } from './engine/analyze.ts'
import { OPS, SOURCES, TABLE_DATE } from './engine/support.ts'
import { Code } from './ui.tsx'

const TOC = [['export', 'Exporting your model'], ['formats', 'What is read from each format'], ['support', 'Op support table'], ['estimates', 'How estimates work'], ['devices', 'Device profiles'], ['privacy', 'Privacy and sync']] as const

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
            <li><b>ONNX</b>: ModelProto, GraphProto, NodeProto, TensorProto and ValueInfoProto through a hand-written protobuf decoder. Weight bytes are skipped, not decoded. Small integer constants (Reshape shapes, axes) are kept for shape inference.</li>
            <li><b>TFLite</b>: the FlatBuffer Model table: operator codes, the main subgraph's tensors, operators and buffers, plus quantization scales (per-channel when more than one scale) and <code>shape_signature</code> for dynamic dims. Builtins are mapped to their ONNX equivalents so one support table covers both.</li>
            <li><b>GGUF</b> v2 and v3: metadata key/values and tensor infos. k-quant and i-quant block sizes give exact byte sizes per tensor.</li>
          </ul>
          <p>The import step validates the graph and reports problems with their location (node index and name): inputs with no producer, outputs produced twice, nodes without an op type, unknown data types.</p>

          <h2 id="support">Op support table</h2>
          <p>Curated by hand, reviewed <b>{TABLE_DATE}</b>, from public documentation for NPU-style delegates:</p>
          <ul>{SOURCES.map(s => <li key={s.url}><a href={s.url} target="_blank" rel="noreferrer">{s.name}</a> (checked {s.checked})</li>)}</ul>
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
            <li><b>Peak activation memory</b>: nodes in topological order; a tensor is allocated when produced and freed after its last consumer (graph outputs live to the end). Weights are excluded. The peak is the largest live total.</li>
            <li><b>Partitions</b>: connected groups of nodes placed on the same unit. <b>Switches</b>: distinct tensors that cross between NPU and CPU.</li>
            <li><b>Latency</b>: per node, the larger of compute time (FLOPs over TOPS × utilisation; fp16 at half the int8 rate; CPU nodes at CPU GFLOPS) and memory time (bytes read and written over DRAM bandwidth). Each switch adds the profile's fixed cost plus the tensor copy. fp32 graphs are assumed to run as fp16 on the NPU.</li>
            <li><b>LLMs (GGUF)</b>: decode is memory-bound, so tokens per second ≈ bandwidth × utilisation × 2 / weight bytes; prefill assumes compute-bound NPU matmuls; KV cache = 2 × layers × tokens × width × (KV heads / heads) × 2 bytes.</li>
            <li><b>Dynamic dims</b> use the values you map at import (batch 1, sequence 128 and image 224 are suggested).</li>
          </ul>

          <h2 id="devices">Device profiles</h2>
          <p>Profiles are editable per project and in Settings. Defaults, and what they are based on:</p>
          <table className="compact-table">
            <thead><tr><th>Profile</th><th className="num">TOPS</th><th className="num">GB/s</th><th className="num">CPU GFLOPS</th><th>Basis</th></tr></thead>
            <tbody>{DEVICES.map(d => <tr key={d.id}><td>{d.name}</td><td className="num">{d.npuTops}</td><td className="num">{d.memGBs}</td><td className="num">{d.cpuGflops}</td><td className="small">{d.note}</td></tr>)}</tbody>
          </table>
          <p className="muted small">Figures as of {TABLE_DATE}. Measure your own device with a microbenchmark and enter the numbers for estimates you can quote.</p>

          <h2 id="privacy">Privacy and sync</h2>
          <p>Files are parsed in your browser. Projects, versions and parsed graphs (without weights) are stored in IndexedDB on this device. Signing in is optional: it syncs project settings, version summaries and parsed graphs under 1.5 MB. Model files are never uploaded. Export everything as JSON from the account menu at any time.</p>
        </div>
      </div>
    </section>
  )
}
