# NPU Check

An on-device model readiness inspector. Load an ONNX, TFLite or GGUF file in the browser and see what will run on a mobile NPU before you spend an afternoon in a converter.

**Live:** https://npu-check.vercel.app

## The pain

Edge ML engineers export a model, convert it for a Snapdragon device, and find it barely faster than the CPU. The usual cause is the graph, not the hardware: an unsupported op mid-network, a dynamic sequence length, fp32 weights, or post-processing that bounces between CPU and NPU. NPU Check shows that up front.

## Features

- Hand-written readers: ONNX protobuf (with external data), TFLite FlatBuffer, GGUF header (multi-GB files read in chunks). Nothing is uploaded.
- Op inventory with counts, MACs and a curated, dated support table (NPU, NPU with constraints, CPU fallback).
- Shape inference for common ops with dynamic dims flagged; parameter count and bytes by precision (fp32/fp16/int8/int4).
- Peak activation memory from topological liveness, partitions and CPU/NPU switch counts.
- Quantization readiness (QDQ, per-channel) and a roofline latency estimate on editable device profiles.
- Fixes: a shape-pinned ONNX file generated in the browser, op rewrites, a graph cut for post-processing, QDQ quantization and on-device verification snippets. Simulate each fix and watch partitions and latency change.
- Graph viewer with search, op and node CSV exports, printable readiness report.
- Projects with versions and a compare view. Optional sign-in syncs reports across devices (model files never leave the device).
- Three synthetic sample models built in code (random weights, labelled synthetic), plus four real public models you can load from the import screen (MNIST-12 from the ONNX Model Zoo, Qualcomm AI Hub MobileNet-v2 w8a8, MediaPipe BlazeFace, TinyStories 260K GGUF).

## Tested against real files

The parsers and the analysis are run against public models: ONNX Model Zoo (MNIST, SqueezeNet 1.1, MobileNet-v2 float / QDQ / QOperator, Tiny YOLOv3), Hugging Face exports (all-MiniLM-L6-v2 float and dynamically quantized, Whisper tiny encoder fp16, Phi-3 mini int4 with external data), Qualcomm AI Hub MobileNet-v2 (ONNX and TFLite), MediaPipe and TensorFlow TFLite models (BlazeFace, EfficientDet-Lite0, EfficientNet-Lite0, MobileNet-v1 quant, SSD MobileNet-v1), and GGUF headers (TinyStories, Qwen2.5 0.5B, SmolLM2 135M, nomic-embed-text). Published parameter and MAC counts are reproduced to within 1%, and the latency estimate for MobileNet-v2 lands within about 2x of Qualcomm AI Hub's measured Snapdragon X Elite numbers. Small real files live in `public/samples` and `src/engine/fixtures` and are used by the tests; sources and licences are on the Docs page.

## Known gaps

- The op support table is curated from public docs (ONNX Runtime QNN EP operator list, checked 2026-11-15). It is not a converter: confirm with partition logs.
- Latency is a roofline estimate, right to within roughly a factor of two for graphs that stay on the NPU.
- Control-flow subgraphs (Loop, If) are flagged but not analysed inside. TFLite models are analysed on the main subgraph only.
- GGUF files carry no operator graph, so LLM results are size, memory and throughput estimates only.

## Run locally

```sh
npm install
npm run dev      # http://localhost:5173
npm test         # engine tests
npm run build
```

Copy `.env.example` to `.env` for optional sync (public Supabase URL and publishable key).

Built by Aryan Kansagara.
