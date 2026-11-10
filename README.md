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
- Three synthetic sample models built in code (random weights, labelled synthetic).

## Run locally

```sh
npm install
npm run dev      # http://localhost:5173
npm test         # engine tests
npm run build
```

Copy `.env.example` to `.env` for optional sync (public Supabase URL and publishable key).

Built by Aryan Kansagara. Independent project, not affiliated with Qualcomm.
