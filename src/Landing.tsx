import { useEffect, useMemo, useRef, useState } from 'react'
import { analyze, DEVICES, FIX_LABELS, fmtBytes, type FixId } from './engine/analyze.ts'
import { SAMPLES } from './engine/samples.ts'
import { GraphView, Legend } from './GraphView.tsx'
import { sampleIr } from './projectData.ts'
import { fmtMs, fmtN, fmtSi, Link } from './ui.tsx'

const HERO_FIXES: { id: FixId; what: string }[] = [
  { id: 'replace-ops', what: 'Einsum to MatMul, Mod to a constant grid' },
  { id: 'split-post', what: 'Cut NMS out of the NPU graph' },
  { id: 'quantize-int8', what: 'QDQ int8, per-channel weights' },
]

export function useTween(target: number, ms = 650) {
  const [v, setV] = useState(target)
  const from = useRef(target)
  useEffect(() => {
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) { setV(target); from.current = target; return }
    const a = from.current, t0 = performance.now()
    let raf = 0
    const step = (t: number) => {
      const k = Math.min(1, (t - t0) / ms)
      const cur = a + (target - a) * (1 - (1 - k) ** 3)
      setV(cur)
      from.current = cur
      if (k < 1) raf = requestAnimationFrame(step)
    }
    raf = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf)
  }, [target, ms])
  return v
}

export function Landing() {
  const ir = useMemo(() => sampleIr(SAMPLES[0].file)!, [])
  const [on, setOn] = useState<FixId[]>([])
  const narrow = useMemo(() => matchMedia('(max-width: 720px)').matches, [])
  const touched = useRef(false)
  useEffect(() => {
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) return
    const t = setTimeout(() => { if (!touched.current) setOn(['replace-ops']) }, 2600)
    return () => clearTimeout(t)
  }, [])
  const base = useMemo(() => analyze(ir, { pins: {}, applied: [], device: DEVICES[1] }), [ir])
  const a = useMemo(() => analyze(ir, { pins: {}, applied: on, device: DEVICES[1] }), [ir, on])
  const parts = useTween(a.partitions), sw = useTween(a.switches), ms = useTween(a.latency.totalMs), score = useTween(a.score)
  const toggle = (id: FixId) => { touched.current = true; setOn(o => o.includes(id) ? o.filter(x => x !== id) : [...o, id]) }

  return (
    <>
      <section className="hero">
        <div className="hero-inner">
          <div className="hero-copy">
            <div className="eyebrow">On-device model readiness</div>
            <h1>Know what will run on the NPU <span className="accent-text">before</span> you convert.</h1>
            <p className="lede">Drop an ONNX, TFLite or GGUF file. NPU Check reads the graph in your browser, flags the ops, dynamic shapes and fp32 paths that fall back to the CPU, counts the CPU/NPU switches, and estimates memory and latency on the device profile you choose.</p>
            <div className="hero-ctas">
              <Link href="/app?sample=detector" className="btn primary lg">Open the sample model</Link>
              <Link href="/app" className="btn outline lg">Inspect your model</Link>
            </div>
            <p className="hero-note">Nothing is uploaded. No account needed. Sample models are synthetic, with random weights.</p>
          </div>
          <div className="hero-card" aria-label="Live demo on a synthetic detector model">
            <div className="hero-card-head">
              <span className="mono small">synthetic_detector.onnx</span>
              <span className="badge synthetic">Synthetic</span>
              <span className="grow" />
              <span className="muted small">{DEVICES[1].name}</span>
            </div>
            <div className="hero-kpis">
              <div className={`kpi-mini big${a.partitions < base.partitions ? ' good' : ''}`}><small>Partitions</small><b className="tab-nums">{Math.round(parts)}</b>{a.partitions < base.partitions && <span className="delta">was {base.partitions}</span>}</div>
              <div className="kpi-mini"><small>CPU/NPU switches</small><b className="tab-nums">{Math.round(sw)}</b></div>
              <div className="kpi-mini"><small>Est. latency</small><b className="tab-nums">{fmtMs(ms)}</b></div>
              <div className="kpi-mini"><small>Readiness</small><b className="tab-nums">{Math.round(score)}</b></div>
            </div>
            <div className="hero-graph">
              <GraphView ir={ir} a={a} dir="LR" wrap={narrow ? 4 : 6} compact animate label="Synthetic detector graph; blue nodes run on the NPU, red nodes fall back to the CPU" />
            </div>
            <div className="fix-toggles" role="group" aria-label="Suggested fixes">
              {HERO_FIXES.map(f => {
                const active = on.includes(f.id)
                return (
                  <label key={f.id} className={`fix-toggle${active ? ' on' : ''}`}>
                    <span className="switch"><input type="checkbox" checked={active} onChange={() => toggle(f.id)} /><span /></span>
                    <span className="ft-text"><b>{FIX_LABELS[f.id]}</b><small>{f.what}</small></span>
                  </label>
                )
              })}
            </div>
            <Legend />
          </div>
        </div>
      </section>

      <section className="stats-band" aria-label="What the sample shows">
        <div className="stats-inner">
          <div><b>{fmtSi(base.macs)}</b><span>MACs counted per op</span></div>
          <div><b>{fmtBytes(base.peakBytes)}</b><span>peak activations by liveness</span></div>
          <div><b>{base.opRows.length}</b><span>op types checked against the support table</span></div>
          <div><b>0 bytes</b><span>uploaded anywhere</span></div>
        </div>
      </section>

      <section className="section">
        <div className="section-head">
          <div className="eyebrow">What edge ML engineers keep asking</div>
          <h2>&ldquo;Is it the hardware or the software?&rdquo; Usually it is the graph.</h2>
        </div>
        <div className="quotes">
          <figure className="quote"><blockquote>High hopes, small models</blockquote><p>A thread on Hexagon NPU inference for phones: people expected big speedups and found the NPU path works mainly for small, carefully prepared models, and could not tell whether hardware or tooling was the limit.</p><figcaption className="src">r/LocalLLM, on QAIRT and Genie based inference</figcaption></figure>
          <figure className="quote"><blockquote>&ldquo;Open-source app for the NPU&rdquo;</blockquote><p>A developer shipped an app to run LLMs on the Snapdragon X NPU; most of the discussion was about which models convert, which ops break and what falls back.</p><figcaption className="src">r/SnapdragonLaptops</figcaption></figure>
          <figure className="quote"><blockquote>GGUF on Hexagon?</blockquote><p>Users asking whether their existing GGUF files can use the Hexagon NPU at all, and what the memory and quantization constraints are before trying.</p><figcaption className="src">r/LocalLLaMA</figcaption></figure>
        </div>
        <p className="muted small">Paraphrased from public threads.</p>
      </section>

      <section className="section">
        <div className="section-head">
          <div className="eyebrow">The workflow</div>
          <h2>From an exported file to a fix list in one sitting.</h2>
        </div>
        <div className="features">
          {FEATURES.map(f => <article className="feature" key={f.t}><div className="ico" aria-hidden="true">{f.i}</div><h3>{f.t}</h3><p>{f.b}</p></article>)}
        </div>
      </section>

      <section className="section">
        <div className="grid-2 align-center">
          <div>
            <div className="eyebrow">Transparent estimates</div>
            <h2>A roofline you can argue with.</h2>
            <p className="muted">Each op costs the larger of its compute time (MACs over TOPS times utilisation, halved for fp16) and its memory time (bytes moved over DRAM bandwidth). Every CPU/NPU hand-off adds a fixed cost plus the copy. All device numbers are editable; profiles say which figures are published and which are assumptions.</p>
            <Link href="/docs#estimates" className="btn ghost">How estimates work</Link>
          </div>
          <div className="panel">
            <table className="compact-table">
              <thead><tr><th>Op</th><th className="num">Count</th><th>Runs on</th></tr></thead>
              <tbody>
                {base.opRows.slice(0, 7).map(r => (
                  <tr key={r.op}><td className="mono">{r.op}</td><td className="num">{fmtN(r.count)}</td><td><span className={`pill ${r.cpu ? 'cpu' : r.support === 'constrained' ? 'con' : 'npu'}`}>{r.cpu ? 'CPU fallback' : r.support === 'constrained' ? 'NPU, constrained' : 'NPU'}</span></td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <div className="cta-band">
        <div className="cta-inner">
          <div><h2>Bring the model you are about to ship.</h2><p>ONNX from PyTorch, TFLite from LiteRT, GGUF from llama.cpp. Under a minute to the first report.</p></div>
          <div className="row gap wrap"><Link href="/app" className="btn primary lg">Inspect a model</Link><Link href="/docs" className="btn lg on-dark">Export guides</Link></div>
        </div>
      </div>
    </>
  )
}

const ico = (d: string) => <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d={d} /></svg>
const FEATURES = [
  { t: 'Reads the file, not a guess', b: 'Hand-written ONNX protobuf, TFLite FlatBuffer and GGUF header readers. External data and multi-GB files are handled without loading the weights.', i: ico('M12 3v12m0 0-4-4m4 4 4-4M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2') },
  { t: 'Op inventory with support', b: 'Every op type with counts, MACs and a curated support status (NPU, NPU with constraints, CPU fallback), with the source and date of the table.', i: ico('M4 6h16M4 12h16M4 18h10') },
  { t: 'Shapes and memory', b: 'Shape inference for the common ops, dynamic dims flagged, parameter count by precision, and peak activation memory from tensor liveness.', i: ico('M4 4h16v16H4zM4 10h16M10 4v16') },
  { t: 'Partitions and switches', b: 'Where the graph breaks between NPU and CPU, how many hand-offs that costs, and which single op causes each break.', i: ico('M5 12h4m6 0h4M9 8l3 4-3 4m6-8-3 4 3 4') },
  { t: 'Fixes you can apply', b: 'A shape-pinned ONNX file generated in the browser, quantization and conversion commands, op rewrites, and a printable readiness report.', i: ico('M14.7 6.3a4 4 0 0 0-5.4 5.4L3 18l3 3 6.3-6.3a4 4 0 0 0 5.4-5.4l-2.5 2.5-2.4-.6-.6-2.4z') },
  { t: 'Versions over time', b: 'Each export becomes a version in its project. Compare two to see which ops, partitions and latency changed.', i: ico('M12 8v4l3 2M3 12a9 9 0 1 0 3-6.7L3 8m0-5v5h5') },
]
