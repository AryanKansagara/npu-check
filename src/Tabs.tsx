import { useEffect, useState } from 'react'
import { analyze, compare, FIX_LABELS, fmtBytes, nodeReportCsv, opReportCsv, summarize, type Analysis, type DeviceProfile, type FixId } from './engine/analyze.ts'
import { pinDims } from './engine/onnx.ts'
import { rule, TABLE_DATE } from './engine/support.ts'
import type { Dim, ModelIR } from './engine/types.ts'
import { GraphView, Legend, MAX_DRAW, nodeClass } from './GraphView.tsx'
import type { Tab, TabProps } from './Project.tsx'
import { loadIr, sampleBytes } from './projectData.ts'
import { deleteVersion, getDefaults, getFile, type Project } from './store.ts'
import { deleteVersionCloud } from './sync.ts'
import { ask, Code, download, Empty, fmtDate, fmtMs, fmtN, fmtSi, toast } from './ui.tsx'

export default function Tabs(p: TabProps & { tab: Tab }) {
  if (p.tab === 'ops') return <OpsTab {...p} />
  if (p.tab === 'graph') return <GraphTab {...p} />
  if (p.tab === 'fixes') return <FixesTab {...p} />
  if (p.tab === 'history') return <HistoryTab {...p} />
  return <Overview {...p} />
}

export const shapeStr = (s: Dim[] | null) => s ? `[${s.join(', ')}]` : 'unknown'
const SEV: Record<string, string> = { blocker: 'Blocker', warn: 'Warning', info: 'Note' }

// ---------- Overview ----------

function Overview({ project, version, ir, a, base, update, setApplied }: TabProps) {
  const sim = project.settings.applied.length > 0
  const delta = (now: number, was: number, f: (n: number) => string) => sim && now !== was ? <span className={`kpi-was ${now < was ? 'down' : 'up'}`}>was {f(was)}</span> : null
  const llm = ir.llm
  return (
    <div className="stack">
      <div className="kpis">
        <div className="kpi"><small>Readiness</small><b>{a.score}<span className="unit">/100</span></b>{delta(a.score, base.score, String)}<p>{a.score >= 85 ? 'Ready to convert' : a.score >= 60 ? 'Converts with fallbacks' : 'Needs work before converting'}</p></div>
        <div className="kpi"><small>{llm ? 'Decode speed' : 'Est. latency'}</small><b>{llm ? `${a.latency.tokensPerSec!.toFixed(1)}` : fmtMs(a.latency.totalMs)}{llm && <span className="unit"> tok/s</span>}</b>{!llm && delta(a.latency.totalMs, base.latency.totalMs, fmtMs)}<p>{project.settings.device.name}</p></div>
        {!llm && <div className="kpi"><small>Partitions</small><b>{a.partitions}</b>{delta(a.partitions, base.partitions, String)}<p>{a.switches} CPU/NPU switch{a.switches === 1 ? '' : 'es'}</p></div>}
        {!llm && <div className="kpi"><small>On the NPU</small><b>{fmtN(a.npuNodes)}<span className="unit">/{fmtN(a.npuNodes + a.cpuNodes)}</span></b><p>{a.cpuNodes} node{a.cpuNodes === 1 ? '' : 's'} on CPU</p></div>}
        <div className="kpi"><small>Parameters</small><b>{fmtSi(a.params)}</b><p>{fmtBytes(a.weightBytes)} of weights</p></div>
        {llm ? <div className="kpi"><small>KV cache</small><b>{a.latency.kvCacheBytes ? fmtBytes(a.latency.kvCacheBytes) : '—'}</b><p>fp16, 4096 tokens</p></div>
          : <div className="kpi"><small>Peak activations</small><b>{fmtBytes(a.peakBytes)}</b><p>{fmtSi(a.macs)} MACs</p></div>}
      </div>

      <div className="grid-2 top">
        <div className="panel">
          <div className="panel-head"><h2>Findings</h2><span className="muted small">{a.findings.length} item{a.findings.length === 1 ? '' : 's'}</span></div>
          {a.findings.length === 0 ? <Empty title="Nothing blocks the NPU"><p className="muted">Every node is placed on the NPU at {a.quant.precision}. Run it with CPU fallback disabled to confirm (see Fixes).</p></Empty> : (
            <ul className="findings">
              {a.findings.map(f => (
                <li key={f.id} className={`finding ${f.severity}`}>
                  <div className="row between top gap-sm">
                    <div><span className={`sev ${f.severity}`}>{SEV[f.severity]}</span><h3>{f.title}</h3></div>
                    {f.fix && <button type="button" className="btn outline sm" onClick={() => setApplied(f.fix!, true)}>Simulate fix</button>}
                  </div>
                  <p>{f.detail}</p>
                  {f.nodes && f.nodes.length > 0 && <p className="muted small mono break">{f.nodes.slice(0, 6).join(', ')}{f.nodes.length > 6 ? ` and ${f.nodes.length - 6} more` : ''}</p>}
                </li>
              ))}
            </ul>
          )}
          {sim && <button type="button" className="linkish small" onClick={() => update({ settings: { ...project.settings, applied: [] } })}>Clear simulated fixes</button>}
        </div>
        <div className="stack">
          <div className="panel">
            <div className="panel-head"><h2>Model</h2><span className="muted small mono">{version.format.toUpperCase()}</span></div>
            <dl className="facts">
              <div><dt>File</dt><dd className="mono break">{version.fileName} · {fmtBytes(version.fileSize)}</dd></div>
              <div><dt>Producer</dt><dd>{ir.producer || '—'}</dd></div>
              {a.opset !== null && <div><dt>Opset</dt><dd>{a.opset}{ir.opset.length > 1 ? ` (+ ${ir.opset.filter(o => o.domain).map(o => `${o.domain} ${o.version}`).join(', ')})` : ''}</dd></div>}
              {llm ? <>
                <div><dt>Architecture</dt><dd>{llm.arch}{llm.name ? ` · ${llm.name}` : ''}</dd></div>
                <div><dt>Layers / width</dt><dd>{llm.blockCount ?? '—'} blocks · {llm.embeddingLength ?? '—'} wide · {llm.headCount ?? '—'} heads ({llm.headCountKv ?? llm.headCount ?? '—'} KV)</dd></div>
                <div><dt>Context</dt><dd>{llm.contextLength ? fmtN(llm.contextLength) : '—'} tokens</dd></div>
                <div><dt>File type</dt><dd>{llm.fileType ?? '—'}</dd></div>
                <div><dt>Prefill (512 tok)</dt><dd>{fmtMs(a.latency.prefillMs!)} if compute-bound on the NPU</dd></div>
              </> : <>
                <div><dt>Nodes</dt><dd>{fmtN(a.nodeCount)} · {a.opRows.length} op types</dd></div>
                <div><dt>Quantization</dt><dd>{QV[a.quant.verdict]}{a.quant.qdq ? ` · ${a.quant.qdq} Q/DQ nodes` : ''}</dd></div>
                <div><dt>Compute</dt><dd>{fmtSi(a.macs)} MACs · {fmtSi(a.flops)} FLOPs</dd></div>
                <div><dt>Peak activations</dt><dd>{fmtBytes(a.peakBytes)} at <span className="mono">{a.peakAt}</span></dd></div>
              </>}
            </dl>
          </div>
          <PrecisionPanel a={a} />
          {!llm && <LatencyPanel a={a} />}
        </div>
      </div>

      {!llm && <IoPanel a={a} />}
      <DevicePanel project={project} update={update} />
      {ir.issues.length > 0 && (
        <div className="panel">
          <div className="panel-head"><h2>Validation</h2><span className="muted small">{ir.issues.length} issue{ir.issues.length === 1 ? '' : 's'}</span></div>
          <div className="issues">{ir.issues.map((i, k) => <div key={k} className={i.level === 'error' ? 'error-row' : ''}><span className="ln">{i.where}</span><span>{i.message}</span></div>)}</div>
        </div>
      )}
    </div>
  )
}

const QV: Record<Analysis['quant']['verdict'], string> = { 'quantized-per-channel': 'QDQ, per-channel', 'quantized-per-tensor': 'QDQ, per-tensor', 'weight-only': 'Weight-only integers', float: 'Float (not quantized)' }

function PrecisionPanel({ a }: { a: Analysis }) {
  const rows = (['fp32', 'fp16', 'int8', 'int4', 'other'] as const).filter(k => a.dtypes[k].tensors)
  const total = Math.max(1, a.weightBytes)
  return (
    <div className="panel">
      <div className="panel-head"><h2>Weights by precision</h2><span className="muted small">{fmtBytes(a.weightBytes)}</span></div>
      <div className="bbar big" role="img" aria-label="Weight bytes by precision">{rows.map(k => <span key={k} className={`p-${k}`} style={{ width: `${a.dtypes[k].bytes / total * 100}%` }} />)}</div>
      <table className="compact-table">
        <thead><tr><th>Precision</th><th className="num">Tensors</th><th className="num">Params</th><th className="num">Bytes</th></tr></thead>
        <tbody>{rows.map(k => <tr key={k}><td><i className={`dot p-${k}`} />{k}</td><td className="num">{fmtN(a.dtypes[k].tensors)}</td><td className="num">{fmtSi(a.dtypes[k].params)}</td><td className="num">{fmtBytes(a.dtypes[k].bytes)}</td></tr>)}</tbody>
      </table>
      {a.externalBytes > 0 && <p className="muted small">{fmtBytes(a.externalBytes)} live in external data files; sizes come from tensor dims.</p>}
    </div>
  )
}

function LatencyPanel({ a }: { a: Analysis }) {
  const L = a.latency, t = Math.max(1e-9, L.totalMs)
  const top = [...a.nodes].sort((x, y) => y.ms - x.ms).slice(0, 5)
  return (
    <div className="panel">
      <div className="panel-head"><h2>Where the time goes</h2><span className="muted small">{Math.round(L.computeShare * 100)}% compute-bound</span></div>
      <div className="bbar big" role="img" aria-label="Latency split">
        <span className="l-npu" style={{ width: `${L.npuMs / t * 100}%` }} /><span className="l-cpu" style={{ width: `${L.cpuMs / t * 100}%` }} /><span className="l-sw" style={{ width: `${L.switchMs / t * 100}%` }} />
      </div>
      <div className="legend"><span><i className="l-npu" />NPU {fmtMs(L.npuMs)}</span><span><i className="l-cpu" />CPU {fmtMs(L.cpuMs)}</span><span><i className="l-sw" />Switches {fmtMs(L.switchMs)}</span></div>
      <table className="compact-table">
        <thead><tr><th>Slowest nodes</th><th>Op</th><th className="num">Est.</th></tr></thead>
        <tbody>{top.map(n => <tr key={n.index}><td className="mono break">{n.name}</td><td><span className={`pill ${nodeClass(a, n.index)}`}>{n.op}</span></td><td className="num">{fmtMs(n.ms)}</td></tr>)}</tbody>
      </table>
    </div>
  )
}

function IoPanel({ a }: { a: Analysis }) {
  return (
    <div className="panel">
      <div className="panel-head"><h2>Inputs and outputs</h2>{a.symbols.length > 0 && <span className="badge warn">Dynamic: {a.symbols.join(', ')}</span>}</div>
      <div className="table-wrap">
        <table>
          <thead><tr><th>Name</th><th>Kind</th><th>Type</th><th>Shape</th></tr></thead>
          <tbody>
            {[...a.inputs.map(x => ({ ...x, kind: 'input' })), ...a.outputs.map(x => ({ ...x, kind: 'output' }))].map(x => (
              <tr key={x.kind + x.name}><td className="mono">{x.name}</td><td>{x.kind}</td><td className="mono">{x.dtype}</td>
                <td className="mono">[{(x.shape ?? []).map((d, i) => <span key={i}>{i > 0 && ', '}<span className={typeof d === 'string' ? 'dyn' : ''}>{d}</span></span>)}]</td></tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function DevicePanel({ project, update }: { project: Project; update(p: Partial<Project>): void }) {
  const d = project.settings.device
  const presets = getDefaults().devices
  const set = (patch: Partial<DeviceProfile>) => update({ settings: { ...project.settings, device: { ...d, ...patch } } })
  const num = (k: keyof DeviceProfile, label: string, step = 'any', hint = '') => (
    <label className="field"><span>{label}</span><input type="number" min={0} step={step} value={d[k] as number} onChange={e => { const v = Number(e.target.value); if (Number.isFinite(v) && v >= 0) set({ [k]: v }) }} />{hint && <small className="muted">{hint}</small>}</label>
  )
  return (
    <details className="panel">
      <summary><h2 className="inline">Device profile</h2> <span className="muted small">{d.name} · {d.npuTops} TOPS · {d.memGBs} GB/s</span></summary>
      <div className="stack">
        <label className="field"><span>Preset</span>
          <select value={presets.some(x => x.id === d.id) ? d.id : ''} onChange={e => { const x = presets.find(y => y.id === e.target.value); if (x) update({ settings: { ...project.settings, device: { ...x } } }) }}>
            {!presets.some(x => x.id === d.id) && <option value="">Custom</option>}
            {presets.map(x => <option key={x.id} value={x.id}>{x.name}</option>)}
          </select>
        </label>
        <div className="inline-fields">
          {num('npuTops', 'NPU int8 TOPS')}{num('memGBs', 'Memory GB/s')}{num('cpuGflops', 'CPU GFLOPS')}{num('vtcmMB', 'NPU scratch MB')}{num('switchMs', 'Switch cost ms')}{num('util', 'Utilisation 0 to 1', '0.05')}
        </div>
        <p className="muted small">{d.note}</p>
      </div>
    </details>
  )
}

// ---------- Ops ----------

function OpsTab({ version, a }: TabProps) {
  const [q, setQ] = useState('')
  const [only, setOnly] = useState<'all' | 'cpu'>('all')
  const rows = a.opRows.filter(r => (only === 'all' || r.cpu > 0) && (!q || r.op.toLowerCase().includes(q.toLowerCase())))
  const base = version.fileName.replace(/\.[^.]+$/, '')
  return (
    <div className="panel">
      <div className="toolbar">
        <label className="sr-only" htmlFor="op-q">Filter ops</label>
        <input id="op-q" type="search" placeholder="Filter ops" value={q} onChange={e => setQ(e.target.value)} />
        <div className="seg-control" role="group" aria-label="Show">
          <button type="button" className={only === 'all' ? 'on' : ''} onClick={() => setOnly('all')}>All ops</button>
          <button type="button" className={only === 'cpu' ? 'on' : ''} onClick={() => setOnly('cpu')}>Falling back</button>
        </div>
        <span className="grow" />
        <button type="button" className="btn ghost sm" onClick={() => download(`${base}-ops.csv`, opReportCsv(a))}>Op report CSV</button>
        <button type="button" className="btn ghost sm" onClick={() => download(`${base}-nodes.csv`, nodeReportCsv(a))}>Node CSV</button>
      </div>
      {a.nodeCount === 0 ? <Empty title="No graph in this file"><p className="muted">GGUF files carry tensors and metadata, not an op graph. See Overview for memory and decode estimates.</p></Empty> : (
        <div className="table-wrap">
          <table>
            <thead><tr><th>Op</th><th className="num">Count</th><th>Support</th><th className="num">On NPU</th><th className="num">On CPU</th><th className="num">MACs</th><th>Notes</th></tr></thead>
            <tbody>
              {rows.map(r => (
                <tr key={r.domain + r.op}>
                  <td className="mono">{r.op}{r.domain && <small className="muted"> {r.domain}</small>}</td>
                  <td className="num">{fmtN(r.count)}</td>
                  <td><span className={`pill ${r.support}`}>{r.support === 'npu' ? 'Supported' : r.support === 'constrained' ? 'With constraints' : 'CPU fallback'}</span></td>
                  <td className="num">{fmtN(r.npu)}</td><td className={`num${r.cpu ? ' bad-text' : ''}`}>{fmtN(r.cpu)}</td>
                  <td className="num">{r.macs ? fmtSi(r.macs) : '—'}</td>
                  <td className="small">{r.note}{r.replace && <><br /><b>Fix:</b> {r.replace}</>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="muted small">Support table reviewed {TABLE_DATE}. It is a curated first pass, not an official list; see Docs for sources and how to confirm with the converter.</p>
    </div>
  )
}

// ---------- Graph ----------

function GraphTab({ ir, a }: TabProps) {
  const [q, setQ] = useState('')
  const [sel, setSel] = useState<number | null>(null)
  const [zoom, setZoom] = useState(ir.nodes.length > 400 ? 0.6 : 1)
  if (!ir.nodes.length) return <div className="panel"><Empty title="No graph in this file"><p className="muted">GGUF files carry tensors and metadata, not an op graph.</p></Empty></div>
  const n = sel !== null ? ir.nodes[sel] : null, r = sel !== null ? a.nodes[sel] : null
  const matches = q ? a.nodes.filter(x => x.name.toLowerCase().includes(q.toLowerCase()) || x.op.toLowerCase().includes(q.toLowerCase())) : []
  return (
    <div className="graph-layout">
      <div className="panel graph-panel">
        <div className="toolbar">
          <label className="sr-only" htmlFor="graph-search">Search nodes</label>
          <input id="graph-search" type="search" placeholder="Search nodes, ops or tensors ( / )" value={q} onChange={e => setQ(e.target.value)} />
          {q && <span className="muted small">{matches.length} match{matches.length === 1 ? '' : 'es'}</span>}
          <span className="grow" />
          <div className="seg-control" role="group" aria-label="Zoom">
            <button type="button" onClick={() => setZoom(z => Math.max(0.25, z / 1.25))} aria-label="Zoom out">−</button>
            <button type="button" onClick={() => setZoom(1)}>{Math.round(zoom * 100)}%</button>
            <button type="button" onClick={() => setZoom(z => Math.min(2.5, z * 1.25))} aria-label="Zoom in">+</button>
          </div>
        </div>
        <Legend />
        <div className="graph-scroll"><GraphView ir={ir} a={a} query={q} selected={sel} onSelect={setSel} zoom={zoom} /></div>
        {ir.nodes.length > MAX_DRAW && matches.length > 0 && (
          <ul className="match-list">{matches.slice(0, 200).map(m => <li key={m.index}><button type="button" className="linkish" onClick={() => setSel(m.index)}>{m.op} · {m.name}</button></li>)}</ul>
        )}
      </div>
      <aside className="panel node-panel" aria-live="polite">
        {!n || !r ? <Empty title="Select a node"><p className="muted small">Click a node to see its shapes, placement and cost. Red edges mark CPU/NPU switches.</p></Empty> : (
          <div className="stack-sm">
            <span className={`pill ${nodeClass(a, r.index)}`}>{r.status === 'npu' ? 'NPU' : r.status === 'cpu' ? 'CPU fallback' : 'Post-processing'}</span>
            <h3 className="mono break">{n.name}</h3>
            <p><b>{n.op}</b>{n.domain && <span className="muted"> · {n.domain}</span>}</p>
            <p className="small">{r.reason}</p>
            {rule(n.op, n.domain).replace && <p className="small"><b>Replacement:</b> {rule(n.op, n.domain).replace}</p>}
            <dl className="facts">
              <div><dt>MACs</dt><dd>{r.macs ? fmtSi(r.macs) : '—'}</dd></div>
              <div><dt>Est. time</dt><dd>{fmtMs(r.ms)}</dd></div>
            </dl>
            <h4>Inputs</h4>
            <ul className="io-list">{n.inputs.filter(Boolean).map(x => <li key={x}><span className="mono break">{x}</span> <span className="muted mono">{ir.tensors[x]?.init ? 'weight ' : ''}{shapeStr(ir.tensors[x]?.shape ?? null)}</span></li>)}</ul>
            <h4>Outputs</h4>
            <ul className="io-list">{n.outputs.filter(Boolean).map(x => <li key={x}><span className="mono break">{x}</span> <span className="muted mono">{shapeStr(x === n.outputs[0] ? r.outShape : ir.tensors[x]?.shape ?? null)}</span></li>)}</ul>
            {Object.keys(n.attrs).length > 0 && <><h4>Attributes</h4><ul className="io-list">{Object.entries(n.attrs).map(([k, v]) => <li key={k}><span className="mono">{k}</span> <span className="muted mono">{Array.isArray(v) ? `[${v.slice(0, 12).join(', ')}${v.length > 12 ? ', …' : ''}]` : String(v)}</span></li>)}</ul></>}
          </div>
        )}
      </aside>
    </div>
  )
}

// ---------- Fixes ----------

const FIX_TEXT: Record<FixId, string> = {
  'pin-dims': 'Compile one static graph per shape you need. The pinned file below is generated in your browser: inputs, outputs and value_info get fixed dims; weights and nodes are copied byte for byte.',
  'replace-ops': 'Rewrite the ops that have no NPU kernel into ones that do, ideally in the PyTorch module before export so the graph stays clean.',
  'split-post': 'Cut the graph before data-dependent post-processing (NMS, NonZero) and run that part in the app on the CPU, once per frame, instead of bouncing in and out of the NPU.',
  'quantize-int8': 'Static QDQ quantization with real calibration samples. int8 runs at the NPU’s full rate and quarters weight and activation traffic. Keep sensitive ops at 16-bit if accuracy drops.',
}

function FixesTab({ project, version, ir, a, base, update, setApplied }: TabProps) {
  const fixes = [...new Set([...base.available, ...project.settings.applied])] as FixId[]
  const what = (f: FixId) => {
    const on = analyze(ir, { ...project.settings, applied: [...new Set([...project.settings.applied, f])] })
    const off = analyze(ir, { ...project.settings, applied: project.settings.applied.filter(x => x !== f) })
    return { on, off }
  }
  return (
    <div className="stack">
      {fixes.length === 0 && <div className="panel"><Empty title="No fixes needed"><p className="muted">Every node lands on the NPU and the model is already quantized. Verify on device with CPU fallback disabled (below).</p></Empty></div>}
      <div className="fix-list">
        {fixes.map((f, i) => {
          const { on, off } = what(f)
          const active = project.settings.applied.includes(f)
          return (
            <article key={f} className={`fix-card${active ? ' on' : ''}`}>
              <div className="fix-rank">{i + 1}</div>
              <div className="grow stack-sm">
                <div className="row between top gap wrap">
                  <h3>{FIX_LABELS[f]}</h3>
                  <label className="row gap-sm small"><span className="switch"><input type="checkbox" checked={active} onChange={e => setApplied(f, e.target.checked)} /><span /></span>Simulate</label>
                </div>
                <p>{FIX_TEXT[f]}</p>
                <div className="fix-impact">
                  <div><small>Partitions</small><b>{off.partitions} → {on.partitions}</b></div>
                  <div><small>CPU nodes</small><b>{off.cpuNodes} → {on.cpuNodes}</b></div>
                  <div><small>Est. latency</small><b>{fmtMs(off.latency.totalMs)} → {fmtMs(on.latency.totalMs)}</b></div>
                  <div><small>Readiness</small><b>{off.score} → {on.score}</b></div>
                </div>
                <FixArtifact f={f} project={project} version={version} ir={ir} a={a} update={update} />
              </div>
            </article>
          )
        })}
      </div>
      <div className="panel">
        <h2>Verify on the device</h2>
        <p className="muted">Estimates are a first pass. Run the converted model with CPU fallback disabled: the session fails to create if any node cannot be placed on the NPU, which is the ground truth for your QAIRT and ONNX Runtime versions.</p>
        <Code label="Python · ONNX Runtime QNN EP" code={verifySnippet(version.fileName, a)} />
      </div>
    </div>
  )
}

function FixArtifact({ f, project, version, ir, a, update }: { f: FixId; project: Project; version: TabProps['version']; ir: ModelIR; a: Analysis; update(p: Partial<Project>): void }) {
  const file = version.fileName
  const out = file.replace(/\.onnx$/i, '')
  if (f === 'pin-dims') {
    const pins = Object.fromEntries(a.symbols.map(s => [s, project.settings.pins[s] ?? 1]))
    const cmd = Object.entries(pins).filter(([s]) => s !== '?').map(([s, v]) => `python -m onnxruntime.tools.make_dynamic_shape_fixed --dim_param ${s} --dim_value ${v} ${file} ${out}_static.onnx`).join('\n')
    return (
      <div className="stack-sm">
        <div className="inline-fields">
          {a.symbols.filter(s => s !== '?').map(s => (
            <label key={s} className="field"><span>{s}</span><input type="number" min={1} value={pins[s]} onChange={e => { const v = Math.max(1, Math.round(Number(e.target.value) || 1)); update({ settings: { ...project.settings, pins: { ...project.settings.pins, [s]: v } } }) }} /></label>
          ))}
        </div>
        {ir.format === 'onnx' && <PinnedDownload project={project} version={version} pins={pins} />}
        {ir.format === 'onnx' && <Code label="Shell · same result with ONNX Runtime tools" code={cmd} />}
        {ir.format === 'tflite' && <Code label="Python · re-export with a fixed signature" code={`# Re-convert with concrete input shapes so the interpreter sees no -1 dims\nconverter = tf.lite.TFLiteConverter.from_concrete_functions([\n    model.__call__.get_concrete_function(tf.TensorSpec(${JSON.stringify((a.inputs[0]?.shape ?? []).map(d => typeof d === 'number' ? d : pins[d] ?? 1))}, tf.float32))\n])\nopen("${out}_static.tflite", "wb").write(converter.convert())`} />}
      </div>
    )
  }
  if (f === 'replace-ops') {
    const rows = a.opRows.filter(r => r.replace && r.cpu > 0)
    return <ul className="io-list">{(rows.length ? rows : a.opRows.filter(r => r.replace)).map(r => <li key={r.op}><b className="mono">{r.op}</b> ({r.count}): {r.replace}</li>)}</ul>
  }
  if (f === 'split-post') {
    const removed = new Set(analyze(ir, { ...project.settings, applied: ['split-post'] }).nodes.filter(n => n.status === 'removed').map(n => n.index))
    const producedOutside = new Set(ir.nodes.flatMap((n, i) => removed.has(i) ? [] : n.outputs))
    const cut = [...new Set([...removed].flatMap(i => ir.nodes[i].inputs).filter(x => producedOutside.has(x)))]
    const keepOut = ir.outputs.filter(o => producedOutside.has(o))
    return <Code label="Python · cut the graph before post-processing" code={`import onnx.utils\n\n# Everything up to the tensors that feed NMS stays on the NPU; NMS runs in your app.\nonnx.utils.extract_model(\n    "${file}", "${out}_npu.onnx",\n    input_names=${JSON.stringify(a.inputs.map(i => i.name))},\n    output_names=${JSON.stringify([...new Set([...cut, ...keepOut])])},\n)`} />
  }
  const ins = a.inputs.map(i => ({ name: i.name, shape: (i.shape ?? []).map(d => typeof d === 'number' ? d : project.settings.pins[d] ?? 1) }))
  return <Code label="Python · ONNX Runtime static QDQ for the QNN EP" code={quantSnippet(file, out, ins)} />
}

function PinnedDownload({ project, version, pins }: { project: Project; version: TabProps['version']; pins: Record<string, number> }) {
  const [busy, setBusy] = useState(false)
  const get = async () => {
    setBusy(true)
    try {
      const bytes = project.sampleId ? sampleBytes(version.fileName) : await getFile(version.id).then(b => b ? b.arrayBuffer().then(x => new Uint8Array(x)) : null)
      if (!bytes) { toast('The original file is not stored on this device. Re-import it, or use the command below.', { tone: 'error', ttl: 7000 }); return }
      const fixed = pinDims(bytes, pins)
      download(version.fileName.replace(/\.onnx$/i, '') + '_static.onnx', new Blob([fixed.slice().buffer], { type: 'application/octet-stream' }))
      toast('Downloaded the pinned model', { tone: 'ok' })
    } finally { setBusy(false) }
  }
  return <div className="row gap wrap"><button type="button" className="btn primary sm" disabled={busy} onClick={() => void get()}>{busy ? 'Writing' : 'Download pinned .onnx'}</button><span className="muted small">{Object.entries(pins).map(([k, v]) => `${k}=${v}`).join(', ')}</span></div>
}

function quantSnippet(file: string, out: string, ins: { name: string; shape: number[] }[]) {
  return `import numpy as np
from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize
from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config, qnn_preprocess_model

class Reader(CalibrationDataReader):
    """Feed 100 to 500 REAL samples, preprocessed exactly like production."""
    def __init__(self, samples):
        self.it = iter(samples)
    def get_next(self):
        return next(self.it, None)

def load_samples():
    # Replace with your data loader. Shapes must match the (pinned) model inputs.
    for _ in range(200):
        yield {${ins.map(i => `\n            "${i.name}": np.random.rand(${i.shape.join(', ')}).astype(np.float32),`).join('')}
        }

qnn_preprocess_model("${file}", "${out}_pre.onnx")
config = get_qnn_qdq_config(
    "${out}_pre.onnx", Reader(load_samples()),
    activation_type=QuantType.QUInt16,   # 16-bit activations: safer accuracy, still on the NPU
    weight_type=QuantType.QUInt8,        # per-channel 8-bit weights
    per_channel=True,
)
quantize("${out}_pre.onnx", "${out}_qdq.onnx", config)`
}

function verifySnippet(file: string, a: Analysis) {
  const name = /\.onnx$/i.test(file) ? file : 'model.onnx'
  return `import onnxruntime as ort

so = ort.SessionOptions()
so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")  # fail instead of silently using the CPU
sess = ort.InferenceSession(
    "${name.replace(/\.onnx$/i, a.quant.verdict === 'float' ? '_qdq.onnx' : '.onnx')}", sess_options=so,
    providers=[("QNNExecutionProvider", {"backend_path": "QnnHtp.dll", "htp_performance_mode": "burst"})],
)
print(sess.get_providers())`
}

// ---------- History ----------

function HistoryTab({ project, version, update }: TabProps) {
  const vs = project.versions
  const [ia, setIa] = useState(vs.length > 1 ? vs[vs.length - 2].id : vs[0].id)
  const [ib, setIb] = useState(vs[vs.length - 1].id)
  const va = vs.find(v => v.id === ia) ?? vs[0], vb = vs.find(v => v.id === ib) ?? vs[vs.length - 1]
  const [live, setLive] = useState<{ a: ReturnType<typeof summarize>; b: ReturnType<typeof summarize> } | null>(null)
  // Recompute both with the current device so differences are not just a changed profile.
  useEffect(() => {
    let ok = true
    void Promise.all([loadIr(project, va), loadIr(project, vb)]).then(([x, y]) => {
      if (!ok) return
      const opts = { ...project.settings, applied: [] }
      setLive(x && y ? { a: summarize(analyze(x, opts)), b: summarize(analyze(y, opts)) } : null)
    })
    return () => { ok = false }
  }, [va.id, vb.id, project.settings.device])
  const d = compare(live?.a ?? va.summary, live?.b ?? vb.summary)
  const fmt = (k: string, n: number) => k === 'latencyMs' ? fmtMs(n) : /Bytes/.test(k) ? fmtBytes(n) : k === 'params' || k === 'macs' ? fmtSi(n) : fmtN(n)
  return (
    <div className="stack">
      <div className="panel">
        <div className="panel-head"><h2>Versions</h2><span className="muted small">Each import is a version. Model files stay on the device that imported them.</span></div>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Label</th><th>File</th><th>Imported</th><th className="num">Readiness</th><th className="num">Partitions</th><th className="num">Est. latency</th><th /></tr></thead>
            <tbody>
              {[...vs].reverse().map(v => (
                <tr key={v.id} className={v.id === version.id ? 'current' : ''}>
                  <td><input className="table-input" aria-label="Version label" defaultValue={v.label} onBlur={e => { const label = e.target.value.trim(); if (label && label !== v.label) update({ versions: vs.map(x => x.id === v.id ? { ...x, label } : x) }) }} /></td>
                  <td className="mono small break">{v.fileName}{v.irMissing && <span className="badge warn" title="Graph is on another device">summary only</span>}</td>
                  <td className="small">{fmtDate(v.createdAt)}</td>
                  <td className="num">{v.summary.score}</td><td className="num">{v.summary.partitions}</td><td className="num">{fmtMs(v.summary.latencyMs)}</td>
                  <td className="num nowrap">
                    {v.id !== version.id && <button type="button" className="btn ghost sm" onClick={() => update({ currentId: v.id })}>Open</button>}
                    {vs.length > 1 && <button type="button" className="btn danger-ghost sm" onClick={async () => {
                      if (!(await ask(`Delete version “${v.label}”?`, 'Removes this version and its stored graph from this device and the cloud. You can undo for a few seconds.', 'Delete version'))) return
                      const undo = await deleteVersion(project, v.id)
                      void deleteVersionCloud(v.id)
                      toast(`Deleted ${v.label}`, { undo: () => void undo() })
                    }}>Delete</button>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      {vs.length < 2 ? <div className="panel"><Empty title="Import another version to compare"><p className="muted">Re-export after a change (pinned shapes, quantized, ops rewritten) and import it into this project.</p></Empty></div> : (
        <div className="panel">
          <div className="panel-head wrap gap">
            <h2>Compare</h2>
            <div className="row gap-sm wrap">
              <label className="sr-only" htmlFor="cmp-a">From</label>
              <select id="cmp-a" value={ia} onChange={e => setIa(e.target.value)}>{vs.map(v => <option key={v.id} value={v.id}>{v.label}</option>)}</select>
              <span aria-hidden="true">→</span>
              <label className="sr-only" htmlFor="cmp-b">To</label>
              <select id="cmp-b" value={ib} onChange={e => setIb(e.target.value)}>{vs.map(v => <option key={v.id} value={v.id}>{v.label}</option>)}</select>
            </div>
          </div>
          <div className="grid-2 top">
            <table className="compact-table">
              <thead><tr><th>Metric</th><th className="num">{va.label}</th><th className="num">{vb.label}</th></tr></thead>
              <tbody>{d.metrics.map(m => <tr key={m.key}><td>{m.label}</td><td className="num">{fmt(m.key, m.a)}</td><td className={`num ${m.better === true ? 'diff-up' : m.better === false ? 'diff-down' : ''}`}>{fmt(m.key, m.b)}</td></tr>)}</tbody>
            </table>
            <div className="stack-sm">
              <h3>Op changes</h3>
              {d.ops.length ? <ul className="io-list">{d.ops.map(o => <li key={o.op}><span className="mono">{o.op}</span> {o.a} → <b>{o.b}</b></li>)}</ul> : <p className="muted small">Same op inventory.</p>}
              <h3>Resolved</h3>
              {d.fixed.length ? <ul className="io-list good-text">{d.fixed.map(t => <li key={t}>{t}</li>)}</ul> : <p className="muted small">None.</p>}
              <h3>New</h3>
              {d.introduced.length ? <ul className="io-list bad-text">{d.introduced.map(t => <li key={t}>{t}</li>)}</ul> : <p className="muted small">None.</p>}
            </div>
          </div>
          {!live && <p className="muted small">Using stored summaries (graphs not on this device).</p>}
        </div>
      )}
    </div>
  )
}

