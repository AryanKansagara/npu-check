import { useEffect } from 'react'
import { FIX_LABELS, fmtBytes } from './engine/analyze.ts'
import { SOURCES, TABLE_DATE } from './engine/support.ts'
import { useAnalysis, useProject } from './projectData.ts'
import { shapeStr } from './Tabs.tsx'
import { Empty, fmtDate, fmtMs, fmtN, fmtSi, Link } from './ui.tsx'

export default function Report({ id }: { id: string }) {
  const { loading, project, version, ir } = useProject(id)
  const { a } = useAnalysis(ir, project)
  useEffect(() => { if (project && version) document.title = `Readiness report: ${version.fileName} · NPU Check` }, [project, version])
  if (loading) return <div className="boot"><span className="spinner" /> Building the report</div>
  if (!project || !version || !ir || !a) return <section className="page narrow"><Empty title="Nothing to report"><p className="muted">Import a model first, or open this on the device that has its graph.</p><Link href={`/app/p/${id}`} className="btn primary">Back</Link></Empty></section>
  const d = project.settings.device
  return (
    <article className="report">
      <div className="report-actions"><Link href={`/app/p/${id}`} className="btn ghost sm">Back to project</Link><button type="button" className="btn primary sm" onClick={() => print()}>Print or save PDF</button></div>
      <header>
        <div className="eyebrow">NPU readiness report</div>
        <h1>{version.fileName}</h1>
        <p className="muted">{project.name} · {version.label} · {version.format.toUpperCase()} · {fmtBytes(version.fileSize)} · generated {fmtDate(Date.now())}{project.sampleId ? ' · synthetic sample model' : ''}</p>
        <p className="muted small">Device profile: {d.name} ({d.npuTops} TOPS int8, {d.memGBs} GB/s, utilisation {d.util}). {project.settings.applied.length ? `Simulated fixes: ${project.settings.applied.map(f => FIX_LABELS[f]).join(', ')}.` : 'No fixes simulated.'}</p>
      </header>
      <div className="kpis">
        <div className="kpi"><small>Readiness</small><b>{a.score}/100</b></div>
        <div className="kpi"><small>Est. latency</small><b>{ir.llm ? `${a.latency.tokensPerSec!.toFixed(1)} tok/s` : fmtMs(a.latency.totalMs)}</b></div>
        <div className="kpi"><small>Partitions</small><b>{a.partitions}</b><p>{a.switches} switches</p></div>
        <div className="kpi"><small>Parameters</small><b>{fmtSi(a.params)}</b><p>{fmtBytes(a.weightBytes)}</p></div>
        <div className="kpi"><small>MACs</small><b>{fmtSi(a.macs)}</b></div>
        <div className="kpi"><small>Peak activations</small><b>{fmtBytes(a.peakBytes)}</b></div>
      </div>
      <h2>Findings</h2>
      {a.findings.length ? <ol>{a.findings.map(f => <li key={f.id}><b>{f.title}.</b> {f.detail}</li>)}</ol> : <p>No blockers found.</p>}
      {a.inputs.length > 0 && <>
        <h2>Inputs and outputs</h2>
        <table><thead><tr><th>Name</th><th>Kind</th><th>Type</th><th>Shape</th></tr></thead>
          <tbody>{[...a.inputs.map(x => ({ ...x, k: 'input' })), ...a.outputs.map(x => ({ ...x, k: 'output' }))].map(x => <tr key={x.k + x.name}><td>{x.name}</td><td>{x.k}</td><td>{x.dtype}</td><td>{shapeStr(x.shape)}</td></tr>)}</tbody></table>
      </>}
      {a.opRows.length > 0 && <>
        <h2>Op inventory</h2>
        <table><thead><tr><th>Op</th><th className="num">Count</th><th>Support</th><th className="num">On CPU</th><th className="num">MACs</th></tr></thead>
          <tbody>{a.opRows.map(r => <tr key={r.domain + r.op}><td>{r.op}</td><td className="num">{fmtN(r.count)}</td><td>{r.support}</td><td className="num">{r.cpu}</td><td className="num">{r.macs ? fmtSi(r.macs) : ''}</td></tr>)}</tbody></table>
      </>}
      <h2>Weights by precision</h2>
      <table><thead><tr><th>Precision</th><th className="num">Tensors</th><th className="num">Params</th><th className="num">Bytes</th></tr></thead>
        <tbody>{(['fp32', 'fp16', 'int8', 'int4', 'other'] as const).filter(k => a.dtypes[k].tensors).map(k => <tr key={k}><td>{k}</td><td className="num">{a.dtypes[k].tensors}</td><td className="num">{fmtSi(a.dtypes[k].params)}</td><td className="num">{fmtBytes(a.dtypes[k].bytes)}</td></tr>)}</tbody></table>
      <h2>Method and assumptions</h2>
      <p className="small">Latency is a roofline estimate per node: the larger of compute time (2 x MACs over TOPS x utilisation, fp16 at half the int8 rate, CPU nodes at the CPU GFLOPS) and memory time (bytes read and written over DRAM bandwidth), plus a fixed cost and copy per CPU/NPU switch. Op support comes from a curated table reviewed {TABLE_DATE} from: {SOURCES.map(s => s.name).join('; ')}. It is not an official compatibility list; confirm on device with CPU fallback disabled.</p>
      <p className="muted small">NPU Check by Aryan Kansagara. Independent project, not affiliated with Qualcomm.</p>
    </article>
  )
}
