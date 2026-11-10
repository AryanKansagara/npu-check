import { useRef, useState } from 'react'
import { analyze, fmtBytes, suggestPin, symbols } from './engine/analyze.ts'
import type { ModelIR } from './engine/types.ts'
import { addVersion, readModel } from './projectData.ts'
import { getDefaults, type Project } from './store.ts'
import { go, Link, toast } from './ui.tsx'

interface Parsed { file: File; ir: ModelIR; ms: number }

export default function ImportTab({ project, first }: { project: Project; first: boolean }) {
  const [over, setOver] = useState(false)
  const [busy, setBusy] = useState<string | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [parsed, setParsed] = useState<Parsed | null>(null)
  const [pins, setPins] = useState<Record<string, number>>({})
  const [label, setLabel] = useState('')
  const input = useRef<HTMLInputElement>(null)

  const take = async (file: File | undefined) => {
    if (!file) return
    setErr(null); setParsed(null); setBusy(`Reading ${file.name}`)
    // Let the spinner paint before a large synchronous parse.
    await new Promise(r => setTimeout(r, 30))
    const t0 = performance.now()
    try {
      const ir = await readModel(file)
      const syms = symbols(ir)
      setPins(Object.fromEntries(syms.map(s => [s, project.settings.pins[s] ?? (/seq|len|token/i.test(s) ? getDefaults().seqLen : suggestPin(s))])))
      setParsed({ file, ir, ms: performance.now() - t0 })
      setLabel(`v${project.versions.length + 1} ${file.name.replace(/\.[^.]+$/, '')}`)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally { setBusy(null) }
  }

  const save = async () => {
    if (!parsed) return
    setBusy('Saving version')
    try {
      const p = await addVersion(project, parsed.ir, parsed.file, parsed.file.name, label.trim() || parsed.file.name, pins)
      toast(`Imported ${parsed.file.name}`, { tone: 'ok' })
      go(`/app/p/${p.id}`)
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)) } finally { setBusy(null) }
  }

  const ir = parsed?.ir
  const errors = ir?.issues.filter(i => i.level === 'error').length ?? 0
  const preview = ir ? analyze(ir, { ...project.settings, pins, applied: [] }) : null

  return (
    <div className="stack">
      {first && <div className="banner">Step 1 of 2: import the model you want to deploy. Later exports of the same model go into this project as new versions.</div>}
      <div className="grid-2 top">
        <div
          className={`drop${over ? ' over' : ''}`}
          onDragOver={e => { e.preventDefault(); setOver(true) }} onDragLeave={() => setOver(false)}
          onDrop={e => { e.preventDefault(); setOver(false); void take(e.dataTransfer.files[0]) }}
        >
          <svg viewBox="0 0 24 24" width="34" height="34" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true"><path d="M12 3v12m0 0-4-4m4 4 4-4M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" /></svg>
          <h3>Drop a model file</h3>
          <p className="muted small"><code>.onnx</code> · <code>.tflite</code> · <code>.gguf</code>. Parsed in this tab; nothing is uploaded.</p>
          <input ref={input} type="file" accept=".onnx,.tflite,.gguf,.ort" className="sr-only" id="file" onChange={e => void take(e.target.files?.[0])} />
          <label htmlFor="file" className="btn primary">Choose file</label>
          {busy && <p className="row gap-sm center-x" role="status"><span className="spinner" /> {busy}</p>}
          {err && <p className="error" role="alert">{err}</p>}
        </div>
        <div className="panel">
          <h2>Getting the file</h2>
          <ul className="tight">
            <li><b>PyTorch:</b> <code>torch.onnx.export(model, example, "model.onnx", opset_version=17)</code>. Drop only the <code>.onnx</code>; an <code>.onnx.data</code> weights file is not needed.</li>
            <li><b>TFLite / LiteRT:</b> the <code>.tflite</code> from your converter or the app bundle.</li>
            <li><b>LLMs:</b> any <code>.gguf</code>; only the header is read, so multi-GB files are fine.</li>
          </ul>
          <Link href="/docs" className="btn ghost sm">Full export guides</Link>
        </div>
      </div>

      {ir && preview && parsed && (
        <div className="panel">
          <div className="panel-head"><h2>Step 2 of 2: check and map</h2><span className="muted small">Read in {Math.round(parsed.ms)} ms</span></div>
          <dl className="facts cols">
            <div><dt>Detected format</dt><dd>{ir.format.toUpperCase()}{ir.producer ? ` · ${ir.producer}` : ''}</dd></div>
            <div><dt>File</dt><dd className="mono break">{parsed.file.name} · {fmtBytes(parsed.file.size)}</dd></div>
            <div><dt>Graph</dt><dd>{ir.nodes.length ? `${ir.nodes.length.toLocaleString()} nodes, ${preview.opRows.length} op types` : `${Object.keys(ir.tensors).length.toLocaleString()} tensors (no op graph)`}</dd></div>
            <div><dt>Weights</dt><dd>{fmtBytes(preview.weightBytes)}{preview.externalBytes ? ` (${fmtBytes(preview.externalBytes)} external)` : ''}</dd></div>
          </dl>

          <h3>Validation</h3>
          {ir.issues.length === 0 ? <p className="good-text">No structural problems: every node input has a producer, initializer or graph input.</p> : (
            <div className="issues">{ir.issues.map((i, k) => <div key={k} className={i.level === 'error' ? 'error-row' : ''}><span className="ln">{i.where}</span><span>{i.level === 'error' ? 'Error: ' : ''}{i.message}</span></div>)}</div>
          )}
          {errors > 0 && <p className="warn-text small">{errors} error{errors > 1 ? 's' : ''}: analysis still runs, but affected nodes may have unknown shapes. Run <code>onnx.checker.check_model</code> on the export.</p>}

          {Object.keys(pins).length > 0 && (
            <>
              <h3>Map dynamic dims</h3>
              <p className="muted small">These inputs have symbolic dims. Values below are used for memory and latency estimates and for the pinned-shape fix. They do not change the file.</p>
              <div className="map-grid">
                {Object.keys(pins).map(s => (
                  <label key={s} className="field"><span className="mono">{s}</span>
                    <input type="number" min={1} value={pins[s]} onChange={e => setPins({ ...pins, [s]: Math.max(1, Math.round(Number(e.target.value) || 1)) })} />
                    <small className="muted">in {preview.inputs.filter(i => i.dynamic.includes(s)).map(i => i.name).join(', ')}</small>
                  </label>
                ))}
              </div>
            </>
          )}
          <div className="row gap wrap end">
            <label className="field grow"><span>Version label</span><input value={label} onChange={e => setLabel(e.target.value)} /></label>
            <button type="button" className="btn ghost" onClick={() => { setParsed(null); if (input.current) input.current.value = '' }}>Cancel</button>
            <button type="button" className="btn primary" disabled={!!busy} onClick={() => void save()}>Save version and analyse</button>
          </div>
        </div>
      )}
    </div>
  )
}
