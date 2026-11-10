import { useEffect, useState } from 'react'
import { fmtBytes } from './engine/analyze.ts'
import { SAMPLES } from './engine/samples.ts'
import { createBlankProject, createSampleProject } from './projectData.ts'
import { deleteProject, saveProject, useProjects, type Project } from './store.ts'
import { useSync } from './sync.ts'
import { ask, fmtMs, fmtSi, go, Link, SyntheticBadge, toast } from './ui.tsx'

export default function Workspace() {
  const list = useProjects()
  const sync = useSync()
  const [busy, setBusy] = useState<string | null>(null)

  // /app?sample=detector opens (or creates) the sample straight away.
  useEffect(() => {
    const q = new URLSearchParams(location.search)
    const s = q.get('sample')
    const tab = q.get('tab')?.replace(/[^a-z]/g, '')
    if (!s || list === null || busy || !SAMPLES.some(x => x.id === s)) return
    setBusy(s)
    void createSampleProject(s).then(p => go(`/app/p/${p.id}${tab ? '/' + tab : ''}`, { replace: true }))
  }, [list, busy])

  const openSample = async (id: string) => { setBusy(id); const p = await createSampleProject(id); go(`/app/p/${p.id}`) }
  const newProject = async () => {
    const p = await createBlankProject(`Model ${(list?.filter(x => !x.sampleId).length ?? 0) + 1}`)
    go(`/app/p/${p.id}/import`)
  }

  if (list === null || busy) return <div className="boot" role="status"><span className="spinner" /> {busy ? 'Building the synthetic model' : 'Opening your projects'}</div>

  return (
    <section className="page">
      <div className="page-head">
        <div>
          <div className="eyebrow">Workspace</div>
          <h1>Projects</h1>
          <p className="muted">One project per model you deploy. Each export you inspect becomes a version, so you can compare what changed. {sync.user ? 'Reports sync to your account; model files stay on this device.' : 'Stored on this device.'}</p>
        </div>
        {list.length > 0 && <button type="button" className="btn primary" onClick={newProject}>New project</button>}
      </div>

      {list.length === 0 ? (
        <div className="onboard">
          <article className="onboard-card feature-card">
            <div className="eyebrow">Two minutes</div>
            <h2>Try the sample</h2>
            <p>Three tiny synthetic models built in code with random weights, each with a different deployment problem. Everything is clickable.</p>
            <div className="stack-sm">
              {SAMPLES.map(s => (
                <button key={s.id} type="button" className="sample-btn" onClick={() => void openSample(s.id)}>
                  <span><strong>{s.name}</strong><small>{s.blurb}</small></span><span aria-hidden="true">&rarr;</span>
                </button>
              ))}
            </div>
          </article>
          <article className="onboard-card">
            <div className="eyebrow">Your model</div>
            <h2>Import your model</h2>
            <p>Drop an <code>.onnx</code>, <code>.tflite</code> or <code>.gguf</code> file. The format is detected from its bytes, the graph is validated with node-level locations, and dynamic dims get suggested values you can change.</p>
            <div className="row gap wrap">
              <button type="button" className="btn primary" onClick={newProject}>Import your model</button>
              <Link href="/docs" className="btn ghost">Export guides</Link>
            </div>
            <p className="muted small">Files are parsed in your browser and never uploaded.</p>
          </article>
        </div>
      ) : (
        <>
          <div className="project-grid">{list.map(p => <ProjectCard key={p.id} p={p} />)}</div>
          <div className="samples-row">
            <span className="muted small">Samples:</span>
            {SAMPLES.map(s => <button key={s.id} type="button" className="linkish small" onClick={() => void openSample(s.id)}>{s.name}</button>)}
          </div>
        </>
      )}
    </section>
  )
}

function ProjectCard({ p }: { p: Project }) {
  const [renaming, setRenaming] = useState(false)
  const v = p.versions.find(x => x.id === p.currentId) ?? p.versions[p.versions.length - 1]
  const s = v?.summary
  return (
    <article className="project-card">
      <div className="row between top gap-sm">
        {renaming ? (
          <form className="grow" onSubmit={e => { e.preventDefault(); const name = String(new FormData(e.currentTarget).get('name')).trim(); if (name) void saveProject({ ...p, name }); setRenaming(false) }}>
            <label className="sr-only" htmlFor={`rn-${p.id}`}>Project name</label>
            <input id={`rn-${p.id}`} name="name" defaultValue={p.name} autoFocus onBlur={e => e.currentTarget.form?.requestSubmit()} />
          </form>
        ) : <h3><Link href={`/app/p/${p.id}`}>{p.name}</Link></h3>}
        {p.sampleId && <SyntheticBadge />}
      </div>
      <p className="muted small mono break">{v ? `${v.fileName} · ${v.format.toUpperCase()} · ${fmtBytes(v.fileSize)}` : 'No model imported yet'}</p>
      <dl className="mini-stats">
        <div><dt>Readiness</dt><dd>{s ? s.score : '—'}</dd></div>
        <div><dt>Partitions</dt><dd>{s ? s.partitions : '—'}</dd></div>
        <div><dt>Est. latency</dt><dd>{s ? fmtMs(s.latencyMs) : '—'}</dd></div>
        <div><dt>Params</dt><dd>{s ? fmtSi(s.params) : '—'}</dd></div>
      </dl>
      <div className="row between">
        <span className="muted small">{p.versions.length} version{p.versions.length === 1 ? '' : 's'} · updated {new Date(p.updatedAt).toLocaleDateString()}</span>
        <div className="row gap-sm">
          <button type="button" className="btn ghost sm" onClick={() => setRenaming(true)}>Rename</button>
          <button type="button" className="btn danger-ghost sm" onClick={async () => {
            if (!(await ask(`Delete “${p.name}”?`, 'Removes the project, its versions and stored graphs from this device (and the cloud, if you are signed in). You can undo for a few seconds.', 'Delete project'))) return
            const undo = await deleteProject(p.id)
            toast(`Deleted ${p.name}`, { undo: () => void undo() })
          }}>Delete</button>
        </div>
      </div>
    </article>
  )
}
