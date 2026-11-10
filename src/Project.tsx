import { lazy, Suspense, useEffect, useState } from 'react'
import { FIX_LABELS, type Analysis, type FixId } from './engine/analyze.ts'
import type { ModelIR } from './engine/types.ts'
import { useAnalysis, useProject } from './projectData.ts'
import type { Project, Version } from './store.ts'
import { Empty, go, isTyping, Link, Modal, SyntheticBadge } from './ui.tsx'

const Tabs = lazy(() => import('./Tabs.tsx'))
const ImportTab = lazy(() => import('./Import.tsx'))

export const TABS = [['overview', 'Overview'], ['ops', 'Ops'], ['graph', 'Graph'], ['fixes', 'Fixes'], ['history', 'History'], ['import', 'Import']] as const
export type Tab = (typeof TABS)[number][0]

export interface TabProps {
  project: Project
  version: Version
  ir: ModelIR
  a: Analysis
  base: Analysis
  update(p: Partial<Project>): void
  setApplied(f: FixId, on: boolean): void
}

export default function ProjectView({ id, tab: rawTab }: { id: string; tab: string }) {
  const { loading, project, version, ir, update } = useProject(id)
  const { a, base } = useAnalysis(ir, project)
  const tab = (TABS.some(t => t[0] === rawTab) ? rawTab : 'overview') as Tab
  const [help, setHelp] = useState(false)
  useEffect(() => { if (project) document.title = `${project.name} · NPU Check` }, [project?.name])
  const setTab = (t: Tab) => go(`/app/p/${id}${t === 'overview' ? '' : '/' + t}`, { keepScroll: true })

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (isTyping(e) || e.metaKey || e.ctrlKey || e.altKey) return
      const n = Number(e.key)
      if (n >= 1 && n <= TABS.length) { e.preventDefault(); setTab(TABS[n - 1][0]) }
      else if (e.key === '?') setHelp(true)
      else if (e.key === 'p' && version) go(`/app/p/${id}/report`)
      else if (e.key === '/' && tab === 'graph') { e.preventDefault(); document.getElementById('graph-search')?.focus() }
    }
    addEventListener('keydown', onKey)
    return () => removeEventListener('keydown', onKey)
  })

  if (loading) return <div className="boot" role="status"><span className="spinner" /> Reading the graph</div>
  if (!project) return (
    <section className="page narrow"><Empty title="Project not found"><p className="muted">It may have been deleted, or it lives on another device. Sign in to sync it here.</p><Link href="/app" className="btn primary">All projects</Link></Empty></section>
  )
  const setApplied = (f: FixId, on: boolean) => update({ settings: { ...project.settings, applied: on ? [...new Set([...project.settings.applied, f])] : project.settings.applied.filter(x => x !== f) } })

  return (
    <section className="page wide">
      <div className="proj-head">
        <div className="grow">
          <div className="row gap-sm wrap"><Link href="/app" className="muted small">Projects</Link><span className="muted small">/</span>{project.sampleId && <SyntheticBadge />}</div>
          <h1>{project.name}</h1>
          {version && (
            <div className="row gap-sm wrap">
              <label className="sr-only" htmlFor="ver">Version</label>
              <select id="ver" value={version.id} onChange={e => update({ currentId: e.target.value })}>
                {project.versions.map(v => <option key={v.id} value={v.id}>{v.label} · {v.fileName}</option>)}
              </select>
              {project.settings.applied.length > 0 && <span className="badge sim" title="Estimates include the simulated fixes">Simulating: {project.settings.applied.map(f => FIX_LABELS[f]).join(', ')}</span>}
            </div>
          )}
        </div>
        {version && <div className="row gap-sm"><Link href={`/app/p/${id}/report`} className="btn ghost sm">Readiness report</Link><button type="button" className="btn ghost sm" onClick={() => setHelp(true)} aria-label="Keyboard shortcuts">?</button></div>}
      </div>
      <nav className="tabs" aria-label="Project sections">
        {TABS.map(([t, l], i) => <Link key={t} href={`/app/p/${id}${t === 'overview' ? '' : '/' + t}`} aria-current={t === tab ? 'page' : undefined} title={`${l} (${i + 1})`}>{l}</Link>)}
      </nav>
      <Suspense fallback={<div className="boot"><span className="spinner" /></div>}>
        {tab === 'import' || !version ? <ImportTab project={project} first={!version} />
          : !ir || !a || !base ? <MissingGraph version={version} id={id} />
          : <Tabs tab={tab} project={project} version={version} ir={ir} a={a} base={base} update={update} setApplied={setApplied} />}
      </Suspense>
      <Modal open={help} onClose={() => setHelp(false)} title="Keyboard shortcuts">
        <dl className="shortcuts">
          {TABS.map(([, l], i) => <div key={l}><dt><kbd>{i + 1}</kbd></dt><dd>{l}</dd></div>)}
          <div><dt><kbd>/</kbd></dt><dd>Search the graph</dd></div>
          <div><dt><kbd>p</kbd></dt><dd>Printable readiness report</dd></div>
          <div><dt><kbd>⌘</kbd> <kbd>Z</kbd></dt><dd>Undo the last delete</dd></div>
          <div><dt><kbd>?</kbd></dt><dd>This help</dd></div>
        </dl>
      </Modal>
    </section>
  )
}

function MissingGraph({ version, id }: { version: Version; id: string }) {
  return (
    <div className="panel">
      <Empty title="This version's graph is on another device">
        <p className="muted">The summary synced ({version.summary.nodeCount} nodes, readiness {version.summary.score}), but the parsed graph was too large to sync. Re-import <code>{version.fileName}</code> here to see the full analysis.</p>
        <div className="row gap center-x"><Link href={`/app/p/${id}/history`} className="btn ghost">History</Link><Link href={`/app/p/${id}/import`} className="btn primary">Import again</Link></div>
      </Empty>
    </div>
  )
}
