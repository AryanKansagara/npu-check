import { useEffect, useMemo, useState } from 'react'
import { analyze, summarize, symbols, suggestPin, type Analysis } from './engine/analyze.ts'
import { NeedMoreBytes } from './engine/gguf.ts'
import { parseModel } from './engine/load.ts'
import { SAMPLES } from './engine/samples.ts'
import type { ModelIR } from './engine/types.ts'
import { blankSettings, getDefaults, getIr, getProjects, putFile, putIr, saveProject, useProjects, uuid, type Project, type Version } from './store.ts'

// Sample versions are rebuilt from code: version.fileName identifies which synthetic file.
const sampleIrCache = new Map<string, ModelIR>()
export function sampleIr(fileName: string): ModelIR | null {
  const def = SAMPLES.find(s => s.file === fileName || s.v2?.file === fileName)
  if (!def) return null
  if (!sampleIrCache.has(fileName)) sampleIrCache.set(fileName, parseModel(def.file === fileName ? def.build() : def.v2!.build(), fileName))
  return sampleIrCache.get(fileName)!
}

export function sampleBytes(fileName: string): Uint8Array | null {
  const def = SAMPLES.find(s => s.file === fileName || s.v2?.file === fileName)
  return def ? (def.file === fileName ? def.build() : def.v2!.build()) : null
}

export function makeVersion(ir: ModelIR, fileName: string, label: string, p: Pick<Project, 'settings'>): Version {
  const a = analyze(ir, { ...p.settings, applied: [] })
  return { id: uuid(), label, fileName, format: ir.format, fileSize: ir.fileSize, createdAt: Date.now(), summary: summarize(a) }
}

export async function createSampleProject(sampleId: string): Promise<Project> {
  const existing = getProjects().find(p => p.sampleId === sampleId)
  if (existing) return existing
  const def = SAMPLES.find(s => s.id === sampleId)!
  const now = Date.now()
  const settings = blankSettings()
  const ir = sampleIr(def.file)!
  settings.pins = Object.fromEntries(symbols(ir).map(s => [s, s === 'seq' ? getDefaults().seqLen : suggestPin(s)]))
  const versions: Version[] = [{ ...makeVersion(ir, def.file, 'v1 export', { settings }), createdAt: now - 86400000 * 9 }]
  if (def.v2) versions.push({ ...makeVersion(sampleIr(def.v2.file)!, def.v2.file, 'v2 NPU rewrite', { settings }), createdAt: now - 86400000 * 2 })
  return saveProject({ id: uuid(), name: `${def.name} (synthetic)`, createdAt: now, updatedAt: now, versions, currentId: versions[0].id, settings, sampleId })
}

export async function createBlankProject(name: string): Promise<Project> {
  const now = Date.now()
  return saveProject({ id: uuid(), name, createdAt: now, updatedAt: now, versions: [], currentId: null, settings: blankSettings() })
}

/** Reads only as much of a GGUF file as its header needs; whole file otherwise. */
export async function readModel(file: File): Promise<ModelIR> {
  const head = new Uint8Array(await file.slice(0, 8).arrayBuffer())
  const isGguf = head[0] === 0x47 && head[1] === 0x47 && head[2] === 0x55 && head[3] === 0x46
  if (isGguf) {
    for (let mb = 8; ; mb *= 4) {
      const bytes = new Uint8Array(await file.slice(0, Math.min(file.size, mb * 1024 * 1024)).arrayBuffer())
      try { return parseModel(bytes, file.name, file.size) } catch (e) {
        if (!(e instanceof NeedMoreBytes) || bytes.length >= file.size || mb > 512) throw e
      }
    }
  }
  if (file.size > 2.2 * 1024 ** 3) throw new Error('ONNX files over 2 GB cannot be a single protobuf. Export with external data (the .onnx stays small) and drop the .onnx file.')
  return parseModel(new Uint8Array(await file.arrayBuffer()), file.name, file.size)
}

export async function addVersion(p: Project, ir: ModelIR, file: File | null, fileName: string, label: string, pins: Record<string, number>): Promise<Project> {
  const settings = { ...p.settings, pins: { ...p.settings.pins, ...pins } }
  const v = makeVersion(ir, fileName, label, { settings })
  await putIr(v.id, ir)
  if (file && ir.format === 'onnx' && getDefaults().keepFiles && file.size < 400 * 1024 * 1024) await putFile(v.id, file)
  return saveProject({ ...p, settings, versions: [...p.versions, v], currentId: v.id })
}

export async function loadIr(p: Project, v: Version): Promise<ModelIR | null> {
  if (p.sampleId) return sampleIr(v.fileName)
  return (await getIr(v.id)) ?? null
}

export function useProject(id: string) {
  const list = useProjects()
  const project = list?.find(p => p.id === id) ?? null
  const version = project?.versions.find(v => v.id === project.currentId) ?? project?.versions[project.versions.length - 1] ?? null
  const [ir, setIr] = useState<ModelIR | null | undefined>(undefined)
  useEffect(() => {
    if (!project || !version) { setIr(null); return }
    let live = true
    setIr(undefined)
    void loadIr(project, version).then(x => { if (live) setIr(x) })
    return () => { live = false }
  }, [project?.id, version?.id])
  const update = (patch: Partial<Project>) => project && saveProject({ ...project, ...patch })
  return { loading: list === null || (!!version && ir === undefined), project, version, ir: ir ?? null, update }
}

export function useAnalysis(ir: ModelIR | null, p: Project | null): { a: Analysis | null; base: Analysis | null } {
  return useMemo(() => {
    if (!ir || !p) return { a: null, base: null }
    const base = analyze(ir, { ...p.settings, applied: [] })
    const a = p.settings.applied.length ? analyze(ir, p.settings) : base
    return { a, base }
  }, [ir, p?.settings])
}
