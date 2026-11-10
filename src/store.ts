import { useSyncExternalStore } from 'react'
import { DEVICES, type DeviceProfile, type FixId, type Summary } from './engine/analyze.ts'
import type { ModelIR } from './engine/types.ts'

export interface Version {
  id: string
  label: string
  fileName: string
  format: ModelIR['format']
  fileSize: number
  createdAt: number
  summary: Summary
  /** True when this device has neither the parsed graph nor the file (synced from elsewhere). */
  irMissing?: boolean
}

export interface ProjectSettings {
  device: DeviceProfile
  pins: Record<string, number>
  applied: FixId[]
}

export interface Project {
  id: string
  name: string
  createdAt: number
  updatedAt: number
  versions: Version[]
  /** Version shown on the project pages */
  currentId: string | null
  settings: ProjectSettings
  /** Built-in synthetic sample; its graph is regenerated instead of stored. */
  sampleId?: string
}

// ---------- Tiny IndexedDB wrapper (falls back to memory in private modes) ----------

const DB = 'npu-check'
type StoreName = 'projects' | 'data'
let dbp: Promise<IDBDatabase | null> | null = null
function db(): Promise<IDBDatabase | null> {
  return dbp ??= new Promise(res => {
    try {
      const req = indexedDB.open(DB, 1)
      req.onupgradeneeded = () => {
        req.result.createObjectStore('projects', { keyPath: 'id' })
        req.result.createObjectStore('data')
      }
      req.onsuccess = () => res(req.result)
      req.onerror = () => res(null)
    } catch { res(null) }
  })
}
const mem = { projects: new Map<string, unknown>(), data: new Map<string, unknown>() }

async function tx<T>(store: StoreName, mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const d = await db()
  if (!d) throw new Error('no-idb')
  return new Promise((res, rej) => {
    const r = fn(d.transaction(store, mode).objectStore(store))
    r.onsuccess = () => res(r.result)
    r.onerror = () => rej(r.error)
  })
}
async function idbGetAll<T>(store: 'projects'): Promise<T[]> {
  try { return await tx(store, 'readonly', s => s.getAll()) as T[] } catch { return [...mem[store].values()] as T[] }
}
async function idbGet<T>(store: StoreName, key: string): Promise<T | undefined> {
  try { return await tx(store, 'readonly', s => s.get(key)) as T | undefined } catch { return mem[store].get(key) as T | undefined }
}
async function idbPut(store: StoreName, value: unknown, key?: string) {
  try { await tx(store, 'readwrite', s => key === undefined ? s.put(value) : s.put(value, key)) } catch { mem[store].set(key ?? (value as { id: string }).id, value) }
}
async function idbDel(store: StoreName, key: string) {
  try { await tx(store, 'readwrite', s => s.delete(key)) } catch { mem[store].delete(key) }
}

// ---------- In-memory project list with subscribers ----------

let projects: Project[] = []
let loaded = false
const subs = new Set<() => void>()
const emit = () => { projects = [...projects]; subs.forEach(f => f()) }
const changeHooks = new Set<(p: Project) => void>()
export const onProjectChange = (f: (p: Project) => void) => { changeHooks.add(f); return () => changeHooks.delete(f) }

export async function loadProjects() {
  projects = (await idbGetAll<Project>('projects')).sort((a, b) => b.updatedAt - a.updatedAt)
  loaded = true
  emit()
}

export function useProjects() {
  return useSyncExternalStore(f => { subs.add(f); return () => subs.delete(f) }, () => loaded ? projects : null)
}

export const getProjects = () => projects
export const getProject = (id: string) => projects.find(p => p.id === id)

export async function saveProject(p: Project, opts: { touch?: boolean; silent?: boolean } = {}) {
  const next = opts.touch === false ? p : { ...p, updatedAt: Date.now() }
  await idbPut('projects', next)
  const i = projects.findIndex(x => x.id === p.id)
  if (i >= 0) projects[i] = next
  else projects.unshift(next)
  projects.sort((a, b) => b.updatedAt - a.updatedAt)
  emit()
  if (!opts.silent) changeHooks.forEach(f => f(next))
  return next
}

// Per-version local data: the parsed graph (no weights) and, for ONNX, the original file for the shape fixer.
export const getIr = (vid: string) => idbGet<ModelIR>('data', `ir:${vid}`)
export const putIr = (vid: string, ir: ModelIR) => idbPut('data', ir, `ir:${vid}`)
export const getFile = (vid: string) => idbGet<Blob>('data', `file:${vid}`)
export const putFile = (vid: string, b: Blob) => idbPut('data', b, `file:${vid}`)
async function dropVersionData(vid: string) { await idbDel('data', `ir:${vid}`); await idbDel('data', `file:${vid}`) }

/** Deletes locally; returns an undo function that restores the project and its local data. */
export async function deleteProject(id: string): Promise<() => Promise<void>> {
  const p = getProject(id)
  const kept = p ? await Promise.all(p.versions.map(async v => [v.id, await getIr(v.id), await getFile(v.id)] as const)) : []
  projects = projects.filter(x => x.id !== id)
  emit()
  await idbDel('projects', id)
  for (const v of p?.versions ?? []) await dropVersionData(v.id)
  addTombstone(id)
  return async () => {
    if (!p) return
    removeTombstone(id)
    for (const [vid, ir, file] of kept) { if (ir) await putIr(vid, ir); if (file) await putFile(vid, file) }
    await saveProject(p)
  }
}

export async function deleteVersion(p: Project, vid: string): Promise<() => Promise<void>> {
  const v = p.versions.find(x => x.id === vid)
  const ir = await getIr(vid), file = await getFile(vid)
  const versions = p.versions.filter(x => x.id !== vid)
  await saveProject({ ...p, versions, currentId: p.currentId === vid ? versions[versions.length - 1]?.id ?? null : p.currentId })
  await dropVersionData(vid)
  return async () => {
    if (!v) return
    if (ir) await putIr(vid, ir)
    if (file) await putFile(vid, file)
    const now = getProject(p.id) ?? p
    await saveProject({ ...now, versions: [...now.versions, v].sort((a, b) => a.createdAt - b.createdAt) })
  }
}

export const uuid = () => crypto.randomUUID()

// Tombstones keep a project deleted offline from coming back on the next sync.
const TOMB = 'npu-check.tombstones'
export const tombstones = (): string[] => { try { return JSON.parse(localStorage.getItem(TOMB) ?? '[]') } catch { return [] } }
const addTombstone = (id: string) => localStorage.setItem(TOMB, JSON.stringify([...new Set([...tombstones(), id])]))
const removeTombstone = (id: string) => localStorage.setItem(TOMB, JSON.stringify(tombstones().filter(x => x !== id)))
export const clearTombstones = (ids: string[]) => localStorage.setItem(TOMB, JSON.stringify(tombstones().filter(x => !ids.includes(x))))

// ---------- App defaults (settings page) ----------

export interface Defaults { devices: DeviceProfile[]; deviceId: string; seqLen: number; keepFiles: boolean }
const DEF_KEY = 'npu-check.defaults'
export function getDefaults(): Defaults {
  const base: Defaults = { devices: DEVICES, deviceId: DEVICES[0].id, seqLen: 128, keepFiles: true }
  try { return { ...base, ...JSON.parse(localStorage.getItem(DEF_KEY) ?? '{}') } } catch { return base }
}
export const setDefaults = (d: Defaults) => localStorage.setItem(DEF_KEY, JSON.stringify(d))

export function blankSettings(): ProjectSettings {
  const d = getDefaults()
  return { device: { ...(d.devices.find(x => x.id === d.deviceId) ?? d.devices[0] ?? DEVICES[0]) }, pins: {}, applied: [] }
}
