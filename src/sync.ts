// Optional cloud sync. Local-first: everything works signed out; signing in merges projects by id, last write wins.
// Model files never sync. Parsed graphs (no weights) sync when small; version summaries always do.
import { useSyncExternalStore } from 'react'
import type { SupabaseClient } from '@supabase/supabase-js'
import type { ModelIR } from './engine/types.ts'
import { clearTombstones, getFile, getIr, getProjects, loadProjects, onProjectChange, putIr, saveProject, tombstones, type Project, type Version } from './store.ts'

export const APP_ID = 'npu-check'
/** Parsed graphs above this stay on the device; the version summary still syncs. */
export const IR_SYNC_LIMIT = 1_500_000

const URL_ = import.meta.env.VITE_SUPABASE_URL as string | undefined
const KEY = import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY as string | undefined
export const cloudConfigured = !!(URL_ && KEY)

export type SyncStatus = 'local' | 'syncing' | 'synced' | 'offline' | 'error'
interface State { user: { id: string; email: string } | null; status: SyncStatus; error: string | null; lastSynced: number | null; ready: boolean }
let state: State = { user: null, status: 'local', error: null, lastSynced: null, ready: false }
const subs = new Set<() => void>()
const set = (p: Partial<State>) => { state = { ...state, ...p }; subs.forEach(f => f()) }
export const useSync = () => useSyncExternalStore(f => { subs.add(f); return () => subs.delete(f) }, () => state)

let clientP: Promise<SupabaseClient> | null = null
function client(): Promise<SupabaseClient> {
  if (!cloudConfigured) return Promise.reject(new Error('Cloud sync is not configured for this build.'))
  return clientP ??= import('@supabase/supabase-js').then(({ createClient }) => {
    const c = createClient(URL_!, KEY!, { auth: { persistSession: true, autoRefreshToken: true } })
    c.auth.onAuthStateChange((_e, session) => {
      const u = session?.user
      set({ user: u ? { id: u.id, email: u.email ?? '' } : null, status: u ? state.status === 'local' ? 'synced' : state.status : 'local' })
    })
    return c
  })
}

const hasStoredSession = () => Object.keys(localStorage).some(k => k.startsWith('sb-') && k.endsWith('-auth-token'))

/** Called once at boot. Only loads the Supabase client if a session is already stored. */
export async function initSync() {
  onProjectChange(() => schedule())
  addEventListener('online', () => { if (state.user) void syncNow() })
  addEventListener('offline', () => { if (state.user) set({ status: 'offline' }) })
  if (!hasStoredSession()) { set({ ready: true }); return }
  try {
    const c = await client()
    const { data } = await c.auth.getSession()
    const u = data.session?.user
    set({ user: u ? { id: u.id, email: u.email ?? '' } : null, ready: true })
    if (u) void syncNow()
  } catch (e) {
    set({ ready: true, status: 'error', error: msg(e) })
  }
}

const msg = (e: unknown) => {
  const m = e instanceof Error ? e.message : typeof e === 'object' && e && 'message' in e ? String((e as { message: unknown }).message) : String(e)
  return /fetch|network|load failed/i.test(m) ? 'Could not reach the sync server. Your work is safe on this device.' : m
}

export async function signUp(email: string, password: string) {
  const c = await client()
  const { data, error } = await c.auth.signUp({ email, password })
  if (error) throw new Error(error.message)
  if (!data.session) { await signIn(email, password); return }
  set({ user: { id: data.user!.id, email } })
  await syncNow()
}

export async function signIn(email: string, password: string) {
  const c = await client()
  const { data, error } = await c.auth.signInWithPassword({ email, password })
  if (error) throw new Error(error.message)
  set({ user: { id: data.user.id, email: data.user.email ?? email } })
  await syncNow()
}

export async function signOut() {
  const c = await client()
  await c.auth.signOut()
  set({ user: null, status: 'local', lastSynced: null, error: null })
}

let timer: ReturnType<typeof setTimeout> | undefined
function schedule() {
  if (!state.user) return
  clearTimeout(timer)
  timer = setTimeout(() => void syncNow(), 1500)
}

interface CloudRow { id: string; name: string; data: { project: Omit<Project, 'versions'> }; updated_at: string }
interface CloudVersion { id: string; project_id: string; data: { version: Version; ir?: ModelIR } }

let running: Promise<void> | null = null
export function syncNow(): Promise<void> {
  if (!state.user) return Promise.resolve()
  if (running) return running.then(() => syncNow())
  running = doSync().finally(() => { running = null })
  return running
}

async function doSync() {
  if (!navigator.onLine) { set({ status: 'offline' }); return }
  set({ status: 'syncing', error: null })
  try {
    const c = await client()
    const dead = tombstones()
    if (dead.length) {
      const { error } = await c.from('projects').delete().in('id', dead).eq('app', APP_ID)
      if (error) throw error
      clearTombstones(dead)
    }
    const { data: rows, error } = await c.from('projects').select('id,name,data,updated_at').eq('app', APP_ID)
    if (error) throw error
    const cloud = new Map((rows as CloudRow[]).map(r => [r.id, r]))
    const local = getProjects()
    const ids = [...new Set([...cloud.keys(), ...local.map(p => p.id)])]
    const { data: vrows, error: verr } = ids.length
      ? await c.from('project_versions').select('id,project_id,data').in('project_id', ids)
      : { data: [], error: null }
    if (verr) throw verr
    const cloudVersions = new Map<string, CloudVersion[]>()
    for (const v of vrows as CloudVersion[]) cloudVersions.set(v.project_id, [...(cloudVersions.get(v.project_id) ?? []), v])

    for (const id of ids) {
      const lp = local.find(p => p.id === id)
      const cp = cloud.get(id)
      const cvs = cloudVersions.get(id) ?? []
      const cloudTime = cp ? Date.parse(cp.updated_at) : 0
      if (lp && (!cp || lp.updatedAt > cloudTime)) {
        const { versions: _v, ...meta } = lp
        const { error: e1 } = await c.from('projects').upsert({ id, app: APP_ID, name: lp.name, data: { project: meta }, updated_at: new Date(lp.updatedAt).toISOString() })
        if (e1) throw e1
      } else if (cp && (!lp || cloudTime > lp.updatedAt)) {
        await saveProject({ ...cp.data.project, id, name: cp.name, updatedAt: cloudTime, versions: lp?.versions ?? [] }, { touch: false, silent: true })
      }
      // Versions: union by id in both directions; deletes go to the cloud directly (deleteVersionCloud).
      const now = getProjects().find(p => p.id === id)
      if (!now) continue
      const up = now.versions.filter(v => !cvs.some(x => x.id === v.id))
      for (const v of up) {
        const ir = now.sampleId ? undefined : await getIr(v.id)
        const fits = ir && JSON.stringify(ir).length <= IR_SYNC_LIMIT
        const { error: e2 } = await c.from('project_versions').insert({ id: v.id, project_id: id, label: v.label, data: { version: v, ...(fits ? { ir } : {}) } })
        if (e2) throw e2
      }
      const missing = cvs.filter(x => !now.versions.some(v => v.id === x.id))
      if (missing.length) {
        for (const s of missing) if (s.data.ir) await putIr(s.id, s.data.ir)
        const add = missing.map(s => ({ ...s.data.version, irMissing: !s.data.ir && !now.sampleId }))
        await saveProject({ ...now, versions: [...now.versions, ...add].sort((a, b) => a.createdAt - b.createdAt), currentId: now.currentId ?? add[add.length - 1].id }, { touch: false, silent: true })
      }
    }
    set({ status: 'synced', lastSynced: Date.now() })
  } catch (e) {
    set({ status: navigator.onLine ? 'error' : 'offline', error: msg(e) })
  }
}

export async function deleteVersionCloud(id: string) {
  if (!state.user) return
  try { const c = await client(); await c.from('project_versions').delete().eq('id', id) } catch { /* offline: it may come back on the next sync, which is harmless */ }
}

export async function deleteCloudData() {
  const c = await client()
  const { data: rows } = await c.from('projects').select('id').eq('app', APP_ID)
  const ids = (rows ?? []).map(r => r.id as string)
  if (ids.length) await c.from('project_versions').delete().in('project_id', ids)
  const { error } = await c.from('projects').delete().eq('app', APP_ID)
  if (error) throw new Error(error.message)
  set({ status: 'synced', lastSynced: Date.now() })
}

export async function deleteAccount() {
  const c = await client()
  const { error } = await c.rpc('delete_my_account')
  if (error) throw new Error(error.message)
  await c.auth.signOut()
  set({ user: null, status: 'local', lastSynced: null })
}

export async function exportAll(): Promise<Blob> {
  await loadProjects()
  const list = await Promise.all(getProjects().map(async p => ({
    ...p,
    versions: await Promise.all(p.versions.map(async v => ({ ...v, graph: await getIr(v.id), hasLocalFile: !!(await getFile(v.id)) }))),
  })))
  return new Blob([JSON.stringify({ app: APP_ID, exportedAt: new Date().toISOString(), note: 'Model files are not included; graphs are stored without weights.', projects: list }, null, 1)], { type: 'application/json' })
}
