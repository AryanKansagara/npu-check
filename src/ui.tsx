import { useEffect, useRef, useState, useSyncExternalStore, type AnchorHTMLAttributes, type ReactNode } from 'react'
import { cloudConfigured, deleteAccount, deleteCloudData, exportAll, signIn, signOut, signUp, syncNow, useSync, type SyncStatus } from './sync.ts'

// ---------- Navigation ----------

export function go(href: string, opts: { replace?: boolean; keepScroll?: boolean } = {}) {
  if (opts.replace) history.replaceState({}, '', href)
  else history.pushState({}, '', href)
  dispatchEvent(new PopStateEvent('popstate'))
  if (!opts.keepScroll) scrollTo(0, 0)
}

export function Link({ href, children, ...rest }: AnchorHTMLAttributes<HTMLAnchorElement> & { href: string; children: ReactNode }) {
  return (
    <a {...rest} href={href} onClick={e => {
      rest.onClick?.(e)
      if (e.defaultPrevented || e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return
      e.preventDefault()
      go(href)
    }}>{children}</a>
  )
}

// ---------- Formatting ----------

export const fmtN = (n: number, d = 0) => Number.isFinite(n) ? n.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d }) : '—'
export const fmtPct = (n: number, d = 0) => Number.isFinite(n) ? `${(n * 100).toFixed(d)}%` : '—'
export const fmtDate = (t: number) => new Date(t).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
/** 1234567 -> "1.23M" */
export const fmtSi = (n: number, d = 2) => !Number.isFinite(n) ? '—' : n >= 1e12 ? `${(n / 1e12).toFixed(d)}T` : n >= 1e9 ? `${(n / 1e9).toFixed(d)}G` : n >= 1e6 ? `${(n / 1e6).toFixed(d)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(d === 2 ? 1 : d)}K` : String(Math.round(n))
export const fmtMs = (ms: number) => !Number.isFinite(ms) ? '—' : ms < 0.01 ? `${(ms * 1000).toFixed(1)} µs` : ms < 10 ? `${ms.toFixed(2)} ms` : ms < 1000 ? `${ms.toFixed(0)} ms` : `${(ms / 1000).toFixed(2)} s`
export const signed = (n: number, f: (n: number) => string) => (n > 0 ? '+' : n < 0 ? '−' : '') + f(Math.abs(n))

// ---------- Theme ----------

export function useDark() {
  const [mode, setMode] = useState(() => document.documentElement.dataset.themeMode ?? 'light')
  useEffect(() => {
    const ob = new MutationObserver(() => setMode(document.documentElement.dataset.themeMode ?? 'light'))
    ob.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme-mode'] })
    return () => ob.disconnect()
  }, [])
  return mode === 'dark'
}

export const setTheme = (m: 'light' | 'dark') => { document.documentElement.dataset.themeMode = m; localStorage.setItem('themeMode', m) }

export function ThemeToggle() {
  const dark = useDark()
  return (
    <button className="icon-btn" type="button" aria-label={dark ? 'Switch to light theme' : 'Switch to dark theme'} title={dark ? 'Light theme' : 'Dark theme'} onClick={() => setTheme(dark ? 'light' : 'dark')}>
      <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
        {dark
          ? <><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></>
          : <path d="M21 14.5A9 9 0 0 1 9.5 3a9 9 0 1 0 11.5 11.5Z" />}
      </svg>
    </button>
  )
}

export const Monogram = () => <span className="monogram" aria-hidden="true">NC</span>

// ---------- Toasts with undo ----------

interface Toast { id: number; message: string; tone?: 'ok' | 'error'; action?: { label: string; run: () => void }; ttl: number }
let toasts: Toast[] = []
const toastSubs = new Set<() => void>()
let nextId = 1
let lastUndo: { run: () => void; until: number } | null = null
const emitToasts = () => toastSubs.forEach(f => f())

export function toast(message: string, opts: { tone?: Toast['tone']; undo?: () => void; action?: Toast['action']; ttl?: number } = {}) {
  const ttl = opts.ttl ?? (opts.undo ? 8000 : 4200)
  const t: Toast = { id: nextId++, message, tone: opts.tone, ttl, action: opts.undo ? { label: 'Undo', run: opts.undo } : opts.action }
  if (opts.undo) lastUndo = { run: opts.undo, until: Date.now() + ttl }
  toasts = [...toasts.slice(-3), t]
  emitToasts()
  setTimeout(() => dismiss(t.id), ttl)
}
const dismiss = (id: number) => { toasts = toasts.filter(t => t.id !== id); emitToasts() }

export function Toasts() {
  const list = useSyncExternalStore(f => { toastSubs.add(f); return () => toastSubs.delete(f) }, () => toasts)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'z' && !e.shiftKey && lastUndo && Date.now() < lastUndo.until && !isTyping(e)) {
        e.preventDefault()
        const u = lastUndo
        lastUndo = null
        u.run()
        toasts = []
        emitToasts()
      }
    }
    addEventListener('keydown', onKey)
    return () => removeEventListener('keydown', onKey)
  }, [])
  return (
    <div className="toasts" role="status" aria-live="polite">
      {list.map(t => (
        <div key={t.id} className={`toast ${t.tone ?? ''}`}>
          <span>{t.message}</span>
          {t.action && <button type="button" onClick={() => { t.action!.run(); if (t.action!.label === 'Undo') lastUndo = null; dismiss(t.id) }}>{t.action.label}</button>}
          <button type="button" className="toast-x" aria-label="Dismiss" onClick={() => dismiss(t.id)}>&times;</button>
        </div>
      ))}
    </div>
  )
}

export const isTyping = (e: KeyboardEvent) => {
  const el = e.target as HTMLElement | null
  return !!el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT' || el.isContentEditable)
}

// ---------- Small building blocks ----------

export function CopyButton({ text, label = 'Copy', className = '' }: { text: string; label?: string; className?: string }) {
  const [done, setDone] = useState(false)
  return (
    <button type="button" className={`btn ghost sm ${className}`} onClick={async () => {
      try { await navigator.clipboard.writeText(text); setDone(true); setTimeout(() => setDone(false), 1400) } catch { toast('Copy failed. Select the text and copy it manually.', { tone: 'error' }) }
    }}>{done ? 'Copied' : label}</button>
  )
}

export function download(name: string, content: string | Blob, type = 'text/csv') {
  const blob = typeof content === 'string' ? new Blob([content], { type }) : content
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = name
  a.click()
  setTimeout(() => URL.revokeObjectURL(a.href), 1000)
}

export function Code({ code, label }: { code: string; label?: string }) {
  return (
    <div className="code">
      <div className="code-head"><span>{label ?? 'SQL'}</span><CopyButton text={code} /></div>
      <pre><code>{code}</code></pre>
    </div>
  )
}

export function Modal({ open, onClose, title, children, wide }: { open: boolean; onClose(): void; title: string; children: ReactNode; wide?: boolean }) {
  const ref = useRef<HTMLDialogElement>(null)
  useEffect(() => {
    const d = ref.current
    if (!d) return
    if (open && !d.open) d.showModal()
    if (!open && d.open) d.close()
  }, [open])
  return (
    <dialog ref={ref} className={`modal${wide ? ' wide' : ''}`} onClose={onClose} onClick={e => { if (e.target === ref.current) onClose() }} aria-label={title}>
      <div className="modal-head"><h2>{title}</h2><button className="icon-btn" type="button" aria-label="Close" onClick={onClose}>&times;</button></div>
      <div className="modal-body">{open && children}</div>
    </dialog>
  )
}

/** A confirm dialog that returns a promise. */
let confirmState: { title: string; body: string; danger?: boolean; confirm: string; resolve(v: boolean): void } | null = null
const confirmSubs = new Set<() => void>()
export function ask(title: string, body: string, confirm = 'Confirm', danger = true): Promise<boolean> {
  return new Promise(resolve => { confirmState = { title, body, confirm, danger, resolve }; confirmSubs.forEach(f => f()) })
}
export function ConfirmHost() {
  const s = useSyncExternalStore(f => { confirmSubs.add(f); return () => confirmSubs.delete(f) }, () => confirmState)
  const close = (v: boolean) => { s?.resolve(v); confirmState = null; confirmSubs.forEach(f => f()) }
  return (
    <Modal open={!!s} onClose={() => close(false)} title={s?.title ?? ''}>
      <p>{s?.body}</p>
      <div className="row end gap">
        <button type="button" className="btn ghost" onClick={() => close(false)}>Cancel</button>
        <button type="button" className={`btn ${s?.danger ? 'danger' : 'primary'}`} autoFocus onClick={() => close(true)}>{s?.confirm}</button>
      </div>
    </Modal>
  )
}

// ---------- Header, account, footer ----------

const STATUS_TEXT: Record<SyncStatus, string> = { local: 'Local only', syncing: 'Syncing', synced: 'Synced', offline: 'Offline', error: 'Sync failed' }

export function SyncPill() {
  const s = useSync()
  if (!s.user) return null
  return <span className={`sync-pill ${s.status}`} title={s.error ?? (s.lastSynced ? `Last synced ${new Date(s.lastSynced).toLocaleTimeString()}` : '')}><i aria-hidden="true" />{STATUS_TEXT[s.status]}</span>
}

function AccountMenu() {
  const s = useSync()
  const [open, setOpen] = useState(false)
  const [mode, setMode] = useState<'in' | 'up'>('in')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const wrap = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!open) return
    const onDoc = (e: MouseEvent) => { if (!wrap.current?.contains(e.target as Node) && !(e.target as HTMLElement).closest('dialog')) setOpen(false) }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
    document.addEventListener('mousedown', onDoc)
    document.addEventListener('keydown', onKey)
    return () => { document.removeEventListener('mousedown', onDoc); document.removeEventListener('keydown', onKey) }
  }, [open])

  const run = async (f: () => Promise<void>, ok?: string) => {
    setBusy(true); setErr(null)
    try { await f(); if (ok) toast(ok, { tone: 'ok' }) } catch (e) { setErr(e instanceof Error ? e.message : String(e)) } finally { setBusy(false) }
  }

  return (
    <div className="account" ref={wrap}>
      <button type="button" className={`icon-btn avatar${s.user ? ' on' : ''}`} aria-expanded={open} aria-haspopup="true" aria-label={s.user ? `Account: ${s.user.email}` : 'Sign in to sync'} onClick={() => setOpen(o => !o)}>
        {s.user ? s.user.email[0].toUpperCase() : <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true"><circle cx="12" cy="8" r="4" /><path d="M4 21c1.5-4 4.5-6 8-6s6.5 2 8 6" /></svg>}
      </button>
      {open && (
        <div className="account-pop" role="dialog" aria-label="Account">
          {!s.user ? (
            <form onSubmit={e => {
              e.preventDefault()
              const f = new FormData(e.currentTarget)
              const email = String(f.get('email')), pw = String(f.get('password'))
              void run(() => mode === 'in' ? signIn(email, pw) : signUp(email, pw), mode === 'in' ? 'Signed in. Syncing your projects.' : 'Account created. Syncing your projects.')
            }}>
              <h3>{mode === 'in' ? 'Sign in to sync' : 'Create an account'}</h3>
              <p className="muted small">Optional. Everything works signed out. Signing in syncs projects, settings, fixes and version reports across devices. Model files never leave this device.</p>
              {!cloudConfigured && <p className="error small">Sync is not configured for this build.</p>}
              <label className="field"><span>Email</span><input name="email" type="email" required autoComplete="email" /></label>
              <label className="field"><span>Password</span><input name="password" type="password" required minLength={6} autoComplete={mode === 'in' ? 'current-password' : 'new-password'} /></label>
              {err && <p className="error small" role="alert">{err}</p>}
              <button className="btn primary block" disabled={busy || !cloudConfigured}>{busy ? 'Working' : mode === 'in' ? 'Sign in' : 'Sign up'}</button>
              <button type="button" className="linkish small" onClick={() => { setMode(mode === 'in' ? 'up' : 'in'); setErr(null) }}>{mode === 'in' ? 'No account? Sign up' : 'Have an account? Sign in'}</button>
            </form>
          ) : (
            <div className="stack-sm">
              <div><div className="muted small">Signed in as</div><strong className="break">{s.user.email}</strong></div>
              <div className="row gap-sm"><SyncPill />{s.error && <span className="error small">{s.error}</span>}</div>
              <button type="button" className="btn ghost sm block" disabled={busy} onClick={() => void run(syncNow)}>Sync now</button>
              <button type="button" className="btn ghost sm block" onClick={() => void run(async () => download(`npu-check-export-${new Date().toISOString().slice(0, 10)}.json`, await exportAll()))}>Export all my data (JSON)</button>
              <button type="button" className="btn ghost sm block" onClick={() => void run(signOut, 'Signed out. Your projects stay on this device.')}>Sign out</button>
              <hr />
              <button type="button" className="btn danger-ghost sm block" disabled={busy} onClick={async () => { if (await ask('Delete cloud data?', 'Removes every NPU Check project and version from the cloud. Copies on this device are kept.', 'Delete cloud data')) void run(deleteCloudData, 'Cloud data deleted.') }}>Delete my cloud data</button>
              <button type="button" className="btn danger-ghost sm block" disabled={busy} onClick={async () => { if (await ask('Delete your account?', 'Deletes your account and all cloud data. Projects on this device are kept. This cannot be undone.', 'Delete account')) void run(deleteAccount, 'Account deleted.') }}>Delete my account</button>
              {err && <p className="error small" role="alert">{err}</p>}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export function Header() {
  const path = location.pathname
  const [menu, setMenu] = useState(false)
  const nav = [['/', 'Product'], ['/app', 'Projects'], ['/docs', 'Docs'], ['/settings', 'Settings']] as const
  return (
    <header className="header">
      <div className="header-inner">
        <Link href="/" className="brand" aria-label="NPU Check home"><Monogram /><span>NPU Check</span></Link>
        <nav className={`nav${menu ? ' open' : ''}`} aria-label="Main">
          {nav.map(([h, l]) => <Link key={h} href={h} onClick={() => setMenu(false)} aria-current={(h === '/' ? path === '/' : path.startsWith(h)) ? 'page' : undefined}>{l}</Link>)}
        </nav>
        <div className="header-actions">
          <SyncPill />
          <ThemeToggle />
          <AccountMenu />
          {path === '/' && <Link href="/app" className="btn primary sm hide-sm">Open the app</Link>}
          <button type="button" className="icon-btn burger" aria-label="Menu" aria-expanded={menu} onClick={() => setMenu(m => !m)}>
            <svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" strokeWidth="2" aria-hidden="true"><path d="M4 7h16M4 12h16M4 17h16" /></svg>
          </button>
        </div>
      </div>
      <ConfirmHost />
    </header>
  )
}

export function Footer() {
  return (
    <footer className="footer">
      <div className="footer-inner">
        <div className="footer-brand"><Monogram /> <strong>NPU Check</strong></div>
        <nav className="footer-nav" aria-label="Footer"><Link href="/app">Projects</Link><Link href="/docs">Docs</Link><Link href="/settings">Settings</Link></nav>
        <p>Built by Aryan Kansagara. Independent project, not affiliated with Qualcomm.</p>
        <p className="muted small">Sample models are synthetic (random weights). Model files are parsed in your browser and never uploaded; signing in syncs only derived reports.</p>
      </div>
    </footer>
  )
}

export function NotFound() {
  useEffect(() => { document.title = 'Not found · NPU Check' }, [])
  return (
    <section className="page narrow center notfound">
      <div className="nf-code" aria-hidden="true">404</div>
      <h1>This node has no producer</h1>
      <p className="muted">The page you asked for is not in the graph. Nothing fell back to the CPU, it just does not exist.</p>
      <div className="row gap center-x"><Link href="/" className="btn ghost">Home</Link><Link href="/app" className="btn primary">Your projects</Link></div>
    </section>
  )
}

export function Empty({ title, children, icon }: { title: string; children?: ReactNode; icon?: ReactNode }) {
  return <div className="empty">{icon ?? <div className="empty-icon" aria-hidden="true" />}<h3>{title}</h3>{children}</div>
}

export const SyntheticBadge = () => <span className="badge synthetic" title="Generated from scratch with random weights, not a real model">Synthetic</span>
