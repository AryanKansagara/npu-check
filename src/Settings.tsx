import { useState } from 'react'
import { DEVICES, type DeviceProfile } from './engine/analyze.ts'
import { getDefaults, setDefaults, type Defaults } from './store.ts'
import { ask, setTheme, toast, useDark } from './ui.tsx'

export default function Settings() {
  const [d, setD] = useState<Defaults>(getDefaults)
  const dark = useDark()
  const save = (next: Defaults) => { setD(next); setDefaults(next) }
  const setDev = (i: number, patch: Partial<DeviceProfile>) => save({ ...d, devices: d.devices.map((x, k) => k === i ? { ...x, ...patch } : x) })
  return (
    <section className="page narrow">
      <div className="page-head"><div><div className="eyebrow">Settings</div><h1>Settings</h1><p className="muted">Defaults for new projects on this device. Each project keeps its own copy of the device profile.</p></div></div>
      <div className="stack">
        <div className="panel">
          <h2>Appearance</h2>
          <div className="seg-control" role="group" aria-label="Theme">
            <button type="button" className={!dark ? 'on' : ''} onClick={() => setTheme('light')}>Light</button>
            <button type="button" className={dark ? 'on' : ''} onClick={() => setTheme('dark')}>Dark</button>
          </div>
        </div>
        <div className="panel">
          <h2>Import defaults</h2>
          <div className="inline-fields">
            <label className="field"><span>Default sequence length</span><input type="number" min={1} value={d.seqLen} onChange={e => save({ ...d, seqLen: Math.max(1, Math.round(Number(e.target.value) || 1)) })} /><small className="muted">Suggested for dims named seq, len or token.</small></label>
            <label className="field"><span>Default device</span>
              <select value={d.deviceId} onChange={e => save({ ...d, deviceId: e.target.value })}>{d.devices.map(x => <option key={x.id} value={x.id}>{x.name}</option>)}</select>
            </label>
          </div>
          <label className="row gap-sm"><span className="switch"><input type="checkbox" checked={d.keepFiles} onChange={e => save({ ...d, keepFiles: e.target.checked })} /><span /></span>Keep ONNX files on this device (needed to download the pinned-shape file later; up to 400 MB each)</label>
        </div>
        <div className="panel">
          <div className="panel-head"><h2>Device profiles</h2><button type="button" className="btn ghost sm" onClick={async () => { if (await ask('Reset device profiles?', 'Restores the three default profiles. Projects keep their own copies.', 'Reset', false)) { save({ ...d, devices: DEVICES }); toast('Profiles reset', { tone: 'ok' }) } }}>Reset</button></div>
          {d.devices.map((x, i) => (
            <fieldset key={x.id} className="device-fs">
              <legend><input aria-label="Profile name" value={x.name} onChange={e => setDev(i, { name: e.target.value })} /></legend>
              <div className="inline-fields">
                {([['npuTops', 'NPU int8 TOPS'], ['memGBs', 'Memory GB/s'], ['cpuGflops', 'CPU GFLOPS'], ['vtcmMB', 'NPU scratch MB'], ['switchMs', 'Switch ms'], ['util', 'Utilisation']] as const).map(([k, l]) => (
                  <label key={k} className="field"><span>{l}</span><input type="number" min={0} step="any" value={x[k]} onChange={e => { const v = Number(e.target.value); if (Number.isFinite(v) && v >= 0) setDev(i, { [k]: v }) }} /></label>
                ))}
              </div>
              <p className="muted small">{x.note}</p>
            </fieldset>
          ))}
          <button type="button" className="btn outline sm" onClick={() => save({ ...d, devices: [...d.devices, { ...DEVICES[1], id: `custom-${Date.now()}`, name: 'My device', note: 'Custom profile. Enter measured numbers.' }] })}>Add profile</button>
        </div>
        <div className="panel">
          <h2>Local data</h2>
          <p className="muted">Projects, graphs and stored files live in this browser's IndexedDB. Clearing removes them from this device only.</p>
          <button type="button" className="btn danger-ghost" onClick={async () => {
            if (!(await ask('Clear all local data?', 'Deletes every project, version and stored model file on this device. Cloud copies (if signed in) are kept and come back on the next sync. This cannot be undone.', 'Clear local data'))) return
            indexedDB.deleteDatabase('npu-check')
            localStorage.removeItem('npu-check.defaults')
            localStorage.removeItem('npu-check.tombstones')
            location.href = '/app'
          }}>Clear local data</button>
        </div>
      </div>
    </section>
  )
}
