// node scripts/qa-sync.mjs : signs up a throwaway account, round-trips a project and a version through the same tables the app syncs to, then deletes the account
import { readFileSync } from 'node:fs'
import { createClient } from '@supabase/supabase-js'
const env = Object.fromEntries(readFileSync(new URL('../.env', import.meta.url), 'utf8').split('\n').filter(l => l.includes('=') && !l.startsWith('#')).map(l => l.split('=')))
const sb = createClient(env.VITE_SUPABASE_URL, env.VITE_SUPABASE_PUBLISHABLE_KEY, { auth: { persistSession: false } })
const email = `qa-npu-check-${Date.now()}@example.com`, password = 'qa-' + Math.random().toString(36).slice(2) + 'X9!'
const ok = (label, r) => { if (r.error) { console.log('FAIL', label, r.error.message); process.exitCode = 1 } else console.log('ok  ', label) }
const up = await sb.auth.signUp({ email, password }); ok('sign up ' + email, up)
if (!up.data.session) ok('sign in', await sb.auth.signInWithPassword({ email, password }))
const id = crypto.randomUUID(), vid = crypto.randomUUID()
ok('upsert project', await sb.from('projects').upsert({ id, app: 'npu-check', name: 'QA model', data: { project: { id, name: 'QA model', settings: { pins: { seq: 128 }, applied: ['pin-dims'] } } }, updated_at: new Date().toISOString() }))
ok('insert version', await sb.from('project_versions').insert({ id: vid, project_id: id, label: 'v1', data: { version: { id: vid, label: 'v1', fileName: 'qa.onnx', summary: { score: 85, partitions: 1 } } } }))
const rows = await sb.from('projects').select('id,name,data').eq("app", "npu-check"); ok('read back', rows)
console.log(rows.data?.length === 1 && rows.data[0].data.project.settings.pins.seq === 128 ? 'ok   round trip matches' : 'FAIL round trip')
const vs = await sb.from('project_versions').select('id').in('project_id', [id]); console.log(vs.data?.length === 1 ? 'ok   snapshot round trip' : 'FAIL snapshot')
ok('delete account', await sb.rpc('delete_my_account'))
const after = await sb.auth.signInWithPassword({ email, password }); console.log(after.error ? 'ok   account gone (' + after.error.message + ')' : 'FAIL account still exists')
