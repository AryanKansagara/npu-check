// node scripts/qa-delete.mjs <email> <password> : deletes a throwaway QA account and its rows
import { readFileSync } from 'node:fs'
import { createClient } from '@supabase/supabase-js'
const [email, password] = process.argv.slice(2)
const env = Object.fromEntries(readFileSync(new URL('../.env', import.meta.url), 'utf8').split('\n').filter(l => l.includes('=') && !l.startsWith('#')).map(l => l.split('=')))
const sb = createClient(env.VITE_SUPABASE_URL, env.VITE_SUPABASE_PUBLISHABLE_KEY, { auth: { persistSession: false } })
const { error } = await sb.auth.signInWithPassword({ email, password })
if (error) { console.log('sign-in failed:', error.message); process.exit(1) }
const r = await sb.rpc('delete_my_account')
console.log(r.error ? 'delete failed: ' + r.error.message : 'deleted ' + email)
