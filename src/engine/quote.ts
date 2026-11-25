// Names in a model file are written by someone else. These turn them into data for commands meant to be pasted.

// C0/C1 controls would act on a terminal while pasting; U+2028/2029 end a line in some editors.
const CONTROL = /[\x00-\x1f\x7f-\x9f\u2028\u2029]/g
const hex = (c: string) => `\\u${c.charCodeAt(0).toString(16).padStart(4, '0')}`

/** One POSIX shell word (bash and zsh when control characters need escaping). Plain names are left bare. */
export function sh(s: string): string {
  if (/^[\w@%+=:,./-]+$/.test(s)) return s
  if (s.search(CONTROL) < 0) return `'${s.replaceAll("'", `'\\''`)}'`
  return `$'${s.replace(/[\\']/g, c => `\\${c}`).replace(CONTROL, hex)}'`
}
/** A path: "./" keeps a leading dash from being read as an option. */
export const shPath = (s: string) => sh(s.startsWith('-') ? `./${s}` : s)
/** An option with its value; the "=" form is needed when the value starts with a dash. */
export const shOpt = (opt: string, s: string) => s.startsWith('-') ? `${opt}=${sh(s)}` : `${opt} ${sh(s)}`

/** A Python string literal. JSON escapes are valid Python ones. */
export const py = (s: string) => JSON.stringify(s).replace(CONTROL, hex)
export const pyList = (xs: string[]) => `[${xs.map(py).join(',')}]`
