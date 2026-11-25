import { test as base, expect, type Locator, type Page } from '@playwright/test'
import { readFile } from 'node:fs/promises'

/** The only outside hosts the page asks for: web fonts in index.html. They get an empty stylesheet. */
const FONT_HOSTS = new Set(['fonts.googleapis.com', 'fonts.gstatic.com'])

export const test = base.extend({
  context: async ({ context, baseURL }, use) => {
    const errors: string[] = [], outside: string[] = []
    await context.route('**/*', route => {
      const url = new URL(route.request().url())
      if (url.origin === new URL(baseURL!).origin) return route.continue()
      if (FONT_HOSTS.has(url.host)) return route.fulfill({ status: 200, contentType: 'text/css', body: '' })
      outside.push(url.host)
      return route.abort('blockedbyclient')
    })
    context.on('weberror', e => errors.push(String(e.error())))
    // The clipboard is replaced so a test can read what the page copied without browser permissions.
    await context.addInitScript(() => {
      const w = window as unknown as { copied: string }
      Object.defineProperty(navigator, 'clipboard', { value: { writeText: async (t: string) => { w.copied = t } }, configurable: true })
    })
    await use(context)
    expect(errors, 'uncaught page errors').toEqual([])
    expect(outside, 'requests to other hosts').toEqual([])
  },
})
export { expect }

export const sampleFile = (name: string) => readFile(new URL(`../public/samples/${name}`, import.meta.url))

export async function openSample(page: Page, name: string) {
  await page.goto('/app')
  await page.getByRole('button', { name }).click()
  await expect(page.locator('.kpis')).toBeVisible()
}

/** Opens a new project on its import screen and gives it a file. */
export async function choose(page: Page, name: string, bytes: Buffer) {
  await page.goto('/app')
  await page.getByRole('button', { name: /^(Import your model|New project)$/ }).click()
  await giveFile(page, name, bytes)
}
export const giveFile = (page: Page, name: string, buffer: Buffer) => page.locator('#file').setInputFiles({ name, mimeType: 'application/octet-stream', buffer })

export async function save(page: Page) {
  await page.getByRole('button', { name: 'Save version and analyse' }).click()
  await expect(page.locator('.kpis')).toBeVisible()
}
export async function importModel(page: Page, name: string, bytes: Buffer) {
  await choose(page, name, bytes)
  await save(page)
}

export const tab = (page: Page, name: string) => page.getByRole('navigation', { name: 'Project sections' }).getByRole('link', { name, exact: true })
export const kpi = (page: Page, label: string): Locator => page.locator('.kpi').filter({ has: page.locator('small', { hasText: new RegExp(`^${label}$`) }) })
export const fact = (page: Page, label: string): Locator => page.locator('.facts > div').filter({ has: page.locator('dt', { hasText: new RegExp(`^${label}$`) }) }).locator('dd')
export const finding = (page: Page, text: string | RegExp) => page.locator('.finding').filter({ hasText: text })

export async function download(page: Page, button: string) {
  const [d] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: button }).click()])
  return { name: d.suggestedFilename(), bytes: await readFile(await d.path()) }
}

/** RFC 4180 reader, so quoting mistakes in an export are seen. */
export function parseCsv(text: string): string[][] {
  const rows: string[][] = [[]]
  let cell = '', quoted = false
  for (let i = 0; i < text.length; i++) {
    const c = text[i]
    if (quoted) { if (c === '"' && text[i + 1] === '"') { cell += '"'; i++ } else if (c === '"') quoted = false; else cell += c }
    else if (c === '"') quoted = true
    else if (c === ',') { rows.at(-1)!.push(cell); cell = '' }
    else if (c === '\n' || c === '\r') { if (c === '\r' && text[i + 1] === '\n') i++; rows.at(-1)!.push(cell); cell = ''; rows.push([]) }
    else cell += c
  }
  if (cell || rows.at(-1)!.length) rows.at(-1)!.push(cell)
  else rows.pop()
  return rows
}
export async function nodeCsv(page: Page) {
  await tab(page, 'Ops').click()
  const [head, ...rows] = parseCsv((await download(page, 'Node CSV')).bytes.toString())
  return rows.map(r => Object.fromEntries(head.map((h, i) => [h, r[i]])))
}

/** Text of the code block with this label, as the Copy button copies it. */
export async function copyCode(page: Page, label: string | RegExp): Promise<string> {
  await page.locator('.code').filter({ has: page.locator('.code-head span', { hasText: label }) }).getByRole('button').click()
  return page.evaluate(() => (window as unknown as { copied: string }).copied)
}
