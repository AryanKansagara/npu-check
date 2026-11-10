// Layered DAG layout: longest-path layers, then two barycenter sweeps to reduce crossings.
import { topoOrder } from './analyze.ts'
import type { ModelIR } from './types.ts'

export interface Laid { index: number; layer: number; pos: number; x: number; y: number }
export interface Layout { nodes: Laid[]; edges: { from: number; to: number; tensor: string }[]; layers: number; width: number; height: number }

/** `dir` LR puts layers on x; TB on y. Constant nodes are skipped. */
export function layoutGraph(ir: ModelIR, opts: { dir?: 'LR' | 'TB'; gapLayer?: number; gapNode?: number; wrap?: number; rowGap?: number } = {}): Layout {
  const { dir = 'TB', gapLayer = 64, gapNode = 150, wrap = 0, rowGap = 40 } = opts
  const show = (i: number) => ir.nodes[i].op !== 'Constant'
  const producer = new Map<string, number>()
  ir.nodes.forEach((n, i) => n.outputs.forEach(o => o && producer.set(o, i)))
  const edges: Layout['edges'] = []
  const layer = new Map<number, number>()
  for (const i of topoOrder(ir)) {
    if (!show(i)) continue
    let l = 0
    for (const x of ir.nodes[i].inputs) {
      const p = producer.get(x)
      if (p === undefined || !show(p) || p === i) continue
      edges.push({ from: p, to: i, tensor: x })
      l = Math.max(l, (layer.get(p) ?? 0) + 1)
    }
    layer.set(i, l)
  }
  const nLayers = Math.max(0, ...layer.values()) + 1
  const rows: number[][] = Array.from({ length: layer.size ? nLayers : 0 }, () => [])
  for (const [i, l] of layer) rows[l].push(i)
  const pos = new Map<number, number>()
  rows.forEach(r => r.forEach((i, k) => pos.set(i, k)))
  const preds = new Map<number, number[]>(), succs = new Map<number, number[]>()
  for (const e of edges) { preds.set(e.to, [...(preds.get(e.to) ?? []), e.from]); succs.set(e.from, [...(succs.get(e.from) ?? []), e.to]) }
  const sweep = (order: number[][], nb: Map<number, number[]>) => order.forEach(r => {
    const bc = (i: number) => { const l = nb.get(i); return l?.length ? l.reduce((s, j) => s + pos.get(j)!, 0) / l.length : pos.get(i)! }
    r.sort((a, b) => bc(a) - bc(b) || a - b).forEach((i, k) => pos.set(i, k))
  })
  sweep(rows, preds)
  sweep([...rows].reverse(), succs)
  sweep(rows, preds)
  const widest = Math.max(1, ...rows.map(r => r.length))
  const nodes: Laid[] = []
  if (wrap > 0 && dir === 'LR') {
    // Boustrophedon: layers fill rows of `wrap`, alternating direction, so long chains fit a wide card.
    const band = (widest - 1) * gapNode + rowGap
    rows.forEach((r, l) => r.forEach(i => {
      const row = Math.floor(l / wrap), col = l % wrap
      const p = pos.get(i)! + (widest - r.length) / 2
      nodes.push({ index: i, layer: l, pos: p, x: (row % 2 ? wrap - 1 - col : col) * gapLayer, y: row * band + p * gapNode })
    }))
    const nRows = Math.ceil(nLayers / wrap)
    return { nodes, edges, layers: nLayers, width: (Math.min(wrap, nLayers) - 1) * gapLayer, height: (nRows - 1) * band + (widest - 1) * gapNode }
  }
  rows.forEach((r, l) => r.forEach(i => {
    // Center each layer on the widest one.
    const p = pos.get(i)! + (widest - r.length) / 2
    const across = p * gapNode, along = l * gapLayer
    nodes.push({ index: i, layer: l, pos: p, x: dir === 'TB' ? across : along, y: dir === 'TB' ? along : across })
  }))
  const w = (widest - 1) * gapNode, h = (nLayers - 1) * gapLayer
  return { nodes, edges, layers: nLayers, width: dir === 'TB' ? w : h, height: dir === 'TB' ? h : w }
}
