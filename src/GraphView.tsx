import { useEffect, useMemo, useRef, useState } from 'react'
import type { Analysis } from './engine/analyze.ts'
import { layoutGraph } from './engine/layout.ts'
import type { ModelIR } from './engine/types.ts'

export const MAX_DRAW = 3000

interface Props {
  ir: ModelIR
  a: Analysis
  dir?: 'LR' | 'TB'
  compact?: boolean
  query?: string
  selected?: number | null
  onSelect?: (i: number) => void
  zoom?: number
  animate?: boolean
  label?: string
  wrap?: number
}

export function nodeClass(a: Analysis, i: number) {
  const n = a.nodes[i]
  if (n.status === 'removed') return 'removed'
  if (n.status === 'cpu') return 'cpu'
  return n.support === 'constrained' ? 'con' : 'npu'
}

export function GraphView({ ir, a, dir = 'TB', compact, query = '', selected, onSelect, zoom = 1, animate, label, wrap = 0 }: Props) {
  const W = compact ? 92 : 136, H = compact ? 26 : 40
  const layout = useMemo(() => layoutGraph(ir, { dir, wrap, rowGap: 56, gapLayer: dir === 'LR' ? (compact ? 112 : 170) : 72, gapNode: dir === 'LR' ? (compact ? 36 : 58) : 160 }), [ir, dir, compact, wrap])
  const byIdx = useMemo(() => new Map(layout.nodes.map(n => [n.index, n])), [layout])
  const q = query.trim().toLowerCase()
  const hits = useMemo(() => q ? new Set(layout.nodes.filter(n => { const m = ir.nodes[n.index]; return m.name.toLowerCase().includes(q) || m.op.toLowerCase().includes(q) || m.outputs.some(o => o.toLowerCase().includes(q)) }).map(n => n.index)) : null, [q, layout, ir])
  const svgRef = useRef<SVGSVGElement>(null)
  useEffect(() => {
    const first = hits && [...hits][0]
    const target = selected ?? first
    if (target == null || !svgRef.current) return
    const el = svgRef.current.querySelector(`[data-i="${target}"]`)
    el?.scrollIntoView?.({ block: 'center', inline: 'center', behavior: 'smooth' })
  }, [hits, selected])
  const [hover, setHover] = useState<number | null>(null)
  const pad = 16
  const vw = layout.width + W + pad * 2, vh = layout.height + H + pad * 2
  const cx = (i: number) => byIdx.get(i)!.x + pad, cy = (i: number) => byIdx.get(i)!.y + pad
  const edgePath = (f: number, t: number) => {
    if (dir === 'LR') {
      const row = (i: number) => wrap ? Math.floor(byIdx.get(i)!.layer / wrap) : 0
      if (row(f) !== row(t)) {
        const x1 = cx(f) + W / 2, y1 = cy(f) + H, x2 = cx(t) + W / 2, y2 = cy(t), m = (y1 + y2) / 2
        return `M${x1},${y1} C${x1},${m} ${x2},${m} ${x2},${y2}`
      }
      const rev = row(f) % 2 === 1
      const x1 = rev ? cx(f) : cx(f) + W, y1 = cy(f) + H / 2, x2 = rev ? cx(t) + W : cx(t), y2 = cy(t) + H / 2, m = (x1 + x2) / 2
      return `M${x1},${y1} C${m},${y1} ${m},${y2} ${x2},${y2}`
    }
    const x1 = cx(f) + W / 2, y1 = cy(f) + H, x2 = cx(t) + W / 2, y2 = cy(t), m = (y1 + y2) / 2
    return `M${x1},${y1} C${x1},${m} ${x2},${m} ${x2},${y2}`
  }
  if (layout.nodes.length > MAX_DRAW) return <p className="muted">This graph has {layout.nodes.length.toLocaleString()} nodes, more than the viewer draws ({MAX_DRAW.toLocaleString()}). Use search and the node table below instead.</p>
  const short = (s: string, n: number) => s.length > n ? s.slice(0, n - 1) + '…' : s
  return (
    <svg ref={svgRef} className={`graph${compact ? ' compact' : ''}${animate ? ' animate' : ''}`} viewBox={`0 0 ${vw} ${vh}`} width={compact ? '100%' : vw * zoom} height={compact ? undefined : vh * zoom} role="img" aria-label={label ?? `Model graph with ${layout.nodes.length} nodes, coloured by where each runs`}>
      <g className="edges">
        {layout.edges.map((e, k) => {
          const sw = a.nodes[e.from].status !== a.nodes[e.to].status
          const dim = hits && !hits.has(e.from) && !hits.has(e.to)
          return <path key={k} d={edgePath(e.from, e.to)} className={`edge${sw ? ' switch' : ''}${dim ? ' dim' : ''}`} style={animate ? { animationDelay: `${byIdx.get(e.from)!.layer * 55 + 120}ms` } : undefined} />
        })}
      </g>
      <g className="nodes">
        {layout.nodes.map(n => {
          const m = ir.nodes[n.index]
          const cls = nodeClass(a, n.index)
          const dim = hits && !hits.has(n.index)
          return (
            <g key={n.index} data-i={n.index} className={`node ${cls}${dim ? ' dim' : ''}${hits?.has(n.index) ? ' hit' : ''}${selected === n.index ? ' sel' : ''}`}
              transform={`translate(${cx(n.index)},${cy(n.index)})`} style={animate ? { animationDelay: `${n.layer * 55}ms` } : undefined}
              onClick={onSelect ? () => onSelect(n.index) : undefined} onMouseEnter={() => setHover(n.index)} onMouseLeave={() => setHover(h => h === n.index ? null : h)}
              tabIndex={onSelect ? 0 : undefined} role={onSelect ? 'button' : undefined} aria-label={onSelect ? `${m.op} ${m.name}, ${a.nodes[n.index].status}` : undefined}
              onKeyDown={onSelect ? e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelect(n.index) } } : undefined}>
              <rect width={W} height={H} rx={compact ? 4 : 5} />
              <text x={W / 2} y={compact ? H / 2 + 4 : 17} textAnchor="middle" className="op">{short(m.op, compact ? 13 : 18)}</text>
              {!compact && <text x={W / 2} y={31} textAnchor="middle" className="nm">{short(m.name, 22)}</text>}
              {hover === n.index && <title>{`${m.op} · ${m.name}\n${a.nodes[n.index].reason}`}</title>}
            </g>
          )
        })}
      </g>
    </svg>
  )
}

export function Legend() {
  return (
    <div className="legend graph-legend" aria-label="Legend">
      <span><i className="sw npu" />NPU</span>
      <span><i className="sw con" />NPU with constraints</span>
      <span><i className="sw cpu" />CPU fallback</span>
      <span><i className="sw removed" />Moved to post-processing</span>
      <span><i className="sw switch" />CPU/NPU switch</span>
    </div>
  )
}
