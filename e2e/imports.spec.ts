import { choose, expect, fact, finding, importModel, kpi, nodeCsv, sampleFile, save, tab, test } from './harness.ts'
import { handWorked, llm, onnx, small, tflite } from './files.ts'

test('hand-worked ONNX: every headline number matches the sums on paper', async ({ page }) => {
  await choose(page, 'paper.onnx', handWorked())
  await expect(page.getByText('No structural problems')).toBeVisible()
  await save(page)
  // Parameters: Conv 4*3*3*3 + 4 = 112; Gemm 10*256 + 10 = 2,570. Total 2,682; fp32 bytes 10,728 = 10.477 KB.
  await expect(kpi(page, 'Parameters')).toContainText('2.7K')
  await expect(kpi(page, 'Parameters')).toContainText('10.5 KB of weights')
  // MACs: Conv 8*8*4 outputs * 27 = 6,912; Gemm 10 outputs * 256 = 2,560. Total 9,472. FLOPs 2*9,472 + 256 + 256 = 19,456.
  await expect(fact(page, 'Compute')).toHaveText('9.5K MACs · 19.5K FLOPs')
  // Peak: the 4x8x8 fp32 map read and written by Relu, 2 * 1,024 B.
  await expect(kpi(page, 'Peak activations')).toContainText('2.0 KB')
  await expect(fact(page, 'Peak activations')).toContainText('at act')
  // 4/4 nodes on the NPU in one partition, minus 15 for a float model.
  await expect(kpi(page, 'Readiness')).toContainText('85/100')
  await expect(kpi(page, 'Partitions').locator('b')).toHaveText('1')
  // At 0.002 GB/s memory time dominates. Bytes read and written: Conv 768+1,024+448, Relu 2,048, Flatten 2,048,
  // Gemm 1,024+40+10,280. Total 17,680 B / 2e6 B/s = 8.84 ms.
  await page.locator('summary').filter({ hasText: 'Device profile' }).click()
  await page.getByLabel('Memory GB/s').fill('0.002')
  await expect(kpi(page, 'Est. latency')).toContainText('8.84 ms')
  const rows = await nodeCsv(page)
  expect(rows.map(r => [r.name, r.macs, r.output_shape, r.est_ms])).toEqual([
    ['conv', '6912', '1x4x8x8', '1.1200'], ['act', '0', '1x4x8x8', '1.0240'], ['flat', '0', '1x256', '1.0240'], ['fc', '2560', '1x10', '5.6720'],
  ])
})

test('hand-worked dynamic ONNX: mapped dims drive the estimates and the suggested pins', async ({ page }) => {
  await choose(page, 'dyn.onnx', onnx({
    inputs: [{ name: 'x', shape: ['batch', 'seq', 32] }], outputs: [{ name: 'y', shape: ['batch', 'seq', 16] }],
    inits: [{ name: 'w', dims: [32, 16] }], nodes: [{ op: 'MatMul', name: 'mm', in: ['x', 'w'], out: ['y'] }],
  }))
  await expect(page.locator('.map-grid').getByLabel('batch')).toHaveValue('1')
  await expect(page.locator('.map-grid').getByLabel('seq')).toHaveValue('128')
  await page.locator('.map-grid').getByLabel('seq').fill('10')
  await save(page)
  // 10 positions * 16 outputs * 32 inputs = 5,120 MACs. The whole graph waits on the CPU until pinned: 0 - 15 clamps to 0.
  await expect(fact(page, 'Compute')).toContainText('5.1K MACs')
  await expect(kpi(page, 'Readiness')).toContainText('0/100')
  await finding(page, 'Dynamic input shapes: batch, seq').getByRole('button', { name: 'Simulate fix' }).click()
  await expect(kpi(page, 'Readiness')).toContainText('85/100')
})

test('ONNX external data: weights are sized from dims', async ({ page }) => {
  await choose(page, 'ext.onnx', small([{ op: 'MatMul', in: ['x', 'w'], out: ['y'] }], { inits: [{ name: 'w', dims: [8, 8], external: true }] }))
  await expect(page.getByText('256 B (256 B external)')).toBeVisible()
  await save(page)
  await expect(finding(page, '256 B of weights in external data')).toBeVisible()
})

test('TFLite: NHWC convolution and weights', async ({ page }) => {
  await importModel(page, 'conv.tflite', tflite(3, [{ name: 'x', shape: [1, 8, 8, 3] }, { name: 'w', shape: [4, 3, 3, 3], weight: 432 }, { name: 'y', shape: [1, 8, 8, 4] }], { in: [0, 1], out: [2] }, { in: [0], out: [2] }))
  // 8*8*4 outputs * 27 weights each = 6,912 MACs; 108 parameters.
  await expect(fact(page, 'Compute')).toContainText('6.9K MACs')
  await expect(kpi(page, 'Parameters')).toContainText('108')
})

test('GGUF: size, KV cache and decode speed follow the documented formulas', async ({ page }) => {
  await importModel(page, 'm.gguf', llm([['llama.block_count', 16], ['llama.context_length', 8192], ['llama.embedding_length', 2048], ['llama.attention.head_count', 32], ['llama.attention.head_count_kv', 8]]))
  // 1024*1024 F16 weights = 2 MiB. Decode: 135e9 * 0.35 * 2 / 2,097,152 = 45,061.1 tokens/s.
  await expect(kpi(page, 'Parameters')).toContainText('2.0 MB of weights')
  await expect(kpi(page, 'Decode speed')).toContainText('45061.1 tok/s')
  // 2 * 16 layers * 4,096 tokens * 2,048 * 8/32 * 2 B = 134,217,728 B.
  await expect(kpi(page, 'KV cache')).toContainText('128.0 MB')
  await tab(page, 'Ops').click()
  await expect(page.getByText('No graph in this file')).toBeVisible()
})

for (const [button, label, value] of [
  ['MNIST-12 (ONNX)', 'Parameters', '6.0K'],
  ['MobileNet-v2 w8a8', 'Partitions', '1'],
  ['BlazeFace short-range', 'Parameters', '101.5K'],
  ['TinyStories 260K', 'Parameters', '292.8K'],
]) test(`bundled public model: ${button}`, async ({ page }) => {
  await page.goto('/app')
  await page.getByRole('button', { name: 'Import your model' }).click()
  await page.getByRole('button', { name: button }).click()
  await save(page)
  await expect(kpi(page, label).locator('b')).toHaveText(value)
})

test('validation points at the node with a missing input', async ({ page }) => {
  await choose(page, 'ghost.onnx', small([{ op: 'Add', name: 'bad_add', in: ['x', 'ghost'], out: ['y'] }]))
  await expect(page.locator('.error-row')).toContainText('node 0 (bad_add)')
  await expect(page.locator('.error-row')).toContainText('"ghost" has no producer')
})

test('wrong and empty files are refused with a message, and nothing is offered for saving', async ({ page }) => {
  const cases: [string, Buffer, RegExp][] = [
    ['empty.onnx', Buffer.alloc(0), /Not an ONNX model/],
    ['empty.tflite', Buffer.alloc(0), /Unrecognised file/],
    ['notes.txt', Buffer.from('hello there'), /Unrecognised file/],
    ['model.onnx.data', Buffer.from([1, 2, 3, 4, 5, 6, 7, 8, 9]), /external weights file/],
    ['model.safetensors', Buffer.from('{"a":1}'), /Export to ONNX or TFLite first/],
    ['old.gguf', Buffer.from([0x47, 0x47, 0x55, 0x46, 1, 0, 0, 0, 0, 0]), /GGUF version 1 is not supported/],
  ]
  for (const [name, bytes, message] of cases) {
    await choose(page, name, bytes)
    await expect(page.getByRole('alert'), name).toHaveText(message)
    await expect(page.getByRole('button', { name: 'Save version and analyse' }), name).toHaveCount(0)
  }
})

test('cut ONNX and TFLite files are refused at every length tried', async ({ page }) => {
  for (const file of ['mnist-12.onnx', 'face_detection_short_range.tflite']) {
    const whole = await sampleFile(file)
    for (const share of [0.999, 0.75, 0.5, 0.1]) {
      await choose(page, file, whole.subarray(0, Math.floor(whole.length * share)))
      await expect(page.getByRole('alert'), `${file} ${share}`).not.toBeEmpty()
      await expect(page.getByRole('button', { name: 'Save version and analyse' })).toHaveCount(0)
    }
  }
})

test('names from a model are shown as text, never as markup', async ({ page }) => {
  const evil = '<img src=x onerror="window.pwned=1">'
  await importModel(page, `${evil}.onnx`.replace(/[/]/g, ''), handWorked(evil))
  await tab(page, 'Graph').click()
  await page.locator('svg.graph g.node').first().click()
  await expect(page.locator('.node-panel h3')).toHaveText(evil)
  await tab(page, 'Overview').click()
  await page.getByRole('link', { name: 'Readiness report' }).click()
  await expect(page.locator('h1')).toContainText('<img')
  expect(await page.evaluate(() => (window as unknown as { pwned?: number }).pwned)).toBeUndefined()
})
