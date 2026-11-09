// Deep-link resolution against a store whose synthetic fit holds six seed
// runs, shaped like the camdl docs' first-fit store. Run: npm test (in web/).
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { type RunRef, resolveRunToken } from '../src/lib/deeplink.ts'

const SYN = 'fit_synthetic-02e1fdff'
const seeds: RunRef[] = [1, 2, 3, 4, 5, 6].map((n) => ({
  run_id: `${SYN}~seed_${n}-0${n}aaaaaa`,
  folder: SYN,
  folder_default: n === 1,
  leaf: `pmmh-83841c1b/seed_${n}-0${n}aaaaaa`,
  leaf_run_id: `${n}${n}${n}${n}`.padEnd(64, 'c'),
  label: n === 1 ? 'synthetic' : `synthetic-s${n}`,
}))
const r0: RunRef = {
  run_id: 'fit_r0-84761258', folder: 'fit_r0-84761258', folder_default: true,
  leaf: 'pmmh-83841c1b/seed_4-8b49cfaa', leaf_run_id: 'f1c0c39d577c'.padEnd(64, '0'),
  label: 'r0',
}
const runs = [r0, ...seeds]
const ok = (runId: string) => ({ kind: 'ok', runId })

test('a leaf path from camdl opens that seed, not the folder default', () => {
  assert.deepEqual(
    resolveRunToken(`results/fits/${SYN}/pmmh-83841c1b/seed_6-06aaaaaa`, runs),
    ok(seeds[5]!.run_id),
  )
})

test('a folder name or folder path opens the default run', () => {
  assert.deepEqual(resolveRunToken(SYN, runs), ok(seeds[0]!.run_id))
  assert.deepEqual(resolveRunToken(`results/fits/${SYN}`, runs), ok(seeds[0]!.run_id))
})

test('a seed run id, a label and a camdl run_id each name one run', () => {
  assert.deepEqual(resolveRunToken(seeds[2]!.run_id, runs), ok(seeds[2]!.run_id))
  assert.deepEqual(resolveRunToken('@synthetic-s4', runs), ok(seeds[3]!.run_id))
  assert.deepEqual(resolveRunToken('5555cc', runs), ok(seeds[4]!.run_id))
  assert.deepEqual(resolveRunToken('84761258', runs), ok(r0.run_id))
})

test('an ambiguous token is refused, never guessed', () => {
  assert.equal(resolveRunToken('fit_', runs).kind, 'ambiguous')
  assert.equal(resolveRunToken('nothing-like-this', runs).kind, 'none')
})
