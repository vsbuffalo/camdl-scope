import assert from 'node:assert/strict'
import { test } from 'node:test'
import { chainLoglikCells, reasonText } from '../src/lib/chain-loglik.ts'

test('an assessed chain shows camdl numbers and camdl flag as written', () => {
  const c = chainLoglikCells({
    chain: 2, mean: '-44.75', status: 'assessed', diff: '-9.10', z: '5.20',
    cutoff: '3.66', flagged: true,
  })
  assert.deepEqual(c, {
    mean: '-44.75', diff: '-9.10', z: '5.20', cutoff: '3.66', flag: 'flagged', note: null,
  })
})

test('a chain camdl did not flag reads as not flagged, never as certified', () => {
  const c = chainLoglikCells({ chain: 0, mean: '-41.59', status: 'assessed', flagged: false })
  assert.equal(c.flag, 'not flagged')
  assert.equal(c.z, '—')
})

test('a not-assessed chain carries camdl reason, with the ESS when camdl gave one', () => {
  const c = chainLoglikCells({
    chain: 1, mean: 'NaN', status: 'not_assessed', reason: 'too_few_effective_draws', ess: '4.20',
  })
  assert.equal(c.flag, 'not assessed')
  assert.equal(c.mean, 'NaN')
  assert.equal(c.note, 'too few effective draws (ESS 4.20)')
})

test('an unknown reason code is shown, not dropped', () => {
  assert.equal(reasonText({ reason: 'new_reason_code' }), 'new reason code')
  assert.equal(reasonText({}), 'no reason recorded')
})
