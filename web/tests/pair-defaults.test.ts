import assert from 'node:assert/strict'
import { test } from 'node:test'
import { defaultPairObjectives } from '../src/lib/pair-defaults.ts'

test('a complete-data fit pairs against obs_ll, not the path-dominated log_posterior', () => {
  assert.deepEqual(
    defaultPairObjectives(['log_posterior', 'log_complete_data_ll', 'obs_ll', 'transition_ll']),
    ['obs_ll'],
  )
})

test('a fit without the split pairs against log_posterior', () => {
  assert.deepEqual(defaultPairObjectives(['log_posterior', 'log_likelihood']), ['log_posterior'])
  assert.deepEqual(defaultPairObjectives([]), [])
})
