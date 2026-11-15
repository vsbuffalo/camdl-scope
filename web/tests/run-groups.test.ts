// The fit dropdown's grouping, against the store shapes it has to tell apart.
// Run: npm test (in web/).
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { type GroupRun, runRows, selectedTitle } from '../src/lib/run-groups.ts'

function run(folder: string, method: string, seed: number, opts: Partial<GroupRun> = {}): GroupRun {
  const user_label = opts.user_label ?? null
  const config_label = 'm0_mh · mh/cpu'
  return {
    run_id: `${folder}~${method}~${seed}`, folder, method, seed,
    leaf: `${method}/seed_${seed}-0000000${seed}`, method_default: false,
    user_label, config_label, label: user_label ?? config_label, ...opts,
  }
}

const shape = (rows: ReturnType<typeof runRows>) =>
  rows.map((r) => `${'  '.repeat(r.kind === 'folder' ? 0 : r.indent)}${r.kind}:${
    r.kind === 'run' ? r.text : r.title}`)

test('two sampler configurations of one problem are two fits, not two seeds', () => {
  // The measles guide's m0_mh folder: a plain run and a run warm-started from
  // its posterior. Same TOML, different method dirs.
  const runs = [
    run('m0_mh-76aecea7', 'mh-97f0a52f', 1, { user_label: 'm0_mh_mode_a', method_default: true }),
    run('m0_mh-76aecea7', 'mh-92946094', 2, { user_label: 'm0_mh', method_default: true }),
  ]
  assert.deepEqual(shape(runRows(runs)), [
    'folder:m0_mh-76aecea7',
    '  run:m0_mh',
    '  run:m0_mh_mode_a',
  ])
  assert.equal(selectedTitle(runs[0]!, runs), 'm0_mh_mode_a')
})

test('replicate seeds nest under their configuration, which names the open seed', () => {
  const F = 'fit-4d481b66'
  const runs = [
    run(F, 'if2-6e7a546f', 1),
    ...[2, 3, 4].map((n) => run(F, 'pmmh-35a17c58', n, { method_default: n === 2 })),
    run(F, 'pmmh-11f48a0a', 1),
  ]
  assert.deepEqual(shape(runRows(runs)), [
    `folder:${F}`,
    '  run:if2-6e7a546f',
    '  run:pmmh-11f48a0a',
    '  config:pmmh-35a17c58',
    '    run:seed 2',
    '    run:seed 3',
    '    run:seed 4',
  ])
  assert.equal(selectedTitle(runs[3]!, runs), 'pmmh-35a17c58 · seed 4 · 3 seeds')
})

test('a folder with one configuration gets no header', () => {
  const S = 'fit_synthetic-02e1fdff'
  const runs = [
    run('fit_r0-84761258', 'pmmh-8d6841d6', 4, { user_label: 'r0', method_default: true }),
    ...[1, 2].map((n) =>
      run(S, 'pmmh-8d6841d6', n, {
        user_label: n === 1 ? 'synthetic' : `synthetic-s${n}`, method_default: n === 1,
      })),
  ]
  assert.deepEqual(shape(runRows(runs)), [
    'run:r0',
    'config:m0_mh · mh/cpu',
    '  run:seed 1 · synthetic',
    '  run:seed 2 · synthetic-s2',
  ])
  assert.equal(selectedTitle(runs[2]!, runs), 'synthetic-s2 · seed 2 · 2 seeds')
  assert.equal(selectedTitle(runs[0]!, runs), 'r0')
})

test('the points of a sweep read in value order, not hash order', () => {
  const pt = (h: string, tau: string) =>
    run(`m1_profile-${h}`, 'mh-0a61aa9b', 5, { user_label: `m1_tau_${tau}`, method_default: true })
  const runs = [pt('0c081779', '1'), pt('39236dfd', '0'), pt('3bb2ea7e', '4.5'),
    pt('3d91d234', '3.5'), pt('9d94fe18', '2'), pt('eaeb1c80', '3')]
  assert.deepEqual(shape(runRows(runs)), [
    'run:m1_tau_0', 'run:m1_tau_1', 'run:m1_tau_2', 'run:m1_tau_3', 'run:m1_tau_3.5', 'run:m1_tau_4.5',
  ])
})
