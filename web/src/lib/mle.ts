import type { MleResponse } from '@/api/client'

/** Whether the run that wrote this MLE fit is still going, from its status. */
export function isLive(status: string | undefined): boolean {
  return status === 'running' || status === 'warming'
}

/** `3 of 20 starts finished`, or `3 starts finished` when camdl did not record
 *  how many the run was configured with. */
export function startsFinished(data: MleResponse): string {
  const k = data.n_restarts
  const noun = (n: number) => (n === 1 ? 'start' : 'starts')
  return data.n_expected != null
    ? `${k} of ${data.n_expected} ${noun(data.n_expected)} finished`
    : `${k} ${noun(k)} finished`
}
