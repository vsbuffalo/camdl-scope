import { useState } from 'react'
import { GlobalHeader, type Workspace } from '@/components/GlobalHeader'
import { ExploreWorkspace } from '@/components/ExploreWorkspace'
import { CompareWorkspace } from '@/components/CompareWorkspace'
import { ProfileWorkspace } from '@/components/ProfileWorkspace'
import { SimWorkspace } from '@/components/SimWorkspace'
import { loadJson, saveJson } from '@/lib/persist'
import { DEMO } from '@/lib/demo'

/**
 * App shell: the persistent header with the workspace nav, and the active
 * workspace below it. Each workspace owns its own data, selection, and inner
 * tabs — App only switches between them. New top-level modes plug in here.
 */
function App() {
  const [workspace, setWorkspace] = useState<Workspace>(() =>
    loadJson<Workspace>('workspace', 'explore'),
  )
  const onWorkspace = (w: Workspace) => {
    saveJson('workspace', w)
    setWorkspace(w)
  }
  // Compare is hidden in the demo; never land on it (e.g. from a stale saved
  // choice), and keep the nav highlight consistent.
  const active = DEMO && workspace === 'compare' ? 'explore' : workspace

  return (
    <div className="min-h-screen bg-white text-neutral-900 antialiased">
      <GlobalHeader workspace={active} onWorkspace={onWorkspace} />

      <main className="mx-auto w-full max-w-6xl px-4 py-4 sm:px-6 sm:py-6">
        {active === 'explore' && <ExploreWorkspace />}
        {active === 'compare' && <CompareWorkspace />}
        {active === 'profile' && <ProfileWorkspace />}
        {active === 'sims' && <SimWorkspace />}
      </main>
    </div>
  )
}

export default App
