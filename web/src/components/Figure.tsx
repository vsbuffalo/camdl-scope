import { useEffect, useRef, useState } from 'react'
import { PlotDownloadButton } from '@/components/PlotDownloadButton'

/**
 * A self-measuring plot wrapper: a ref'd figure box that re-renders on width
 * change (ResizeObserver) plus a hover PNG-download button. The `render`
 * callback receives the live element and measured width and draws into it (via
 * `el.replaceChildren(node)`); `deps` re-runs it like a `useEffect` dependency
 * list. Shared by every Observable Plot panel so measuring + export live in one
 * place.
 */
export function Figure({
  name,
  aria,
  render,
  deps,
}: {
  name: string
  aria: string
  render: (el: HTMLDivElement, width: number) => void
  deps: unknown[]
}) {
  const ref = useRef<HTMLDivElement>(null)
  const figRef = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(0)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    setWidth(Math.round(el.getBoundingClientRect().width))
    const ro = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width ?? 0
      if (w > 0) setWidth(Math.round(w))
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  useEffect(() => {
    const el = ref.current
    if (el && width > 0) render(el, width)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [width, ...deps])
  return (
    <div className="group/fig relative border-t border-neutral-100 px-3 py-3">
      <div ref={figRef} className="bg-white">
        <div ref={ref} className="w-full min-w-0 overflow-x-auto" role="img" aria-label={aria} />
      </div>
      <PlotDownloadButton targetRef={figRef} name={name} />
    </div>
  )
}
