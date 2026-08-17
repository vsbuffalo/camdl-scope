import { useEffect, useRef, useState, type ReactNode } from 'react'
import type { SourceFile } from '@/api/client'
import { useModelGraph, useModelRender, useRun, useSource } from '@/api/queries'
import { ForestSkeleton, MutedNotice } from '@/components/States'
import { ModelView } from '@/components/ModelView'
import { FlowDiagram } from '@/components/FlowDiagram'
import { Segmented } from '@/components/Segmented'
import { Card } from '@/components/ui/card'
import { loadJson, saveJson } from '@/lib/persist'

const HIGHLIGHT_STYLE_ID = 'camdl-highlight-css'

/**
 * Inject the Pygments token stylesheet ONCE into a single `<style>` in the head,
 * keyed by a stable id so remounting the tab (or switching runs) reuses the same
 * element instead of stacking duplicates. The token classes live under `.codehl`
 * in the highlighted HTML; we render that CSS verbatim — keeping the spans
 * coloured matters more than retoning them, so we don't rewrite it.
 */
function useHighlightCss(css: string | undefined) {
  useEffect(() => {
    if (!css) return
    let el = document.getElementById(
      HIGHLIGHT_STYLE_ID,
    ) as HTMLStyleElement | null
    if (!el) {
      el = document.createElement('style')
      el.id = HIGHLIGHT_STYLE_ID
      document.head.appendChild(el)
    }
    if (el.textContent !== css) el.textContent = css
  }, [css])
}

/**
 * Copy `text` to the clipboard. The dashboard is served over plain http on
 * LAN/Tailscale — NOT a secure context — so `navigator.clipboard` is often
 * unavailable. Use it only when the context is secure, else fall back to a
 * throwaway off-screen `<textarea>` + `execCommand('copy')`.
 */
async function copyText(text: string): Promise<void> {
  if (window.isSecureContext && navigator.clipboard) {
    await navigator.clipboard.writeText(text)
    return
  }
  const ta = document.createElement('textarea')
  ta.value = text
  ta.setAttribute('readonly', '')
  ta.style.position = 'fixed'
  ta.style.top = '0'
  ta.style.left = '0'
  ta.style.opacity = '0'
  ta.style.pointerEvents = 'none'
  document.body.appendChild(ta)
  ta.focus()
  ta.select()
  try {
    document.execCommand('copy')
  } finally {
    document.body.removeChild(ta)
  }
}

/** Flat terminal copy button — flips to `copied ✓` for ~1.2s. */
function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)
  const timer = useRef<number | null>(null)

  useEffect(
    () => () => {
      if (timer.current != null) window.clearTimeout(timer.current)
    },
    [],
  )

  const onCopy = async () => {
    try {
      await copyText(text)
      setCopied(true)
      if (timer.current != null) window.clearTimeout(timer.current)
      timer.current = window.setTimeout(() => setCopied(false), 1200)
    } catch {
      // Best-effort: if both paths fail, leave the label unchanged.
    }
  }

  return (
    <button
      type="button"
      onClick={onCopy}
      className="shrink-0 rounded-sm border border-neutral-200 px-2 py-0.5 font-mono text-[11px] text-neutral-500 transition-colors hover:border-neutral-300 hover:text-neutral-800"
    >
      {copied ? 'copied ✓' : 'copy'}
    </button>
  )
}

/** One source artifact: a header row (title · subline · copy) over the code. */
function SourcePanel({
  title,
  subline,
  file,
}: {
  title: string
  subline: ReactNode
  file: SourceFile
}) {
  return (
    <Card className="overflow-hidden">
      <div className="flex items-start justify-between gap-3 border-b border-neutral-200 px-3 py-2">
        <div className="min-w-0">
          <div className="font-mono text-xs text-neutral-800">{title}</div>
          <div className="mt-0.5 truncate font-mono text-[10px] text-neutral-400">
            {subline}
          </div>
        </div>
        {file.present && <CopyButton text={file.text} />}
      </div>

      {file.present ? (
        <div
          className="codehl overflow-x-auto px-3 py-2.5 font-mono text-xs leading-relaxed [&_pre]:m-0 [&_pre]:whitespace-pre"
          dangerouslySetInnerHTML={{ __html: file.html }}
        />
      ) : (
        <MutedNotice
          bordered={false}
          title="model source not found"
          detail={
            file.path
              ? `Couldn't read the model at ${file.path} — this fit predates source archiving and the file has moved since.`
              : 'No model path was recorded for this fit.'
          }
        />
      )}
    </Card>
  )
}

function basename(path: string | null | undefined): string | null {
  if (!path) return null
  const parts = path.split('/')
  return parts[parts.length - 1] || path
}

/** Provenance blurb for a source panel's subline, keyed by where the bytes
 * came from: the self-contained fit run leaf, or a live read of the checkout. */
function originNote(origin: SourceFile['origin']): string {
  if (origin === 'leaf') return 'archived in the run leaf'
  if (origin === 'live') return 'read live from its recorded path'
  return ''
}

/**
 * The fit's sources, stacked: the `.camdl` model on top (the copy archived in
 * the run leaf when present, else read live from its recorded path) and the
 * `fit.toml` below (always archived in the leaf). Pygments-highlighted HTML is
 * rendered verbatim; the token stylesheet is injected once. A comfortable
 * reading width — this is text, not a figure.
 */
/** The rendered-model view: structured math from `model.render.json`. Its own
 *  loading/empty states so a slow or absent render never blocks the raw source. */
function RenderedModel({ runId }: { runId: string }) {
  const { data, isPending, isError } = useModelRender(runId, true)
  if (isPending)
    return (
      <Card className="overflow-hidden">
        <ForestSkeleton rows={3} />
      </Card>
    )
  if (isError || !data)
    return (
      <MutedNotice
        bordered
        title="No rendered model"
        detail="This run has no model.render.json — see the raw source instead."
      />
    )
  return <ModelView data={data} />
}

/** The flow-diagram view: the compartmental graph from `model.graph.json`. Its
 *  own loading/empty states so a slow or absent graph never blocks the source. */
function RenderedGraph({ runId }: { runId: string }) {
  const { data, isPending, isError } = useModelGraph(runId, true)
  if (isPending)
    return (
      <Card className="overflow-hidden">
        <ForestSkeleton rows={3} />
      </Card>
    )
  if (isError || !data)
    return (
      <MutedNotice
        bordered
        title="No flow diagram"
        detail="This run has no model.graph.json — see the equations or raw source instead."
      />
    )
  return <FlowDiagram graph={data} />
}

export function SourceTab({ runId }: { runId: string }) {
  const { data, isPending, isError } = useSource(runId)
  useHighlightCss(data?.highlight_css)

  // Three lenses on the same model, offered only when their artifact is present.
  // Precedence (most visual first): diagram → equations → raw source. Source is
  // always the floor. The reader can drop down; a run without the richer
  // artifacts opens straight on source with no toggle.
  const run = useRun(runId)
  const hasGraph = run.data?.has_model_graph ?? false
  const hasRender = run.data?.has_model_render ?? false
  const options = [
    ...(hasGraph ? ['diagram'] : []),
    ...(hasRender ? ['equations'] : []),
    'source',
  ]
  const [view, setView] = useState<string>(() => loadJson('explore:model-view', 'diagram'))
  const effectiveView = options.includes(view) ? view : options[0]
  const onView = (v: string) => {
    saveJson('explore:model-view', v)
    setView(v)
  }

  if (isPending) {
    return (
      <div className="max-w-4xl">
        <Card className="overflow-hidden">
          <ForestSkeleton rows={4} />
        </Card>
      </div>
    )
  }

  if (isError || !data) {
    return (
      <div className="max-w-4xl">
        <MutedNotice
          title="Couldn't load the source"
          detail="The backend returned an error fetching this fit's source files."
        />
      </div>
    )
  }

  const modelBase = basename(data.model.path)

  return (
    <div className="max-w-4xl space-y-4">
      {options.length > 1 && (
        <div className="px-1">
          <Segmented
            label="view"
            options={options}
            value={effectiveView}
            onChange={onView}
          />
        </div>
      )}

      {effectiveView === 'diagram' && <RenderedGraph runId={runId} />}

      {effectiveView === 'equations' && <RenderedModel runId={runId} />}

      {effectiveView === 'source' && (
        <>
      <SourcePanel
        title={modelBase ? `model · ${modelBase}` : 'model'}
        subline={
          <>
            {data.model.path ?? 'no recorded path'}
            <span className="text-neutral-300">
              {' '}
              · {originNote(data.model.origin)}
            </span>
          </>
        }
        file={data.model}
      />

      <SourcePanel
        title="fit.toml"
        subline={
          <>
            {basename(data.fit_toml.path) ?? 'fit.toml'}
            <span className="text-neutral-300">
              {' '}
              · {originNote(data.fit_toml.origin)}
            </span>
          </>
        }
        file={data.fit_toml}
      />
        </>
      )}
    </div>
  )
}
