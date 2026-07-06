import { useEffect, useRef, useState, type ReactNode } from 'react'
import type { SourceFile } from '@/api/client'
import { useSource } from '@/api/queries'
import { ForestSkeleton, MutedNotice } from '@/components/States'
import { Card } from '@/components/ui/card'

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
export function SourceTab({ runId }: { runId: string }) {
  const { data, isPending, isError } = useSource(runId)
  useHighlightCss(data?.highlight_css)

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
    </div>
  )
}
