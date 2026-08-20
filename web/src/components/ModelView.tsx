import { Fragment, useMemo, type ReactNode } from 'react'
import katex from 'katex'
import 'katex/dist/katex.min.css'
import type { ModelRender } from '@/api/client'
import { Description } from '@/components/Description'
import { Card } from '@/components/ui/card'
import { normalizeTex } from '@/lib/tex'

/**
 * Render one KaTeX-safe string. `throwOnError: false` renders a malformed
 * expression inline (in red) rather than crashing the whole view — the strings
 * come from an upstream contract, so a bad leaf shouldn't take down the page.
 * `display` switches to block (centred) math.
 */
function Tex({ tex, display = false }: { tex: string; display?: boolean }) {
  const html = useMemo(
    () =>
      katex.renderToString(normalizeTex(tex), {
        throwOnError: false,
        displayMode: display,
        output: 'html',
      }),
    [tex, display],
  )
  return (
    <span
      className={display ? 'block' : 'inline-block'}
      // KaTeX output is sanitized markup for a trusted, model-derived string.
      dangerouslySetInnerHTML={{ __html: html }}
    />
  )
}

/** A labelled section within the model card. */
function Section({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="border-b border-neutral-100 px-3 py-2.5 last:border-b-0">
      <div className="mb-1.5 text-[10px] font-medium uppercase tracking-wider text-neutral-400">
        {label}
      </div>
      {children}
    </div>
  )
}

/**
 * The structured model, rendered from `model.render.json`: a title from
 * `model`+`mode`, dimension chips, a parameter glossary (symbol + description), a
 * reaction table (reactants → products, rate), and the ODE list — all math via
 * KaTeX client-side. Sections that the contract left empty are omitted.
 */
export function ModelView({ data }: { data: ModelRender }) {
  return (
    <Card className="overflow-hidden">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 border-b border-neutral-200 px-3 py-2">
        <span className="text-sm font-semibold text-neutral-900">{data.model}</span>
        <span className="font-mono text-[11px] text-neutral-400">{data.mode}</span>
        {data.dimensions.length > 0 && (
          <span className="ml-auto flex flex-wrap gap-1">
            {data.dimensions.map((d) => (
              <span
                key={d.name}
                className="rounded-sm bg-neutral-100 px-1.5 py-0.5 font-mono text-[10px] text-neutral-500"
                title={d.levels.join(', ')}
              >
                {d.name}: {d.levels.join(' · ')}
              </span>
            ))}
          </span>
        )}
      </div>

      {data.states.length > 0 && (
        <Section label="states">
          {/* Raw compartment identifiers (e.g. `I_inf`), not KaTeX — render as
              plain mono so an underscore isn't read as a subscript. */}
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-sm text-neutral-800">
            {data.states.map((s) => (
              <span key={s}>{s}</span>
            ))}
          </div>
        </Section>
      )}

      {data.parameters.length > 0 && (
        <Section label="parameters">
          <dl className="grid grid-cols-[auto_1fr] items-baseline gap-x-4 gap-y-1.5">
            {data.parameters.map((p) => (
              <Fragment key={p.name}>
                <dt className="text-right text-neutral-900">
                  <Tex tex={p.symbol} />
                </dt>
                <dd className="min-w-0 text-xs text-neutral-600">
                  <span className="font-mono text-[10px] text-neutral-400">{p.name}</span>
                  {p.description && (
                    <Description
                      text={p.description}
                      className="text-xs text-neutral-600"
                    />
                  )}
                </dd>
              </Fragment>
            ))}
          </dl>
        </Section>
      )}

      {data.definitions.length > 0 && (
        <Section label="definitions">
          <div className="space-y-1 overflow-x-auto">
            {data.definitions.map((d) => (
              <Tex key={d.name} tex={d.tex} display />
            ))}
          </div>
        </Section>
      )}

      {data.transitions.length > 0 && (
        <Section label="reactions">
          <div className="overflow-x-auto">
            <table className="text-sm">
              <thead>
                <tr className="text-[10px] font-medium uppercase tracking-wider text-neutral-400">
                  <th className="py-1 pr-3 text-left" />
                  <th className="py-1 pr-4 text-left">reaction</th>
                  <th className="py-1 text-left">reaction rate</th>
                </tr>
              </thead>
              <tbody>
                {data.transitions.map((t) => (
                  <tr key={t.name} className="align-baseline">
                    <td className="py-1 pr-3 font-mono text-[10px] text-neutral-400">
                      {t.name}
                    </td>
                    <td className="whitespace-nowrap py-1 pr-4 text-neutral-800">
                      <Tex tex={t.reactants} />
                      <span className="mx-1.5 text-neutral-400">→</span>
                      <Tex tex={t.products} />
                    </td>
                    <td className="py-1 text-neutral-700">
                      <Tex tex={t.rate} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-2 text-[10px] text-neutral-400">
            reaction rate = rate constant × reactants — the flux (ODE) / propensity
            (stochastic) at which the reaction fires.
          </p>
        </Section>
      )}

      {data.dynamics.length > 0 && (
        <Section label="dynamics">
          <div className="space-y-2 overflow-x-auto">
            {data.dynamics.map((d) => (
              <Tex key={d.state} tex={d.tex} display />
            ))}
          </div>
        </Section>
      )}
    </Card>
  )
}
