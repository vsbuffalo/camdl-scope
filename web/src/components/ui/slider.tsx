import type { ComponentProps } from 'react'
import * as SliderPrimitive from '@radix-ui/react-slider'
import { cn } from '@/lib/utils'

export function Slider({
  className,
  ...props
}: ComponentProps<typeof SliderPrimitive.Root>) {
  // One thumb per value — a single-value slider (warm-up) gets one; a
  // [from, to] range slider (sim window) gets two.
  const n = (props.value ?? props.defaultValue)?.length ?? 1
  return (
    <SliderPrimitive.Root
      className={cn(
        'relative flex w-full touch-none select-none items-center py-1',
        className,
      )}
      {...props}
    >
      <SliderPrimitive.Track className="relative h-1 w-full grow overflow-hidden rounded-none bg-neutral-200">
        <SliderPrimitive.Range className="absolute h-full bg-neutral-900" />
      </SliderPrimitive.Track>
      {Array.from({ length: n }, (_, i) => (
        <SliderPrimitive.Thumb
          key={i}
          className={cn(
            'block h-3.5 w-2 rounded-none border border-neutral-400 bg-neutral-900',
            'transition-colors hover:border-neutral-600',
            'focus:outline-none focus-visible:ring-1 focus-visible:ring-neutral-900/40',
          )}
          aria-label={n > 1 ? `window bound ${i + 1}` : 'slider'}
        />
      ))}
    </SliderPrimitive.Root>
  )
}
