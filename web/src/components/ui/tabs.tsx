import type { ComponentProps } from 'react'
import * as TabsPrimitive from '@radix-ui/react-tabs'
import { cn } from '@/lib/utils'

export const Tabs = TabsPrimitive.Root

export function TabsList({
  className,
  ...props
}: ComponentProps<typeof TabsPrimitive.List>) {
  return (
    <TabsPrimitive.List
      className={cn(
        'flex w-full items-center gap-x-5 gap-y-1 border-b border-neutral-200',
        // On a phone the strip is wider than the screen. It used to scroll with
        // the scrollbar explicitly hidden, which is the worst pair: tabs off
        // screen and nothing saying so. Wrap instead — every tab visible, no
        // gesture to discover, at the cost of one extra line. Wide screens fit
        // in one row, where the scroll fallback stays for a long tab set.
        'flex-wrap sm:flex-nowrap',
        'sm:overflow-x-auto sm:[scrollbar-width:none] sm:[&::-webkit-scrollbar]:hidden',
        className,
      )}
      {...props}
    />
  )
}

export function TabsTrigger({
  className,
  ...props
}: ComponentProps<typeof TabsPrimitive.Trigger>) {
  return (
    <TabsPrimitive.Trigger
      className={cn(
        'shrink-0 border-b-2 border-transparent pb-2 -mb-px text-sm text-neutral-500',
        'transition-colors hover:text-neutral-800',
        'focus:outline-none focus-visible:text-neutral-900',
        'disabled:pointer-events-none disabled:opacity-50',
        'data-[state=active]:border-neutral-900 data-[state=active]:text-neutral-900 data-[state=active]:font-medium',
        className,
      )}
      {...props}
    />
  )
}

export function TabsContent({
  className,
  ...props
}: ComponentProps<typeof TabsPrimitive.Content>) {
  return (
    <TabsPrimitive.Content
      className={cn('mt-4 focus:outline-none', className)}
      {...props}
    />
  )
}
