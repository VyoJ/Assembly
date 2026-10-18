import { cn } from '@/lib/utils'

export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger' | 'success'
export type ButtonSize = 'sm' | 'md' | 'lg' | 'icon'

const buttonVariants: Record<ButtonVariant, string> = {
  primary: 'bg-accent text-white hover:bg-accent/90 shadow-[inset_0_1px_0_rgb(255_255_255/0.15)]',
  secondary: 'bg-surface-3 text-fg border border-line-strong hover:bg-surface-3/70 hover:border-fg-subtle/40',
  ghost: 'text-fg-muted hover:text-fg hover:bg-surface-3',
  danger: 'bg-bad-soft text-bad border border-bad/30 hover:bg-bad/20',
  success: 'bg-ok text-white hover:bg-ok/90 shadow-[inset_0_1px_0_rgb(255_255_255/0.15)]',
}
const buttonSizes: Record<ButtonSize, string> = {
  sm: 'h-7 px-2.5 text-xs gap-1.5',
  md: 'h-9 px-3.5 text-sm gap-2',
  lg: 'h-10 px-4 text-sm gap-2',
  icon: 'h-8 w-8 justify-center',
}

export function buttonClasses(variant: ButtonVariant = 'secondary', size: ButtonSize = 'md', className?: string) {
  return cn(
    'inline-flex shrink-0 items-center rounded-md font-medium transition-colors select-none',
    buttonVariants[variant],
    buttonSizes[size],
    className,
  )
}
