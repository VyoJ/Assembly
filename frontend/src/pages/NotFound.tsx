import { Compass } from 'lucide-react'
import { EmptyState } from '@/components/ui/feedback'
import { ButtonLink } from '@/components/ui/primitives'

export function NotFoundPage() {
  return (
    <EmptyState
      className="py-32"
      icon={<Compass className="size-5" />}
      title="Page not found"
      action={
        <ButtonLink to="/" size="sm">
          Back to overview
        </ButtonLink>
      }
    />
  )
}
