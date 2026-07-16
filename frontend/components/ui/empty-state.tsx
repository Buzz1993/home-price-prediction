// Reusable empty state (Phase 12.4). One shared panel for every "there is
// nothing to show yet" case: an icon, a friendly message and an optional CTA.
// Replaces the ad-hoc centered `<p>` / dashed-border blocks that each API-driven
// section used to hand-roll, so empty UI is consistent across the app. It is not
// for loading or error states — those have their own components.

import type { LucideIcon } from "lucide-react";

import { cn } from "@/lib/utils";

type EmptyStateProps = {
  // Icon that hints at what is missing (search results, saved list, tray, …).
  icon: LucideIcon;
  // Optional short headline. When omitted, only the description is shown.
  title?: string;
  // One-line, friendly explanation of why it's empty and what to do next.
  description: string;
  // Optional call to action (e.g. a Button/Link) shown below the description.
  action?: React.ReactNode;
  className?: string;
};

export function EmptyState({
  icon: Icon,
  title,
  description,
  action,
  className,
}: EmptyStateProps) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center gap-3 text-center",
        className
      )}
    >
      {/* Icon in a soft green gradient ring — friendly, premium focal point. */}
      <span className="flex size-14 items-center justify-center rounded-2xl bg-gradient-to-br from-primary/10 via-accent to-transparent ring-1 ring-primary/10">
        <Icon className="size-6 text-primary/70" />
      </span>
      {title && <p className="text-sm font-semibold">{title}</p>}
      <p className="max-w-sm text-sm text-muted-foreground">{description}</p>
      {action && <div className="pt-1">{action}</div>}
    </div>
  );
}
