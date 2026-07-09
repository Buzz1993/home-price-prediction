import type { LucideIcon } from "lucide-react";

import { cn } from "@/lib/utils";

// Single reusable step card for the "How EstateMind Works" workflow section.
// Presentational only — reused for every step so the timeline never duplicates
// markup.
export type WorkflowStepProps = {
  step: number;
  icon: LucideIcon;
  title: string;
  description: string;
  className?: string;
};

export function WorkflowStep({
  step,
  icon: Icon,
  title,
  description,
  className,
}: WorkflowStepProps) {
  return (
    <div
      className={cn(
        "flex h-full flex-col items-center gap-3 rounded-xl border bg-card p-6 text-center shadow-sm transition-shadow hover:shadow-md",
        className
      )}
    >
      <span className="flex size-12 items-center justify-center rounded-full bg-primary/10 text-primary">
        <Icon className="size-6" />
      </span>
      <span className="font-heading text-xs font-semibold uppercase tracking-wide text-primary">
        Step {step}
      </span>
      <h3 className="font-heading text-base font-semibold">{title}</h3>
      <p className="text-sm text-muted-foreground text-pretty">{description}</p>
    </div>
  );
}
