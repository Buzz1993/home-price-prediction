"use client";

// AI Explanation card (Phase 15.4). Renders Claude's natural-language summary of
// what a backend result means. Shown above the existing result renderers, it
// only explains the backend result — it never predicts, values, scores, ranks
// or recomputes anything. Reused across AI features (Analysis, Comparison, …);
// `unavailableMessage` tailors the graceful fallback wording per feature.
//
// Claude is optional: when the backend omits the explanation (Claude
// unavailable / failed), an elegant compact empty state is shown and the
// backend result still renders normally. Premium presentation (Phase 15.18):
// the card carries the report document's brand accent bar and uppercase
// eyebrow title, with compact padding.

import { Sparkles } from "lucide-react";

export function AnalysisExplanation({
  explanation,
  unavailableMessage = "Property analysis is available, but the AI explanation is temporarily unavailable.",
}: {
  explanation?: string | null;
  unavailableMessage?: string;
}) {
  const text = explanation?.trim();

  // Elegant compact empty state instead of a large notification banner.
  if (!text) {
    return (
      <p className="flex items-center gap-1.5 rounded-lg border border-dashed px-3 py-1.5 text-xs text-muted-foreground">
        <Sparkles className="size-3.5 shrink-0 text-primary/50" />
        {unavailableMessage}
      </p>
    );
  }

  return (
    <div className="overflow-hidden rounded-xl border border-primary/20 bg-gradient-to-br from-primary/10 via-primary/5 to-transparent shadow-float">
      <div className="bg-brand-gradient h-1" />
      <div className="flex gap-3 p-4">
        <span className="bg-brand-gradient shadow-brand-glow flex size-8 shrink-0 items-center justify-center rounded-lg text-primary-foreground">
          <Sparkles className="size-4" />
        </span>
        <div className="min-w-0 space-y-1">
          <p className="text-xs font-semibold uppercase tracking-wide text-primary">
            EstateMind Insight
          </p>
          <p className="whitespace-pre-wrap break-words text-sm leading-relaxed text-foreground">
            {text}
          </p>
        </div>
      </div>
    </div>
  );
}
