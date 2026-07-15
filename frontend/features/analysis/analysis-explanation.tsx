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
    <div className="overflow-hidden rounded-xl border border-primary/20 bg-primary/5 shadow-sm">
      <div className="h-1 bg-primary" />
      <div className="space-y-1.5 p-3">
        <p className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-primary">
          <Sparkles className="size-3.5" />
          EstateMind Insight
        </p>
        <p className="whitespace-pre-wrap break-words text-sm leading-relaxed text-foreground">
          {text}
        </p>
      </div>
    </div>
  );
}
