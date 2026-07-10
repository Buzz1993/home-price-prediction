"use client";

// AI Explanation card (Phase 15.4). Renders Claude's natural-language summary of
// what a backend result means. Shown above the existing result renderers, it
// only explains the backend result — it never predicts, values, scores, ranks
// or recomputes anything. Reused across AI features (Analysis, Comparison, …);
// `unavailableMessage` tailors the graceful fallback wording per feature.
//
// Claude is optional: when the backend omits the explanation (Claude
// unavailable / failed), a graceful message is shown and the backend result
// still renders normally.

import { Sparkles } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export function AnalysisExplanation({
  explanation,
  unavailableMessage = "Property analysis is available, but the AI explanation is temporarily unavailable.",
}: {
  explanation?: string | null;
  unavailableMessage?: string;
}) {
  const text = explanation?.trim();

  return (
    <Card className="gap-3 border-primary/20 bg-primary/5 py-4">
      <CardHeader className="px-4">
        <CardTitle className="flex items-center gap-2 text-sm text-primary">
          <Sparkles className="size-4" />
          AI explanation
        </CardTitle>
      </CardHeader>
      <CardContent className="px-4 text-sm">
        {text ? (
          <p className="whitespace-pre-wrap break-words leading-relaxed text-foreground">
            {text}
          </p>
        ) : (
          <p className="text-muted-foreground">{unavailableMessage}</p>
        )}
      </CardContent>
    </Card>
  );
}
