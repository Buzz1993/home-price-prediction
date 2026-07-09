"use client";

// AI Search Explanation (Phase 15.3). Renders Claude's natural-language summary
// of WHY the backend recommended the returned properties. Shown above the
// Property Cards, it only explains the backend search result — it never
// searches, ranks or changes which properties appear.
//
// Claude is optional: when the backend omits the explanation (Claude
// unavailable / failed), a graceful message is shown and the search results
// still render normally.

import { Sparkles } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export function SearchExplanation({ explanation }: { explanation?: string }) {
  const text = explanation?.trim();

  return (
    <Card className="gap-3 border-primary/20 bg-primary/5 py-4">
      <CardHeader className="px-4">
        <CardTitle className="flex items-center gap-2 text-sm text-primary">
          <Sparkles className="size-4" />
          Why these properties
        </CardTitle>
      </CardHeader>
      <CardContent className="px-4 text-sm">
        {text ? (
          <p className="whitespace-pre-wrap break-words leading-relaxed text-foreground">
            {text}
          </p>
        ) : (
          <p className="text-muted-foreground">
            Property results are available, but the AI explanation is
            temporarily unavailable.
          </p>
        )}
      </CardContent>
    </Card>
  );
}
