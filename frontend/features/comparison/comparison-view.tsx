"use client";

// Comparison result view, extracted from the former Property Comparison page
// so the unified AI Analysis workspace can render the same result without
// duplicating components. Renders the unchanged backend comparison: the AI
// explanation card, the Property Score Cards and the AI Recommendation &
// Ranking table. No comparison logic lives here — the backend scores and
// ranks via POST /analysis/comparison.

import { AnalysisExplanation } from "@/features/analysis/analysis-explanation";
import { SectionLabel } from "@/features/analysis/ui/analysis-ui";
import { ComparisonResult } from "@/features/dashboard/comparison-result";
import { PropertyScoreCards } from "./property-score-cards";
import type { ComparisonResult as ComparisonResultData } from "@/types/dashboard";

export function ComparisonView({
  content,
  explanation,
}: {
  content: ComparisonResultData;
  explanation?: string | null;
}) {
  return (
    <div className="space-y-4">
      {/* Claude explains the backend comparison; the Score Cards and
          Comparison Table below still render the unchanged backend result
          even if the explanation is absent. */}
      <AnalysisExplanation
        explanation={explanation}
        unavailableMessage="Property comparison is available, but the AI explanation is temporarily unavailable."
      />

      <div className="space-y-2">
        <SectionLabel>Property Score Cards</SectionLabel>
        <PropertyScoreCards
          rankings={content.rankings}
          winnerId={content.winner.id}
        />
      </div>

      <div className="space-y-2">
        <SectionLabel>AI Recommendation &amp; Ranking</SectionLabel>
        <ComparisonResult data={content} />
      </div>
    </div>
  );
}
