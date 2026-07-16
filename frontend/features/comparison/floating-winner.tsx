"use client";

// Floating winner card (Phase 17.2, Goal 9). A compact fixed card that keeps
// the backend comparison winner in view while the user scrolls the long
// comparison: name, verdict stars, category-win count and asking price — all
// values the page already shows (backend winner, win-tally counting, backend
// property record). It appears only once the Executive Summary has scrolled
// out of view (IntersectionObserver), sits in the bottom-right corner without
// covering the sticky headers, and can be dismissed.

import { useEffect, useState, type RefObject } from "react";
import { Trophy, X } from "lucide-react";

import { StarRow, ratingStars } from "@/features/analysis/ui/decision-summary";
import { formatCr } from "@/features/dashboard/format";
import { hasText } from "./compare-utils";
import { winTally } from "./win-tally";
import type { CompareBundle } from "./use-compare-data";

export function FloatingWinner({
  bundle,
  resolveName,
  watchRef,
}: {
  bundle: CompareBundle;
  resolveName: (id: string) => string;
  // The Executive Summary wrapper — the card shows while it is off-screen.
  watchRef: RefObject<HTMLElement | null>;
}) {
  const [summaryVisible, setSummaryVisible] = useState(true);
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    const target = watchRef.current;
    if (!target || typeof IntersectionObserver === "undefined") return;
    const observer = new IntersectionObserver(
      ([entry]) => setSummaryVisible(entry.isIntersecting),
      { threshold: 0 }
    );
    observer.observe(target);
    return () => observer.disconnect();
  }, [watchRef]);

  const { winner } = bundle.comparison;
  const index = bundle.ids.indexOf(winner.id);
  const price = bundle.details[index]?.price;
  const stars = hasText(winner.verdict)
    ? ratingStars(String(winner.verdict))
    : null;
  const tally = winTally(bundle);
  const wins = tally.entries.find((entry) => entry.id === winner.id)?.wins ?? 0;

  if (summaryVisible || dismissed) return null;

  return (
    <div
      role="status"
      aria-label="Comparison winner"
      className="fixed bottom-4 right-4 z-40 hidden w-60 rounded-xl border border-primary/30 bg-card p-3 shadow-lg animate-in fade-in slide-in-from-bottom-2 duration-300 sm:block"
    >
      <div className="flex items-start justify-between gap-2">
        <p className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-primary">
          <Trophy className="size-3.5" />
          Winner
        </p>
        <button
          type="button"
          aria-label="Dismiss winner card"
          onClick={() => setDismissed(true)}
          className="text-muted-foreground/60 transition-colors hover:text-muted-foreground"
        >
          <X className="size-3.5" />
        </button>
      </div>

      <p className="mt-1 truncate font-heading text-sm font-bold">
        {resolveName(winner.id)}
      </p>

      <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1">
        {typeof stars === "number" && <StarRow count={stars} />}
        {wins > 0 && (
          <span className="text-xs font-medium text-muted-foreground">
            {wins} Category {wins === 1 ? "Win" : "Wins"}
          </span>
        )}
        {hasText(price) && (
          <span className="font-heading text-sm font-bold tabular-nums text-primary">
            {formatCr(price as number | string)}
          </span>
        )}
      </div>
    </div>
  );
}
