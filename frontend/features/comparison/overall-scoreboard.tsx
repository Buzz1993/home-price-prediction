"use client";

// Overall Winner Scoreboard (Phase 17.1, Goal 1) — one glance answers "who
// wins overall?". Rendered right after the Executive Winner Summary: each
// compared property gets a medal, its win count and an animated proportional
// bar, plus chips naming the categories it won. Wins are counted from the
// EXISTING Phase 17.0 category winners (win-tally) — nothing is re-scored.

import { Award, Medal, Trophy } from "lucide-react";

import { cn } from "@/lib/utils";
import { winTally } from "./win-tally";
import type { CompareBundle } from "./use-compare-data";

// Medal per rank position: gold trophy, silver medal, bronze medal — each in
// its own soft-tinted chip.
const MEDALS = [
  { icon: Trophy, className: "text-amber-500", chip: "bg-amber-100/80" },
  { icon: Medal, className: "text-muted-foreground", chip: "bg-muted" },
  { icon: Medal, className: "text-amber-700", chip: "bg-orange-100/70" },
];

export function OverallScoreboard({
  bundle,
  resolveName,
}: {
  bundle: CompareBundle;
  resolveName: (id: string) => string;
}) {
  const tally = winTally(bundle);
  if (tally.totalCategories === 0) return null;

  return (
    <section className="card-accent-top overflow-hidden rounded-xl border bg-card shadow-float transition-shadow duration-200 hover:shadow-float-lg">
      <div className="border-b p-4">
        <h3 className="flex items-center gap-2.5 font-heading text-sm font-semibold">
          <span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-primary/10">
            <Award className="size-4 text-primary" />
          </span>
          Overall Comparison Score
        </h3>
        <p className="mt-1 text-xs text-muted-foreground">
          Category wins across all {tally.totalCategories} comparison sections.
        </p>
      </div>

      <div className="space-y-5 p-5">
        {tally.entries.map((entry, rank) => {
          const medal = MEDALS[Math.min(rank, MEDALS.length - 1)];
          const MedalIcon = medal.icon;
          const leader = rank === 0 && entry.wins > 0;
          const fraction = entry.wins / tally.totalCategories;

          return (
            <div key={entry.id} className="space-y-2">
              <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
                <p className="flex min-w-0 items-center gap-2.5">
                  <span
                    className={cn(
                      "flex size-7 shrink-0 items-center justify-center rounded-lg",
                      medal.chip
                    )}
                  >
                    <MedalIcon
                      className={cn("size-4 shrink-0", medal.className)}
                    />
                  </span>
                  <span
                    className={cn(
                      "min-w-0 truncate font-heading text-sm",
                      leader ? "font-bold" : "font-semibold"
                    )}
                  >
                    {resolveName(entry.id)}
                  </span>
                </p>
                <p
                  className={cn(
                    "shrink-0 text-sm font-semibold tabular-nums",
                    leader ? "text-primary" : "text-muted-foreground"
                  )}
                >
                  {entry.wins} {entry.wins === 1 ? "Win" : "Wins"}
                </p>
              </div>

              {/* Proportional win bar — gradient for the leader, animates in. */}
              <div className="h-2.5 overflow-hidden rounded-full bg-muted">
                <div
                  className={cn(
                    "h-full rounded-full transition-[width] duration-700 ease-out",
                    leader ? "bg-brand-gradient" : "bg-primary/30"
                  )}
                  style={{ width: `${Math.max(fraction * 100, entry.wins > 0 ? 6 : 0)}%` }}
                />
              </div>

              {entry.categories.length > 0 && (
                <ul className="flex flex-wrap gap-1.5">
                  {entry.categories.map((category) => (
                    <li
                      key={category}
                      className="rounded-full border border-primary/15 bg-primary/5 px-2 py-0.5 text-[10px] font-medium text-primary"
                    >
                      {category.replace(/ Winner$/, "")}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}
