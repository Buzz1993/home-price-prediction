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

// Medal per rank position: gold trophy, silver medal, bronze medal.
const MEDALS = [
  { icon: Trophy, className: "text-amber-500" },
  { icon: Medal, className: "text-slate-400" },
  { icon: Medal, className: "text-amber-700" },
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
    <section className="overflow-hidden rounded-xl border bg-card shadow-float transition-shadow hover:shadow-float-lg">
      <div className="h-1 bg-primary" />
      <div className="border-b p-3">
        <h3 className="flex items-center gap-2 font-heading text-sm font-semibold">
          <Award className="size-4 text-primary" />
          Overall Comparison Score
        </h3>
        <p className="mt-0.5 text-xs text-muted-foreground">
          Category wins across all {tally.totalCategories} comparison sections.
        </p>
      </div>

      <div className="space-y-4 p-4">
        {tally.entries.map((entry, rank) => {
          const medal = MEDALS[Math.min(rank, MEDALS.length - 1)];
          const MedalIcon = medal.icon;
          const leader = rank === 0 && entry.wins > 0;
          const fraction = entry.wins / tally.totalCategories;

          return (
            <div key={entry.id} className="space-y-1.5">
              <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
                <p className="flex min-w-0 items-center gap-2">
                  <MedalIcon
                    className={cn("size-4 shrink-0", medal.className)}
                  />
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

              {/* Proportional win bar — width animates in on render. */}
              <div className="h-2.5 overflow-hidden rounded-full bg-muted">
                <div
                  className={cn(
                    "h-full rounded-full transition-[width] duration-700 ease-out",
                    leader ? "bg-primary" : "bg-primary/35"
                  )}
                  style={{ width: `${Math.max(fraction * 100, entry.wins > 0 ? 6 : 0)}%` }}
                />
              </div>

              {entry.categories.length > 0 && (
                <ul className="flex flex-wrap gap-1">
                  {entry.categories.map((category) => (
                    <li
                      key={category}
                      className="rounded-full bg-primary/10 px-2 py-0.5 text-[10px] font-medium text-primary"
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
