"use client";

// At-a-Glance winner strip (Phase 17.2, Goal 2). One horizontal row of
// compact category-winner cards rendered right below the Executive Summary:
// Lowest Price, Highest Rental, Lowest Risk, Best Growth, Best Investment.
// Each card names the winner and restates its backend-derived reason and
// supporting metric — the SAME winners and points the Phase 17.0 category
// winners already compute (compare-winners); nothing is re-scored here.

import {
  Brain,
  KeyRound,
  LineChart,
  ShieldCheck,
  Tag,
  Trophy,
  type LucideIcon,
} from "lucide-react";

import {
  growthWinner,
  investmentWinner,
  priceWinner,
  rentalWinner,
  riskWinner,
  type CategoryWinner,
} from "./compare-winners";
import type { CompareBundle } from "./use-compare-data";

// The strip's categories, in display order. Each reads one existing
// compare-winners builder.
const STRIP: {
  label: string;
  icon: LucideIcon;
  pick: (bundle: CompareBundle) => CategoryWinner;
}[] = [
  { label: "Lowest Price", icon: Tag, pick: priceWinner },
  { label: "Highest Rental", icon: KeyRound, pick: rentalWinner },
  { label: "Lowest Risk", icon: ShieldCheck, pick: riskWinner },
  { label: "Best Growth", icon: LineChart, pick: growthWinner },
  { label: "Best Investment", icon: Brain, pick: investmentWinner },
];

export function WinnerStrip({
  bundle,
  resolveName,
}: {
  bundle: CompareBundle;
  resolveName: (id: string) => string;
}) {
  const cards = STRIP.map((entry) => ({
    ...entry,
    winner: entry.pick(bundle),
  })).filter((entry) => entry.winner !== null);
  if (cards.length === 0) return null;

  return (
    <section
      aria-label="Category winners at a glance"
      className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5"
    >
      {cards.map(({ label, icon: Icon, winner }) => (
        <div
          key={label}
          className="card-accent-top rounded-xl border bg-card p-3.5 shadow-float transition-all duration-200 hover:-translate-y-0.5 hover:shadow-float-lg"
        >
          <p className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-primary">
            <Trophy className="size-3.5 shrink-0" />
            {label}
          </p>
          <p className="mt-1.5 flex items-center gap-2 font-heading text-sm font-semibold">
            <span className="flex size-6 shrink-0 items-center justify-center rounded-md bg-primary/10">
              <Icon className="size-3.5 text-primary" />
            </span>
            <span className="min-w-0 truncate">
              {resolveName(winner!.id)}
            </span>
          </p>
          {/* Reason + one supporting metric, verbatim backend-derived points */}
          {winner!.points[0] && (
            <p className="mt-1.5 truncate text-xs text-muted-foreground">
              {winner!.points[0]}
            </p>
          )}
          {winner!.points[1] && (
            <p className="truncate text-[11px] text-muted-foreground/80">
              {winner!.points[1]}
            </p>
          )}
        </div>
      ))}
    </section>
  );
}
