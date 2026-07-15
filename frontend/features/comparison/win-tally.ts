// Overall win tally for the Property Comparison workspace (Phase 17.1).
// Counts how many of the EXISTING Phase 17.0 scoreboard categories each
// compared property wins — pure counting over compare-winners, no new scoring
// algorithm. The Overall Winner Scoreboard and the Final Decision Card both
// read from this.

import { scoreboard } from "./compare-winners";
import type { CompareBundle } from "./use-compare-data";

export type WinTallyEntry = {
  id: string;
  wins: number;
  // The category labels this property won, e.g. ["Price Winner", …].
  categories: string[];
};

export type WinTally = {
  // Every compared property, ranked by wins (ties keep the compared order).
  entries: WinTallyEntry[];
  // Total categories that produced a winner — the denominator for the bars
  // and the "Won X of Y categories" stat.
  totalCategories: number;
};

export function winTally(bundle: CompareBundle): WinTally {
  const entries = bundle.ids.map((id) => ({
    id,
    wins: 0,
    categories: [] as string[],
  }));
  const byId = new Map(entries.map((entry) => [entry.id, entry]));

  const board = scoreboard(bundle);
  for (const { label, winner } of board) {
    const entry = byId.get(winner.id);
    if (!entry) continue;
    entry.wins += 1;
    entry.categories.push(label);
  }

  return {
    entries: [...entries].sort((a, b) => b.wins - a.wins),
    totalCategories: board.length,
  };
}
