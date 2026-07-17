"use client";

// Premium COMPARISON report document (Phase 17.4). Pure presentation: renders
// the parsed comparison report (the same report-parser.ts model — the parser
// is untouched) with the /compare page's own visual system: the green Best
// Overall Investment hero, the Overall Comparison Score win bars, one
// side-by-side table per analysis with emerald 🏆 winner cells, the Final
// Scoreboard category cards and the large green Final Recommendation block.
// Every value comes verbatim from the backend report text; nothing is
// computed here beyond reading the win counts the report already states.
//
// The standard Reports page document (report-document.tsx) is untouched —
// this renderer is used only for comparison reports (isComparisonReport).

import { createElement } from "react";
import {
  Award,
  Brain,
  Building2,
  Check,
  Gauge,
  Handshake,
  KeyRound,
  LineChart,
  Medal,
  ShieldAlert,
  TrendingUp,
  Trophy,
  type LucideIcon,
} from "lucide-react";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";
import { valueTone } from "@/lib/value-tone";
import type {
  GenericSection,
  ReportBlock,
  ReportModel,
  ReportPair,
} from "./report-parser";

// A report is a comparison report ONLY by its title — the Phase 17.4
// template always prints "EstateMind Property Comparison Report". The
// standard investment report (which can also contain a "Best Overall
// Investment" section for multi-property reports) never carries this title,
// so the Reports page flow can never land here.
export function isComparisonReport(model: ReportModel): boolean {
  return /comparison report/i.test(model.title);
}

// The compare page's own section icons, keyed by report section title.
const SECTION_ICONS: [RegExp, LucideIcon][] = [
  [/basic information/i, Building2],
  [/price prediction/i, TrendingUp],
  [/rental/i, KeyRound],
  [/risk/i, ShieldAlert],
  [/growth/i, LineChart],
  [/valuation/i, Gauge],
  [/investment advisor/i, Brain],
  [/negotiation/i, Handshake],
];

function sectionIcon(title: string): LucideIcon {
  return SECTION_ICONS.find(([re]) => re.test(title))?.[1] ?? Award;
}

// Table-row labels whose values read as a status — rendered as tinted pills,
// same wording→tone rules as the compare page (value-tone).
const PILL_ROWS =
  /^(status|risk level|demand|verdict|growth potential|valuation status|negotiation power)$/i;

const TROPHY_PREFIX = /^🏆\s*/;

function Stars({ value }: { value: string }) {
  return (
    <span className="text-sm tracking-[0.2em] text-primary" aria-label={value}>
      {value}
    </span>
  );
}

function CellValue({ value, rowLabel }: { value: string; rowLabel: string }) {
  if (/^[★☆\s]+$/.test(value)) return <Stars value={value} />;
  if (PILL_ROWS.test(rowLabel)) {
    return (
      <span
        className={cn(
          "inline-flex rounded-full px-2.5 py-0.5 text-xs font-semibold",
          valueTone(value)
        )}
      >
        {value}
      </span>
    );
  }
  return <>{value}</>;
}

// ---------------------------------------------------------------------------
// Side-by-side comparison table — sticky "Metric" column look of the compare
// page's MatrixTable: property header row, zebra rows, and the winning cell
// (the report's own '🏆 ' prefix) tinted emerald with a trophy, exactly like
// the compare page winner cells.
// ---------------------------------------------------------------------------
function ComparisonTable({
  header,
  rows,
}: {
  header: string[];
  rows: string[][];
}) {
  return (
    <div className="overflow-hidden rounded-lg border">
      <Table>
        <TableHeader>
          <TableRow className="bg-primary/5 hover:bg-primary/5">
            {header.map((cell, index) => (
              <TableHead
                key={index}
                className={cn(
                  "font-semibold text-foreground",
                  index > 0 && "text-center"
                )}
              >
                {cell}
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((row, rowIndex) => (
            <TableRow
              key={rowIndex}
              className={cn(rowIndex % 2 === 1 && "bg-muted/40")}
            >
              {row.map((cell, cellIndex) => {
                const isWinner = cellIndex > 0 && TROPHY_PREFIX.test(cell);
                const value = cell.replace(TROPHY_PREFIX, "");
                return (
                  <TableCell
                    key={cellIndex}
                    className={cn(
                      "text-sm",
                      cellIndex === 0
                        ? "font-medium text-muted-foreground"
                        : "text-center",
                      isWinner && "bg-emerald-50/80 font-semibold text-emerald-900"
                    )}
                  >
                    {isWinner ? (
                      <span className="inline-flex items-center gap-1.5">
                        <Trophy className="size-3.5 shrink-0 text-emerald-600" />
                        <CellValue value={value} rowLabel={row[0] ?? ""} />
                      </span>
                    ) : (
                      <CellValue value={value} rowLabel={row[0] ?? ""} />
                    )}
                  </TableCell>
                );
              })}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

// The "Winner: <name> — <figure>" strip closing each analysis section — the
// compare page's SectionWinnerCard look.
function WinnerStrip({ value }: { value: string }) {
  const [name, ...rest] = value.split("—").map((part) => part.trim());
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 rounded-lg border border-primary/25 bg-primary/5 px-3 py-2">
      <p className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-primary">
        <Trophy className="size-3.5" />
        Winner
      </p>
      <p className="font-heading text-sm font-semibold">{name}</p>
      {rest.length > 0 && (
        <p className="text-xs text-muted-foreground">{rest.join(" — ")}</p>
      )}
    </div>
  );
}

function BulletList({
  items,
}: {
  items: { marker: string; text: string }[];
}) {
  return (
    <ul className="space-y-1.5">
      {items.map((item, index) => (
        <li
          key={index}
          className="flex items-start gap-2.5 text-sm leading-relaxed"
        >
          {item.marker === "✓" ? (
            <Check className="mt-0.5 size-4 shrink-0 text-primary" />
          ) : item.marker === "□" || item.marker === "☐" ? (
            <span className="mt-0.5 size-4 shrink-0 rounded-[4px] border-2 border-primary/40" />
          ) : (
            <span className="mt-1.5 size-1.5 shrink-0 rounded-full bg-primary/70" />
          )}
          <span>{item.text}</span>
        </li>
      ))}
    </ul>
  );
}

// Collect every "Label: Value" pair of a section, in order.
function sectionPairs(section: GenericSection): ReportPair[] {
  return section.blocks.flatMap((block) =>
    block.type === "pairs" ? block.pairs : []
  );
}

function findPair(pairs: ReportPair[], label: RegExp): string | null {
  return pairs.find((pair) => label.test(pair.label))?.value ?? null;
}

// The value printed under a standalone label line (e.g. "Property" /
// "Recommended Property" followed by the project name).
function valueAfterLabel(blocks: ReportBlock[], label: RegExp): string | null {
  for (let i = 0; i < blocks.length; i++) {
    const block = blocks[i];
    if (block.type === "label" && label.test(block.text)) {
      const next = blocks[i + 1];
      if (next?.type === "text") return next.lines.join(" ");
    }
  }
  return null;
}

function bullets(blocks: ReportBlock[]): { marker: string; text: string }[] {
  return blocks.flatMap((block) =>
    block.type === "bullets" ? block.items : []
  );
}

// ---------------------------------------------------------------------------
// 🏆 Best Overall Investment — the compare page's green Executive Winner hero.
// ---------------------------------------------------------------------------
function BestInvestmentHero({ section }: { section: GenericSection }) {
  const pairs = sectionPairs(section);
  const name = valueAfterLabel(section.blocks, /^property$/i);
  const score = findPair(pairs, /^overall score$/i);
  const verdict = findPair(pairs, /^verdict$/i);
  const wins = findPair(pairs, /^category wins$/i);
  const reasons = bullets(section.blocks);

  return (
    <section className="report-avoid-break overflow-hidden rounded-xl border border-primary/30 shadow-md">
      <div className="bg-primary p-6 text-center text-primary-foreground">
        <p className="flex items-center justify-center gap-1.5 text-[11px] font-semibold uppercase tracking-[0.2em] text-primary-foreground/80">
          <Trophy className="size-3.5" />
          Best Overall Investment
        </p>
        <h3 className="mt-2 break-words font-heading text-3xl font-bold leading-tight sm:text-4xl">
          {name ?? section.title}
        </h3>
        {verdict && (
          <span className="mt-2 inline-flex rounded-full bg-primary-foreground/15 px-3 py-1 text-xs font-semibold">
            {verdict}
          </span>
        )}
        <div className="mx-auto mt-4 flex max-w-md justify-center gap-10 border-t border-primary-foreground/15 pt-4">
          {score && (
            <div>
              <p className="text-[11px] font-medium uppercase tracking-wide text-primary-foreground/70">
                Overall Score
              </p>
              <p className="font-heading text-2xl font-bold tabular-nums sm:text-3xl">
                {score}
              </p>
            </div>
          )}
          {wins && (
            <div>
              <p className="text-[11px] font-medium uppercase tracking-wide text-primary-foreground/70">
                Category Wins
              </p>
              <p className="font-heading text-2xl font-bold tabular-nums sm:text-3xl">
                {wins}
              </p>
            </div>
          )}
        </div>
      </div>
      {reasons.length > 0 && (
        <div className="bg-card p-4">
          <p className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
            Why this property?
          </p>
          <ul className="grid gap-x-6 gap-y-1.5 sm:grid-cols-2">
            {reasons.map((reason, i) => (
              <li
                key={i}
                className="flex items-start gap-1.5 text-sm font-medium"
              >
                <Check className="mt-0.5 size-4 shrink-0 text-primary" />
                <span className="min-w-0 break-words">{reason.text}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}

// ---------------------------------------------------------------------------
// 📊 Overall Comparison Score — the compare page's win-tally bars. The win
// counts come verbatim from the report ("<Name>: <n> Wins" lines); the bar
// widths only draw those stated counts proportionally.
// ---------------------------------------------------------------------------
type TallyEntry = { name: string; winsText: string; wins: number; categories: string | null };

function parseTally(section: GenericSection): TallyEntry[] {
  const entries: TallyEntry[] = [];
  for (const pair of sectionPairs(section)) {
    if (/^categories won$/i.test(pair.label)) {
      const last = entries[entries.length - 1];
      if (last) last.categories = /^none$/i.test(pair.value) ? null : pair.value;
      continue;
    }
    const wins = parseInt(pair.value, 10);
    if (/win/i.test(pair.value) && !Number.isNaN(wins)) {
      entries.push({ name: pair.label, winsText: pair.value, wins, categories: null });
    }
  }
  return entries;
}

const MEDALS = [
  { icon: Trophy, className: "text-amber-500" },
  { icon: Medal, className: "text-slate-400" },
  { icon: Medal, className: "text-amber-700" },
];

function ScoreboardBars({ section }: { section: GenericSection }) {
  const entries = parseTally(section);
  if (entries.length === 0) return null;
  const total = entries.reduce((sum, entry) => sum + entry.wins, 0);

  return (
    <section className="report-avoid-break overflow-hidden rounded-xl border bg-card shadow-sm">
      <div className="h-1 bg-primary" />
      <div className="border-b p-3">
        <h3 className="flex items-center gap-2 font-heading text-sm font-semibold">
          <Award className="size-4 text-primary" />
          Overall Comparison Score
        </h3>
        <p className="mt-0.5 text-xs text-muted-foreground">
          Category wins across the compared properties.
        </p>
      </div>
      <div className="space-y-4 p-4">
        {entries.map((entry, rank) => {
          const medal = MEDALS[Math.min(rank, MEDALS.length - 1)];
          const MedalIcon = medal.icon;
          const leader = rank === 0 && entry.wins > 0;
          const fraction = total > 0 ? entry.wins / total : 0;
          return (
            <div key={entry.name} className="space-y-1.5">
              <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
                <p className="flex min-w-0 items-center gap-2">
                  <MedalIcon className={cn("size-4 shrink-0", medal.className)} />
                  <span
                    className={cn(
                      "min-w-0 truncate font-heading text-sm",
                      leader ? "font-bold" : "font-semibold"
                    )}
                  >
                    {entry.name}
                  </span>
                </p>
                <p
                  className={cn(
                    "shrink-0 text-sm font-semibold tabular-nums",
                    leader ? "text-primary" : "text-muted-foreground"
                  )}
                >
                  {entry.winsText}
                </p>
              </div>
              <div className="h-2.5 overflow-hidden rounded-full bg-muted">
                <div
                  className={cn(
                    "h-full rounded-full",
                    leader ? "bg-primary" : "bg-primary/35"
                  )}
                  style={{
                    width: `${Math.max(fraction * 100, entry.wins > 0 ? 6 : 0)}%`,
                  }}
                />
              </div>
              {entry.categories && (
                <ul className="flex flex-wrap gap-1">
                  {entry.categories.split(/,\s*/).map((category) => (
                    <li
                      key={category}
                      className="rounded-full bg-primary/10 px-2 py-0.5 text-[10px] font-medium text-primary"
                    >
                      {category.replace(/ Winner$/i, "")}
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

// ---------------------------------------------------------------------------
// One analysis section — icon-chip banner + side-by-side table + winner strip,
// the compare page's accordion section permanently expanded.
// ---------------------------------------------------------------------------
function AnalysisSection({ section }: { section: GenericSection }) {
  // Lucide icon picked from the section title — rendered via createElement so
  // no component is defined during render (react-hooks/static-components).
  const icon = sectionIcon(section.title);
  const winner = findPair(sectionPairs(section), /^winner$/i);

  return (
    <section className="report-avoid-break overflow-hidden rounded-xl border bg-card shadow-sm">
      <div className="flex items-center gap-2.5 border-b p-3">
        <span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-primary/10">
          {createElement(icon, { className: "size-4 text-primary" })}
        </span>
        <h3 className="font-heading text-sm font-semibold">{section.title}</h3>
      </div>
      <div className="space-y-3 p-3">
        {section.blocks.map((block, index) => {
          switch (block.type) {
            case "table":
              return (
                <ComparisonTable
                  key={index}
                  header={block.header}
                  rows={block.rows}
                />
              );
            case "pairs": {
              // The "Winner:" pair renders as the closing strip below; any
              // other pair (rare) renders as a plain labelled line.
              const rest = block.pairs.filter((p) => !/^winner$/i.test(p.label));
              if (rest.length === 0) return null;
              return (
                <dl key={index} className="space-y-1">
                  {rest.map((pair) => (
                    <div key={pair.label} className="flex justify-between gap-4">
                      <dt className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                        {pair.label}
                      </dt>
                      <dd className="text-sm font-medium">{pair.value}</dd>
                    </div>
                  ))}
                </dl>
              );
            }
            case "bullets":
              return <BulletList key={index} items={block.items} />;
            case "text":
              return (
                <p key={index} className="text-sm leading-relaxed">
                  {block.lines.join(" ")}
                </p>
              );
            default:
              return null;
          }
        })}
        {winner && <WinnerStrip value={winner} />}
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// 📋 Final Scoreboard — the compare page's category-winner cards grid.
// ---------------------------------------------------------------------------
function FinalScoreboardCards({ section }: { section: GenericSection }) {
  const pairs = sectionPairs(section);
  if (pairs.length === 0) return null;

  return (
    <section className="report-avoid-break overflow-hidden rounded-xl border bg-card shadow-sm">
      <div className="h-1 bg-primary" />
      <div className="border-b p-3">
        <h3 className="flex items-center gap-2 font-heading text-sm font-semibold">
          <Award className="size-4 text-primary" />
          Final Scoreboard
        </h3>
        <p className="mt-0.5 text-xs text-muted-foreground">
          Category winners across the compared properties.
        </p>
      </div>
      <div className="grid gap-2 p-3 sm:grid-cols-2">
        {pairs.map(({ label, value }) => {
          const [name, ...detail] = value.split("—").map((part) => part.trim());
          return (
            <div key={label} className="rounded-lg border bg-muted/20 px-3 py-2">
              <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                {label}
              </p>
              <p className="mt-0.5 flex items-center gap-1.5 font-heading text-sm font-semibold">
                <Trophy className="size-3.5 shrink-0 text-primary" />
                <span className="min-w-0 truncate">{name}</span>
              </p>
              {detail.length > 0 && (
                <p className="mt-0.5 truncate text-xs text-muted-foreground">
                  {detail.join(" — ")}
                </p>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// 🏆 Final Recommendation — the compare page's large green closing block.
// ---------------------------------------------------------------------------
function FinalRecommendationHero({ section }: { section: GenericSection }) {
  const pairs = sectionPairs(section);
  const name = valueAfterLabel(section.blocks, /^recommended property$/i);
  const score = findPair(pairs, /^overall score$/i);
  const verdict = findPair(pairs, /^verdict$/i);
  const wins = findPair(pairs, /^category wins$/i);
  const runnerUp = findPair(pairs, /^runner up$/i);
  const action = findPair(pairs, /^recommended action$/i);
  const reasons = bullets(section.blocks);

  return (
    <section className="report-avoid-break overflow-hidden rounded-xl border border-primary/30 shadow-md">
      <div className="bg-primary p-5 text-primary-foreground sm:p-6">
        <p className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-[0.2em] text-primary-foreground/80">
          <Trophy className="size-3.5" />
          Final Recommendation
        </p>
        <div className="mt-1.5 flex flex-wrap items-end justify-between gap-x-6 gap-y-3">
          <div className="min-w-0">
            <h3 className="break-words font-heading text-3xl font-bold leading-tight sm:text-4xl">
              {name ?? section.title}
            </h3>
            {verdict && (
              <span className="mt-2 inline-flex rounded-full bg-primary-foreground/15 px-2.5 py-0.5 text-xs font-semibold">
                {verdict}
              </span>
            )}
          </div>
          <div className="flex shrink-0 gap-6 text-right">
            {score && (
              <div>
                <p className="text-[11px] font-medium uppercase tracking-wide text-primary-foreground/70">
                  Overall Score
                </p>
                <p className="font-heading text-2xl font-bold tabular-nums sm:text-3xl">
                  {score}
                </p>
              </div>
            )}
            {wins && (
              <div>
                <p className="text-[11px] font-medium uppercase tracking-wide text-primary-foreground/70">
                  Category Wins
                </p>
                <p className="font-heading text-2xl font-bold tabular-nums sm:text-3xl">
                  {wins}
                </p>
              </div>
            )}
          </div>
        </div>
        {reasons.length > 0 && (
          <div className="mt-4 border-t border-primary-foreground/15 pt-3">
            <p className="text-[11px] font-semibold uppercase tracking-wide text-primary-foreground/70">
              Why choose it?
            </p>
            <ul className="mt-1.5 grid gap-x-6 gap-y-1 sm:grid-cols-2">
              {reasons.map((reason, i) => (
                <li
                  key={i}
                  className="flex items-start gap-1.5 text-xs font-medium text-primary-foreground/90"
                >
                  <Check className="mt-0.5 size-3.5 shrink-0" />
                  <span className="min-w-0 break-words">{reason.text}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
      {(action || runnerUp) && (
        <div className="space-y-2 bg-card p-3">
          {action && (
            <p className="flex items-start gap-2 text-sm">
              <Handshake className="mt-0.5 size-4 shrink-0 text-primary" />
              <span className="min-w-0">
                <span className="block text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
                  Recommended action
                </span>
                <span className="break-words font-medium">{action}</span>
              </span>
            </p>
          )}
          {runnerUp && (
            <p className="flex items-start gap-2 text-sm">
              <Medal className="mt-0.5 size-4 shrink-0 text-amber-500" />
              <span className="min-w-0">
                <span className="block text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
                  Runner up
                </span>
                <span className="break-words">{runnerUp}</span>
              </span>
            </p>
          )}
        </div>
      )}
    </section>
  );
}

// Fallback card for any section this renderer does not specially style
// (e.g. Suggested Next Steps) — same generic card look as the compare page.
function GenericCard({ section }: { section: GenericSection }) {
  return (
    <section className="report-avoid-break rounded-xl border bg-card p-4 shadow-sm">
      <h3 className="mb-3 flex items-center gap-2 border-b pb-2 font-heading text-sm font-semibold">
        {section.icon && (
          <span aria-hidden className="text-base">
            {section.icon}
          </span>
        )}
        {section.title}
      </h3>
      <div className="space-y-3">
        {section.blocks.map((block, index) => {
          switch (block.type) {
            case "bullets":
              return <BulletList key={index} items={block.items} />;
            case "table":
              return (
                <ComparisonTable
                  key={index}
                  header={block.header}
                  rows={block.rows}
                />
              );
            case "pairs":
              return (
                <dl key={index} className="space-y-1">
                  {block.pairs.map((pair) => (
                    <div key={pair.label} className="flex justify-between gap-4">
                      <dt className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                        {pair.label}
                      </dt>
                      <dd className="text-sm font-medium">{pair.value}</dd>
                    </div>
                  ))}
                </dl>
              );
            case "label":
              return (
                <p
                  key={index}
                  className="pt-1 text-xs font-semibold uppercase tracking-wider text-muted-foreground"
                >
                  {block.text}
                </p>
              );
            case "text":
              return (
                <p key={index} className="text-sm leading-relaxed">
                  {block.lines.join(" ")}
                </p>
              );
          }
        })}
      </div>
    </section>
  );
}

function DocumentHeader({ model }: { model: ReportModel }) {
  return (
    <header className="report-avoid-break overflow-hidden rounded-xl border bg-card shadow-sm">
      <div className="h-1.5 bg-primary" />
      <div className="space-y-5 p-6">
        <div className="flex items-center gap-2 font-heading text-lg font-semibold">
          <span className="flex size-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
            <Building2 className="size-5" />
          </span>
          EstateMind
        </div>
        <h2 className="font-heading text-2xl font-bold tracking-tight sm:text-3xl">
          {model.title}
        </h2>
        <dl className="grid gap-4 border-t pt-4 sm:grid-cols-2">
          {model.generatedFor && (
            <div>
              <dt className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                Compared Properties
              </dt>
              <dd className="mt-0.5 text-sm font-medium">
                {model.generatedFor}
              </dd>
            </div>
          )}
          {model.reportDate && (
            <div>
              <dt className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                Report Date
              </dt>
              <dd className="mt-0.5 text-sm font-medium">{model.reportDate}</dd>
            </div>
          )}
        </dl>
      </div>
    </header>
  );
}

export function ComparisonReportDocument({ model }: { model: ReportModel }) {
  return (
    <div className="report-document mx-auto w-full max-w-3xl space-y-5">
      <DocumentHeader model={model} />
      {model.sections.map((section, index) => {
        if (section.kind !== "generic") return null;
        const title = section.title.toLowerCase();
        if (title.includes("best overall investment")) {
          return <BestInvestmentHero key={index} section={section} />;
        }
        if (title.includes("overall comparison score")) {
          return <ScoreboardBars key={index} section={section} />;
        }
        if (title.includes("final scoreboard")) {
          return <FinalScoreboardCards key={index} section={section} />;
        }
        if (title.includes("final recommendation")) {
          return <FinalRecommendationHero key={index} section={section} />;
        }
        if (section.blocks.some((block) => block.type === "table")) {
          return <AnalysisSection key={index} section={section} />;
        }
        return <GenericCard key={index} section={section} />;
      })}
      <footer className="border-t pt-3 text-center text-xs text-muted-foreground">
        EstateMind · Property Comparison Report · Generated by the EstateMind
        analysis platform
      </footer>
    </div>
  );
}
