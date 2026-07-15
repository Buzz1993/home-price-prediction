"use client";

// AI analysis comparison sections (Phase 17.0, collapsible since Phase 17.1).
// Each existing backend analysis becomes one accordion section — the unchanged
// rows fetched by use-compare-data, rendered attribute-by-attribute across the
// compared properties — closing with its "Why this property won" strip. The
// Basic Information (Property Overview matrix) joins the same accordion as the
// first section. Open sections persist to localStorage so the last-used layout
// survives a reload. Highlighting and winners only compare existing backend
// values (compare-utils / compare-winners); no analysis result is recomputed
// here.

import { useState } from "react";
import {
  Brain,
  Building2,
  Gauge,
  Handshake,
  KeyRound,
  LineChart,
  ShieldAlert,
  TrendingUp,
  type LucideIcon,
} from "lucide-react";

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { formatCr, formatPerSqft, splitList } from "@/features/dashboard/format";
import { StatusPill } from "@/features/analysis/ui/analysis-ui";
import type {
  AdvisorRow,
  AnalysisRow,
  NegotiationRow,
} from "@/types/dashboard";
import {
  barFractions,
  hasText,
  rankLevelCells,
  rankNumericCells,
  rankStatusCells,
  relativeNotes,
  type CellTone,
} from "./compare-utils";
import {
  advisorWinner,
  growthWinner,
  negotiationWinner,
  predictionWinner,
  rentalWinner,
  riskWinner,
  valuationWinner,
  type CategoryWinner,
} from "./compare-winners";
import {
  MatrixTable,
  SectionWinnerCard,
  type MatrixColumn,
  type MatrixRowData,
} from "./matrix-table";
import type { CompareBundle } from "./use-compare-data";

// Rent estimates from the backend are absolute ₹ amounts (not Crores) — same
// formatting as the Rental Analysis renderer.
function formatRupees(value: unknown): string {
  const n = typeof value === "string" ? Number(value) : (value as number);
  if (typeof n !== "number" || Number.isNaN(n)) return "—";
  return `₹${Math.round(n).toLocaleString("en-IN")}`;
}

// Muted long-text cell (strategies, reasons, concern lists).
function LongText({ value }: { value: unknown }) {
  return (
    <span className="block max-w-72 text-xs leading-relaxed text-muted-foreground">
      {String(value)}
    </span>
  );
}

// Build one highlighted comparison row; null when no property has the value.
function row(
  label: string,
  values: unknown[],
  {
    tooltip,
    tones,
    badge,
    bars,
    notes,
    render,
  }: {
    tooltip?: string;
    tones?: CellTone[];
    badge?: string;
    // Goal 2 proportional bars / Goal 6 relative notes, precomputed from the
    // same backend values by compare-utils.
    bars?: (number | null)[];
    notes?: (string | null)[];
    render?: (value: unknown) => React.ReactNode;
  } = {}
): MatrixRowData | null {
  if (!values.some(hasText)) return null;
  return {
    label,
    tooltip,
    cells: values.map((value, i) => ({
      content: hasText(value) ? (render ? render(value) : String(value)) : "—",
      tone: tones?.[i] ?? "neutral",
      badge,
      barFraction: bars?.[i] ?? null,
      note: notes?.[i] ?? null,
    })),
  };
}

const notNull = (r: MatrixRowData | null): r is MatrixRowData => r !== null;

const pill = (v: unknown) => <StatusPill value={v as string | number} />;

// One collapsible analysis section: accordion item → table + "why the winner
// won" strip. The trigger carries the section icon and title.
function Section({
  value,
  title,
  icon: Icon,
  columns,
  rows,
  winner,
  winnerCategory,
  resolveName,
}: {
  value: string;
  title: string;
  icon: LucideIcon;
  columns: MatrixColumn[];
  rows: MatrixRowData[];
  winner: CategoryWinner;
  winnerCategory: string;
  resolveName: (id: string) => string;
}) {
  if (rows.length === 0) return null;
  return (
    <AccordionItem value={value}>
      <AccordionTrigger>
        <span className="flex items-center gap-2 font-heading text-sm font-semibold">
          <Icon className="size-4 text-primary" />
          {title}
        </span>
      </AccordionTrigger>
      <AccordionContent>
        <MatrixTable columns={columns} rows={rows} bare />
        {winner && (
          <div className="border-t p-3">
            <SectionWinnerCard
              category={winnerCategory}
              name={resolveName(winner.id)}
              points={winner.points}
            />
          </div>
        )}
      </AccordionContent>
    </AccordionItem>
  );
}

// Align backend rows to the compared column order by id.
function byId<T>(rows: T[], ids: string[]): (T | undefined)[] {
  return ids.map((id) =>
    rows.find((r) => String((r as { id?: unknown }).id) === id)
  );
}

// Persisted open-section state (Goal 4): the accordion remembers which
// sections were last open across reloads. localStorage is read lazily in the
// useState initializer so SSR never touches window.
const OPEN_SECTIONS_KEY = "estatemind.compare.open-sections";
const DEFAULT_OPEN = ["basic-info"];

function loadOpenSections(): string[] {
  if (typeof window === "undefined") return DEFAULT_OPEN;
  try {
    const raw = window.localStorage.getItem(OPEN_SECTIONS_KEY);
    const parsed = raw ? JSON.parse(raw) : null;
    return Array.isArray(parsed) ? parsed.map(String) : DEFAULT_OPEN;
  } catch {
    return DEFAULT_OPEN;
  }
}

export function AnalysisComparisonSections({
  bundle,
  columns,
  resolveName,
  overview,
}: {
  bundle: CompareBundle;
  columns: MatrixColumn[];
  resolveName: (id: string) => string;
  // The Property Overview matrix (Basic Information) — rendered as the first
  // collapsible section so the whole comparison shares one accordion.
  overview?: React.ReactNode;
}) {
  const [open, setOpen] = useState<string[]>(loadOpenSections);

  const handleOpenChange = (values: string[]) => {
    setOpen(values);
    try {
      window.localStorage.setItem(OPEN_SECTIONS_KEY, JSON.stringify(values));
    } catch {
      // Storage unavailable (private mode) — the accordion still works.
    }
  };

  const { ids } = bundle;
  // Column names for the Goal-6 relative-difference notes ("cheaper than …").
  const names = ids.map(resolveName);

  // ---- Price Prediction ----------------------------------------------------
  const prediction = byId(bundle.prediction, ids);
  const val = (rows: (AnalysisRow | undefined)[], key: string) =>
    rows.map((r) => r?.[key]);
  const margin = val(prediction, "margin_diff");
  const asking = val(prediction, "original_price");
  const predictionRows = [
    // The backend sets `status` only on failed predictions.
    prediction.some((r) => hasText(r?.status))
      ? row("Status", val(prediction, "status"), { render: pill })
      : null,
    row("Asking Price", asking, {
      tooltip: "The current asking price, in ₹ Crores.",
      tones: rankNumericCells(asking, "lowest"),
      badge: "Lowest",
      bars: barFractions(asking, "lowest"),
      notes: relativeNotes(
        asking,
        "lowest",
        names,
        (diff, other) => `${formatCr(diff)} cheaper than ${other}`
      ),
      render: (v) => formatCr(v as number | string),
    }),
    row("Predicted Price", val(prediction, "predicted_price"), {
      tooltip: "The ML model's estimated market value for the property.",
      render: (v) => formatCr(v as number | string),
    }),
    row("Value Margin", margin, {
      tooltip:
        "Predicted minus asking price — a positive margin means the property is priced below its predicted market value.",
      tones: rankNumericCells(margin, "highest"),
      badge: "Best value",
      bars: barFractions(margin, "highest"),
      notes: relativeNotes(
        margin,
        "highest",
        names,
        (diff, other) => `${formatCr(diff)} better margin than ${other}`
      ),
      render: (v) => formatCr(v as number | string),
    }),
  ].filter(notNull);

  // ---- Rental Analysis -----------------------------------------------------
  const rental = byId(bundle.rental, ids);
  const yields = val(rental, "rental_yield_percent");
  const rentalRows = [
    row("Rental Yield", yields, {
      tooltip: "Annual rental income divided by property price.",
      tones: rankNumericCells(yields, "highest"),
      badge: "Best",
      bars: barFractions(yields, "highest"),
      notes: relativeNotes(
        yields,
        "highest",
        names,
        (diff, other) => `+${diff.toFixed(2)}% higher yield than ${other}`
      ),
    }),
    row("Monthly Rent", val(rental, "monthly_rent_estimate"), {
      tooltip: "Estimated monthly rental income.",
      tones: rankNumericCells(val(rental, "monthly_rent_estimate"), "highest"),
      bars: barFractions(val(rental, "monthly_rent_estimate"), "highest"),
      render: formatRupees,
    }),
    row("Annual Rent", val(rental, "annual_rent"), {
      tooltip: "Estimated yearly rental income.",
      tones: rankNumericCells(val(rental, "annual_rent"), "highest"),
      bars: barFractions(val(rental, "annual_rent"), "highest"),
      notes: relativeNotes(
        val(rental, "annual_rent"),
        "highest",
        names,
        (diff, other) => `${formatRupees(diff)} more per year than ${other}`
      ),
      render: formatRupees,
    }),
    row("Investment Rating", val(rental, "investment_rating"), {
      tooltip: "The backend's overall rental-investment grade.",
      tones: rankStatusCells(val(rental, "investment_rating")),
      render: pill,
    }),
    row("Demand", val(rental, "demand_level"), {
      tooltip: "Expected tenant demand for this property.",
      tones: rankLevelCells(val(rental, "demand_level")),
      render: pill,
    }),
    row("Rental Strategy", val(rental, "rental_strategy"), {
      render: (v) => <LongText value={v} />,
    }),
  ].filter(notNull);

  // ---- Risk Analysis (from the advisor rows) --------------------------------
  const advisor = byId(bundle.advisor, ids);
  const aRec = (r: AdvisorRow | undefined) =>
    (r ?? {}) as unknown as Record<string, unknown>;
  // risk_label when the backend provides it, its verdict otherwise (RiskCards).
  const riskLevels = advisor.map((r) =>
    hasText(aRec(r).risk_label) ? aRec(r).risk_label : r?.verdict
  );
  const concernCounts = advisor.map((r) =>
    r ? splitList(r.risks).length : null
  );
  const riskRows = [
    row("Risk Level", riskLevels, {
      tooltip:
        "The engine's judgement of the property once risks are weighed against its strengths.",
      tones: rankStatusCells(riskLevels),
      badge: "Lowest risk",
      render: pill,
    }),
    row("Risk Score", advisor.map((r) => aRec(r).risk_score), {
      tooltip: "The analysis engine's internal risk score for the property.",
    }),
    row("Major Concerns", advisor.map((r) => r?.risks), {
      tooltip: "Risk factors the analysis engine flagged for this property.",
      tones: rankNumericCells(concernCounts, "lowest"),
      badge: "Fewest",
      render: (v) => {
        const list = splitList(String(v));
        return (
          <span className="block max-w-72">
            <span className="font-medium">
              {list.length} concern{list.length === 1 ? "" : "s"}
            </span>
            <LongText value={v} />
          </span>
        );
      },
    }),
    row("Verdict", advisor.map((r) => r?.verdict), {
      tooltip: "The advisor engine's overall judgement of the property.",
      tones: rankStatusCells(advisor.map((r) => r?.verdict)),
      render: pill,
    }),
  ].filter(notNull);

  // ---- Future Growth (from the advisor rows) --------------------------------
  const growthLabels = advisor.map((r) => r?.growth_label);
  const growthRows = [
    row("Growth Label", growthLabels, {
      tooltip: "The backend's long-term appreciation outlook for the area.",
      tones: rankStatusCells(growthLabels),
      badge: "Best outlook",
      render: pill,
    }),
    row("Growth Score", advisor.map((r) => r?.growth_score), {
      tooltip: "The backend's growth score for the area (higher is stronger).",
      tones: rankNumericCells(
        advisor.map((r) => r?.growth_score),
        "highest"
      ),
      badge: "Highest",
      bars: barFractions(
        advisor.map((r) => r?.growth_score),
        "highest"
      ),
    }),
    row(
      "Infrastructure Signals",
      advisor.map((r) =>
        hasText(r?.infra_detected) ? r?.infra_detected : r?.future_signals
      ),
      { render: (v) => <LongText value={v} /> }
    ),
    row("Future Outlook", advisor.map((r) => r?.growth_reason), {
      render: (v) => <LongText value={v} />,
    }),
  ].filter(notNull);

  // ---- Property Valuation ----------------------------------------------------
  const valuation = byId(bundle.valuation, ids);
  const flags = val(valuation, "analysis_flag");
  const valuationRows = [
    row("Valuation", flags, {
      tooltip:
        "Whether the property is overpriced, fairly priced or undervalued vs. its market benchmark.",
      tones: rankStatusCells(flags),
      badge: "Best value",
      render: pill,
    }),
    row("Assessment", val(valuation, "analysis_msg"), {
      render: (v) => <LongText value={v} />,
    }),
    row("Severity", val(valuation, "analysis_severity"), {
      tooltip: "How strong the valuation deviation is.",
      render: pill,
    }),
    row("Price per sqft", val(valuation, "costpersqft"), {
      tooltip: "Property price divided by area.",
      tones: rankNumericCells(val(valuation, "costpersqft"), "lowest"),
      badge: "Lowest ₹/sqft",
      bars: barFractions(val(valuation, "costpersqft"), "lowest"),
      notes: relativeNotes(
        val(valuation, "costpersqft"),
        "lowest",
        names,
        (diff, other) =>
          `₹${Math.round(diff).toLocaleString("en-IN")}/sqft less than ${other}`
      ),
      render: (v) => formatPerSqft(v as number | string),
    }),
  ].filter(notNull);

  // ---- Investment Advisor -----------------------------------------------------
  const scores = advisor.map((r) => aRec(r).overall_score);
  const advisorRows = [
    row("Final Verdict", advisor.map((r) => r?.verdict), {
      tooltip:
        "The advisor engine's overall judgement, weighing valuation, risk, growth, rental and negotiation together.",
      tones: rankStatusCells(advisor.map((r) => r?.verdict)),
      render: pill,
    }),
    row("Investment Score", scores, {
      tooltip: "The backend's combined investment score for the property.",
      tones: rankNumericCells(scores, "highest"),
      badge: "Highest score",
      bars: barFractions(scores, "highest"),
    }),
    row("Positives", advisor.map((r) => r?.positives), {
      render: (v) => <LongText value={v} />,
    }),
    row("Weaknesses", advisor.map((r) => r?.risks), {
      render: (v) => <LongText value={v} />,
    }),
    row("Suitable For", advisor.map((r) => r?.suitable_for), {
      tooltip: "The investor profile this property fits best.",
    }),
  ].filter(notNull);

  // ---- Negotiation -------------------------------------------------------------
  const negotiation = byId(bundle.negotiation, ids);
  const nVal = (key: keyof NegotiationRow) => negotiation.map((r) => r?.[key]);
  const discounts = nVal("suggested_discount_percent");
  const negotiationRows = [
    row("Negotiation Power", nVal("negotiation_power"), {
      tooltip: "Estimated ability to negotiate below the asking price.",
      tones: rankStatusCells(nVal("negotiation_power")),
      render: pill,
    }),
    row("Target Price", nVal("target_price"), {
      tooltip: "The offer price the negotiation engine recommends.",
      render: (v) => formatCr(v as number | string),
    }),
    row("Suggested Discount", discounts, {
      tooltip: "The discount the engine suggests pursuing off the asking price.",
      tones: rankNumericCells(discounts, "highest"),
      badge: "Highest",
      bars: barFractions(discounts, "highest"),
      notes: relativeNotes(
        discounts,
        "highest",
        names,
        () => "Highest discount among compared properties"
      ),
    }),
    row("Strategy", nVal("strategy"), {
      render: (v) => <LongText value={v} />,
    }),
  ].filter(notNull);

  return (
    <Accordion
      type="multiple"
      value={open}
      onValueChange={handleOpenChange}
      className="space-y-3"
    >
      {overview && (
        <AccordionItem value="basic-info">
          <AccordionTrigger>
            <span className="flex items-center gap-2 font-heading text-sm font-semibold">
              <Building2 className="size-4 text-primary" />
              Basic Information
            </span>
          </AccordionTrigger>
          <AccordionContent className="p-3">{overview}</AccordionContent>
        </AccordionItem>
      )}
      <Section
        value="prediction"
        title="Price Prediction"
        icon={TrendingUp}
        columns={columns}
        rows={predictionRows}
        winner={predictionWinner(bundle)}
        winnerCategory="Price Prediction Winner"
        resolveName={resolveName}
      />
      <Section
        value="rental"
        title="Rental Analysis"
        icon={KeyRound}
        columns={columns}
        rows={rentalRows}
        winner={rentalWinner(bundle)}
        winnerCategory="Rental Winner"
        resolveName={resolveName}
      />
      <Section
        value="risk"
        title="Risk Analysis"
        icon={ShieldAlert}
        columns={columns}
        rows={riskRows}
        winner={riskWinner(bundle)}
        winnerCategory="Risk Winner"
        resolveName={resolveName}
      />
      <Section
        value="growth"
        title="Future Growth"
        icon={LineChart}
        columns={columns}
        rows={growthRows}
        winner={growthWinner(bundle)}
        winnerCategory="Growth Winner"
        resolveName={resolveName}
      />
      <Section
        value="valuation"
        title="Property Valuation"
        icon={Gauge}
        columns={columns}
        rows={valuationRows}
        winner={valuationWinner(bundle)}
        winnerCategory="Valuation Winner"
        resolveName={resolveName}
      />
      <Section
        value="advisor"
        title="Investment Advisor"
        icon={Brain}
        columns={columns}
        rows={advisorRows}
        winner={advisorWinner(bundle)}
        winnerCategory="Investment Winner"
        resolveName={resolveName}
      />
      <Section
        value="negotiation"
        title="Negotiation"
        icon={Handshake}
        columns={columns}
        rows={negotiationRows}
        winner={negotiationWinner(bundle)}
        winnerCategory="Negotiation Winner"
        resolveName={resolveName}
      />
    </Accordion>
  );
}
