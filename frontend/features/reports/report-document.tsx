"use client";

// Premium report document (Phase 15.17). Pure presentation: renders the parsed
// report model (report-parser.ts) as a branded, consulting-style document —
// cover header, section cards, property blocks, comparison tables, star
// scorecard, highlighted final decision and a next-steps checklist. Every piece
// of text comes verbatim from the backend report; nothing is computed here.

import { Building2, Check } from "lucide-react";

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
  PropertySection,
  ReportBlock,
  ReportModel,
  ReportSubsection,
} from "./report-parser";

// Labels whose values read as a status — rendered as a tinted badge.
const BADGE_LABELS = new Set([
  "status",
  "risk level",
  "verdict",
  "growth potential",
  "demand",
  "negotiation strength",
  "investment rating",
  "valuation status",
]);

function Stars({ value }: { value: string }) {
  return (
    <span className="text-sm tracking-[0.2em] text-primary" aria-label={value}>
      {value}
    </span>
  );
}

function PairValue({ label, value }: { label: string; value: string }) {
  if (/^[★☆\s]+$/.test(value)) return <Stars value={value} />;
  if (BADGE_LABELS.has(label.toLowerCase())) {
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
  return <span className="text-sm font-medium text-foreground">{value}</span>;
}

function BulletItem({ marker, text }: { marker: string; text: string }) {
  return (
    <li className="flex items-start gap-2.5 text-sm leading-relaxed">
      {marker === "✓" ? (
        <Check className="mt-0.5 size-4 shrink-0 text-primary" />
      ) : marker === "□" || marker === "☐" ? (
        <span className="mt-0.5 size-4 shrink-0 rounded-[4px] border-2 border-primary/40" />
      ) : marker === "✗" ? (
        <span className="mt-0.5 shrink-0 text-sm text-red-600">✗</span>
      ) : (
        <span className="mt-1.5 size-1.5 shrink-0 rounded-full bg-primary/70" />
      )}
      <span>{text}</span>
    </li>
  );
}

function BlockTable({
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
              {row.map((cell, cellIndex) => (
                <TableCell
                  key={cellIndex}
                  className={cn(
                    "text-sm",
                    cellIndex === 0
                      ? "font-medium text-muted-foreground"
                      : "text-center"
                  )}
                >
                  {/^[★☆\s]+$/.test(cell) ? <Stars value={cell} /> : cell}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

function Blocks({ blocks }: { blocks: ReportBlock[] }) {
  return (
    <div className="space-y-3">
      {blocks.map((block, index) => {
        switch (block.type) {
          case "pairs":
            return (
              <dl key={index} className="space-y-2">
                {block.pairs.map((pair, pairIndex) => (
                  <div
                    key={pairIndex}
                    className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-0.5 border-b border-dashed border-border/70 pb-1.5 last:border-0 last:pb-0"
                  >
                    <dt className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                      {pair.label}
                    </dt>
                    <dd className="text-right">
                      <PairValue label={pair.label} value={pair.value} />
                    </dd>
                  </div>
                ))}
              </dl>
            );
          case "bullets":
            return (
              <ul key={index} className="space-y-1.5">
                {block.items.map((item, itemIndex) => (
                  <BulletItem key={itemIndex} {...item} />
                ))}
              </ul>
            );
          case "table":
            return (
              <BlockTable
                key={index}
                header={block.header}
                rows={block.rows}
              />
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
          case "text": {
            // Text directly under a standalone label is that label's value.
            const emphasized = blocks[index - 1]?.type === "label";
            return (
              <div key={index} className="space-y-1">
                {block.lines.map((line, lineIndex) => (
                  <p
                    key={lineIndex}
                    className={cn(
                      "text-sm leading-relaxed",
                      emphasized && "font-medium text-foreground"
                    )}
                  >
                    {line}
                  </p>
                ))}
              </div>
            );
          }
        }
      })}
    </div>
  );
}

function SubsectionCard({ subsection }: { subsection: ReportSubsection }) {
  return (
    <div className="report-avoid-break rounded-xl border bg-card p-4 shadow-sm">
      {subsection.title && (
        <h4 className="mb-3 flex items-center gap-2 border-b pb-2 text-sm font-semibold text-foreground">
          {subsection.icon && (
            <span aria-hidden className="text-base">
              {subsection.icon}
            </span>
          )}
          {subsection.title}
        </h4>
      )}
      <Blocks blocks={subsection.blocks} />
    </div>
  );
}

function PropertyBlock({ section }: { section: PropertySection }) {
  // Surface the headline facts (configuration + asking price) from the
  // Property Overview pairs as header chips — values shown verbatim.
  const overviewPairs =
    section.subsections
      .find((sub) => /overview/i.test(sub.title))
      ?.blocks.flatMap((block) => (block.type === "pairs" ? block.pairs : [])) ??
    [];
  const chips = overviewPairs.filter((pair) =>
    /^(configuration|asking price)$/i.test(pair.label)
  );

  return (
    <section className="space-y-4">
      <header className="report-avoid-break overflow-hidden rounded-xl bg-primary text-primary-foreground shadow-sm">
        <div className="p-5">
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-primary-foreground/80">
            🏠 Property {section.number}
          </p>
          <h3 className="mt-1 font-heading text-xl font-semibold">
            {section.name || `Property ${section.number}`}
          </h3>
          {section.location && (
            <p className="mt-0.5 text-sm text-primary-foreground/85">
              {section.location}
            </p>
          )}
          {chips.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-2">
              {chips.map((chip) => (
                <span
                  key={chip.label}
                  className="rounded-full bg-primary-foreground/15 px-3 py-1 text-xs font-medium"
                >
                  {chip.value}
                </span>
              ))}
            </div>
          )}
        </div>
      </header>

      <div className="grid gap-4 sm:grid-cols-2">
        {section.subsections.map((subsection, index) => (
          <SubsectionCard key={index} subsection={subsection} />
        ))}
      </div>
    </section>
  );
}

function GenericSectionCard({ section }: { section: GenericSection }) {
  const title = section.title.toLowerCase();
  const isDecision =
    title.includes("final investment decision") ||
    title.includes("best overall");

  return (
    <section
      className={cn(
        "report-avoid-break rounded-xl border p-5 shadow-sm",
        isDecision
          ? "border-primary/40 bg-primary/5"
          : "bg-card"
      )}
    >
      <h3
        className={cn(
          "mb-4 flex items-center gap-2 border-b pb-3 font-heading text-base font-semibold",
          isDecision && "border-primary/20 text-primary"
        )}
      >
        {section.icon && (
          <span aria-hidden className="text-lg">
            {section.icon}
          </span>
        )}
        {section.title}
      </h3>
      <Blocks blocks={section.blocks} />
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
                Generated for
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

export function ReportDocument({ model }: { model: ReportModel }) {
  return (
    <div className="report-document mx-auto w-full max-w-3xl space-y-5">
      <DocumentHeader model={model} />
      {model.sections.map((section, index) =>
        section.kind === "property" ? (
          <PropertyBlock key={index} section={section} />
        ) : (
          <GenericSectionCard key={index} section={section} />
        )
      )}
      <footer className="border-t pt-3 text-center text-xs text-muted-foreground">
        EstateMind · Property Investment Report · Generated by the EstateMind
        analysis platform
      </footer>
    </div>
  );
}
