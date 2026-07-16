"use client";

// Card renderers for the richer, per-property analysis payloads: negotiation
// strategy and investment advisor. Decision-first presentation (Phase 15.20)
// in the executive-report hierarchy: Decision Summary (the verbatim backend
// verdict / negotiation_power) → Why checklist (verbatim positives / talking
// points) → metrics → the verbatim strategy / investor fit as the recommended
// action. Shared by the AI Analysis workspace and the chat payloads with
// unchanged signatures, and every value shown comes verbatim from the backend.

import {
  Brain,
  CheckCircle2,
  Handshake,
  Percent,
  ShieldAlert,
  Trophy,
} from "lucide-react";

import type { AdvisorRow, NegotiationRow } from "@/types/dashboard";
import { toneKey, toneRank } from "@/lib/value-tone";
import { formatCr, formatPercent, splitList } from "./format";
import { CompactPropertyHeader } from "@/features/analysis/property-header";
import {
  CalloutCard,
  MetricCard,
  MetricExplainer,
  SectionLabel,
  hasValue,
} from "@/features/analysis/ui/analysis-ui";
import {
  DecisionSummary,
  RecommendationBar,
  WhyCard,
  ratingStars,
} from "@/features/analysis/ui/decision-summary";
import {
  ExecutiveSummary,
  usePropertyName,
} from "@/features/analysis/ui/executive-summary";

// The backend discount is a string like "5%"; the number is only used to
// compare the existing backend values — the verbatim string is displayed.
function parseDiscount(value: unknown): number | null {
  if (!hasValue(value)) return null;
  const n = Number(String(value).replace(/[^\d.-]/g, ""));
  return Number.isNaN(n) ? null : n;
}

export function NegotiationCards({ rows }: { rows: NegotiationRow[] }) {
  const resolveName = usePropertyName();

  // Executive comparison (Phase 15.21): with several properties analyzed, lead
  // with the one carrying the largest backend suggested discount; ties go to
  // the stronger backend negotiation_power wording (toneRank). Presentation
  // only — both signals are the backend's own.
  const contenders = rows
    .map((item) => ({
      item,
      discount: parseDiscount(item.suggested_discount_percent),
      powerRank: hasValue(item.negotiation_power)
        ? toneRank(String(item.negotiation_power))
        : -1,
    }))
    .filter((c) => c.discount !== null || c.powerRank >= 0);
  const winner =
    rows.length >= 2 && contenders.length >= 2
      ? contenders.reduce((best, c) =>
          (c.discount ?? -1) > (best.discount ?? -1) ||
          ((c.discount ?? -1) === (best.discount ?? -1) &&
            c.powerRank > best.powerRank)
            ? c
            : best
        )
      : null;

  return (
    <div className="space-y-4">
      {winner && (
        <>
          <ExecutiveSummary
            eyebrow="Better negotiation opportunity"
            id={winner.item.id}
            name={resolveName(winner.item.id)}
            badge={
              hasValue(winner.item.negotiation_power)
                ? `${winner.item.negotiation_power} negotiation power`
                : undefined
            }
            statement={
              hasValue(winner.item.suggested_discount_percent)
                ? `Largest suggested discount — ${formatPercent(winner.item.suggested_discount_percent)} off the asking price`
                : undefined
            }
            stat={
              hasValue(winner.item.target_price)
                ? {
                    label: "Target offer",
                    value: formatCr(winner.item.target_price),
                    sub: "Recommended offer price",
                  }
                : undefined
            }
            reasons={splitList(winner.item.talking_points).slice(0, 4)}
            contenders={contenders.map((c) => ({
              id: c.item.id,
              name: resolveName(c.item.id),
              status: hasValue(c.item.negotiation_power)
                ? c.item.negotiation_power
                : undefined,
              display: hasValue(c.item.suggested_discount_percent)
                ? `${formatPercent(c.item.suggested_discount_percent)} discount`
                : undefined,
              isWinner: c === winner,
            }))}
          />
          <SectionLabel>Property breakdown</SectionLabel>
        </>
      )}

      {rows.map((item, index) => {
        // Talking points arrive as a comma-separated backend summary; splitting
        // them into a checklist is formatting only, the wording stays verbatim.
        const talkingPoints = splitList(item.talking_points);
        const power = hasValue(item.negotiation_power)
          ? String(item.negotiation_power)
          : null;
        const tone = power ? toneKey(power) : "neutral";

        return (
          <section key={item.id} className="space-y-2">
            <CompactPropertyHeader
              id={item.id}
              index={index}
              analysisLabel="Negotiation Strategy"
              icon={Handshake}
            />

            {/* 1 — the answer first: the backend's own negotiation power,
                with the backend target price as the headline stat */}
            {power && (
              <DecisionSummary
                eyebrow="Negotiation power"
                headline={power}
                icon={Handshake}
                tone={tone}
                tagline={
                  hasValue(item.suggested_discount_percent)
                    ? `Suggested discount of ${formatPercent(item.suggested_discount_percent)} off the asking price`
                    : undefined
                }
                stat={
                  hasValue(item.target_price)
                    ? {
                        label: "Target offer",
                        value: formatCr(item.target_price),
                        sub: "Recommended offer price",
                      }
                    : undefined
                }
              />
            )}

            {/* 2 — why: the backend's own talking points as leverage */}
            {power && (
              <WhyCard
                title={`Why ${power} negotiation power?`}
                reasons={talkingPoints}
              />
            )}

            {/* 3 — metrics */}
            <div className="grid grid-cols-2 gap-2">
              {hasValue(item.target_price) && (
                <MetricCard
                  label="Target price"
                  value={formatCr(item.target_price)}
                  sub="Recommended offer"
                  icon={Handshake}
                  highlight
                />
              )}
              {hasValue(item.suggested_discount_percent) && (
                <MetricCard
                  label="Suggested discount"
                  value={formatPercent(item.suggested_discount_percent)}
                  sub="Off the asking price"
                  icon={Percent}
                />
              )}
            </div>

            {/* 4 — what to do: the backend's own strategy */}
            {hasValue(item.strategy) && (
              <RecommendationBar
                tone={tone === "neutral" ? "positive" : tone}
                icon={Handshake}
                title="Negotiation strategy"
              >
                {item.strategy}
              </RecommendationBar>
            )}
          </section>
        );
      })}

      <MetricExplainer
        items={[
          {
            term: "Target price",
            meaning:
              "The offer price the negotiation engine recommends, based on the property's pricing position and market signals.",
          },
          {
            term: "Negotiation power",
            meaning:
              "How much leverage you hold as a buyer — stronger leverage means more room to push toward the target price.",
          },
          {
            term: "How to use it",
            meaning:
              "Open below the target price and use the talking points as evidence. The suggested discount is a guide, not a limit.",
          },
        ]}
      />
    </div>
  );
}

export function AdvisorCards({ rows }: { rows: AdvisorRow[] }) {
  const resolveName = usePropertyName();

  // Executive comparison (Phase 15.21): with several properties analyzed, lead
  // with the one whose backend verdict wording grades best — the star
  // restatement of the verdict first (Excellent > Good > …), the tone of the
  // wording as tie-break. Presentation of the backend's own verdicts only.
  const contenders = rows
    .filter((item) => hasValue(item.verdict))
    .map((item) => {
      const verdict = String(item.verdict);
      return {
        item,
        verdict,
        rank: (ratingStars(verdict) ?? 0) * 10 + toneRank(verdict),
      };
    });
  const winner =
    rows.length >= 2 && contenders.length >= 2
      ? contenders.reduce((best, c) => (c.rank > best.rank ? c : best))
      : null;

  return (
    <div className="space-y-4">
      {winner && (
        <>
          <ExecutiveSummary
            eyebrow="EstateMind recommendation"
            id={winner.item.id}
            name={resolveName(winner.item.id)}
            badge={winner.verdict}
            stars={ratingStars(winner.verdict)}
            statement="Best overall verdict among the compared properties"
            reasons={splitList(winner.item.positives).slice(0, 4)}
            contenders={contenders.map((c) => ({
              id: c.item.id,
              name: resolveName(c.item.id),
              status: c.verdict,
              isWinner: c === winner,
            }))}
          />
          <SectionLabel>Property breakdown</SectionLabel>
        </>
      )}

      {rows.map((item, index) => {
        // Comma-separated backend summaries → checklist lines (formatting only).
        const positives = splitList(item.positives);
        const verdict = hasValue(item.verdict) ? String(item.verdict) : null;
        const tone = verdict ? toneKey(verdict) : "neutral";

        return (
          <section key={item.id} className="space-y-2">
            <CompactPropertyHeader
              id={item.id}
              index={index}
              analysisLabel="Investment Advisor"
              icon={Brain}
            />

            {/* 1 — the answer first: the backend's own verdict, shown once */}
            {verdict && (
              <DecisionSummary
                eyebrow="Investment verdict"
                headline={verdict}
                icon={Trophy}
                tone={tone}
                stars={ratingStars(verdict)}
                tagline="Overall investment assessment"
              />
            )}

            {/* 2 — why */}
            {verdict && (
              <WhyCard title={`Why ${verdict}?`} reasons={positives} />
            )}

            {/* 3 — the trade-off side */}
            {hasValue(item.risks) && (
              <CalloutCard tone="red" icon={ShieldAlert} title="Weaknesses">
                {item.risks}
              </CalloutCard>
            )}

            {/* 4 — what to do: the backend's own investor fit */}
            {hasValue(item.suitable_for) && (
              <RecommendationBar tone={tone} icon={CheckCircle2}>
                Best suited for {item.suitable_for} investors.
              </RecommendationBar>
            )}
          </section>
        );
      })}

      <MetricExplainer
        items={[
          {
            term: "Verdict",
            meaning:
              "The advisor engine's overall judgement of the property, weighing valuation, risk, growth, rental and negotiation signals together.",
          },
          {
            term: "Suitable for",
            meaning:
              "The investor profile this property fits best — e.g. conservative buyers who prioritize safety, or aggressive investors chasing growth.",
          },
          {
            term: "How to use it",
            meaning:
              "Match the 'suitable for' profile against your own risk appetite, then dig into the individual analyses behind any strength or weakness that matters to you.",
          },
        ]}
      />
    </div>
  );
}
