import {
  Banknote,
  Briefcase,
  FileText,
  Gauge,
  GitCompareArrows,
  LineChart,
  Search,
  ShieldCheck,
  TrendingUp,
} from "lucide-react";

import { FeatureCard } from "@/features/landing/feature-card";

// Existing backend capabilities only (04_UI.md / 05_Features.md) — no invented
// features. Rendered through the single reusable FeatureCard.
const features = [
  {
    icon: Search,
    title: "AI Property Search",
    description:
      "Find matching properties using natural language and hybrid search.",
  },
  {
    icon: GitCompareArrows,
    title: "Property Comparison",
    description:
      "Compare properties side by side with AI-backed recommendations.",
  },
  {
    icon: LineChart,
    title: "Price Prediction",
    description:
      "Estimate property prices with machine-learning models trained on market data.",
  },
  {
    icon: Banknote,
    title: "Rental Analysis",
    description:
      "Understand rental estimates, yield and ROI before you invest.",
  },
  {
    icon: Gauge,
    title: "Property Valuation",
    description:
      "See how a property is valued against the current market.",
  },
  {
    icon: ShieldCheck,
    title: "Risk Analysis",
    description:
      "Understand investment risks before you commit to a property.",
  },
  {
    icon: TrendingUp,
    title: "Future Growth Analysis",
    description:
      "Gauge a property's future appreciation potential.",
  },
  {
    icon: Briefcase,
    title: "Investment Advisor",
    description:
      "Get a complete, backend-driven investment recommendation.",
  },
  {
    icon: FileText,
    title: "AI Report Generation",
    description:
      "Generate and share a professional property report in one flow.",
  },
];

export function Features() {
  return (
    <section id="features" className="scroll-mt-16 bg-background">
      <div className="mx-auto max-w-6xl px-4 py-20 lg:px-6 lg:py-28">
        <div className="mx-auto mb-12 max-w-2xl text-center">
          <h2 className="font-heading text-3xl font-semibold tracking-tight sm:text-4xl">
            Everything you need to buy smarter
          </h2>
          <p className="mt-3 text-muted-foreground text-pretty">
            EstateMind brings AI search, analysis, and comparison together so you
            can make confident decisions.
          </p>
        </div>

        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {features.map((feature) => (
            <FeatureCard key={feature.title} {...feature} />
          ))}
        </div>
      </div>
    </section>
  );
}
