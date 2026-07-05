import {
  Bot,
  GitCompareArrows,
  LineChart,
  Search,
  ShieldCheck,
  TrendingUp,
} from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

// Features surfaced by the existing backend (01_Project.md core features).
const features = [
  {
    icon: Search,
    title: "AI Property Search",
    description:
      "Find matching properties using natural language and hybrid search.",
  },
  {
    icon: Bot,
    title: "AI Chat Assistant",
    description:
      "Ask questions and get instant, context-aware property guidance.",
  },
  {
    icon: LineChart,
    title: "Price Prediction",
    description:
      "Estimate property prices with machine-learning models trained on market data.",
  },
  {
    icon: GitCompareArrows,
    title: "Property Comparison",
    description:
      "Compare properties side by side with AI-backed recommendations.",
  },
  {
    icon: TrendingUp,
    title: "Investment Insights",
    description:
      "Rental analysis, valuation, and future growth in a single view.",
  },
  {
    icon: ShieldCheck,
    title: "Risk Analysis",
    description:
      "Understand investment risks before you commit to a property.",
  },
];

export function Features() {
  return (
    <section className="mx-auto max-w-6xl px-4 py-20 lg:px-6">
      <div className="mx-auto mb-12 max-w-2xl text-center">
        <h2 className="font-heading text-3xl font-semibold tracking-tight">
          Everything you need to buy smarter
        </h2>
        <p className="mt-3 text-muted-foreground text-pretty">
          EstateMind brings AI search, analysis, and comparison together so you
          can make confident decisions.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {features.map(({ icon: Icon, title, description }) => (
          <Card key={title}>
            <CardHeader>
              <span className="flex size-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
                <Icon className="size-5" />
              </span>
              <CardTitle className="mt-2 text-base">{title}</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground">
              {description}
            </CardContent>
          </Card>
        ))}
      </div>
    </section>
  );
}
