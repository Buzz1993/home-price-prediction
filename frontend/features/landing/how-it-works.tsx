import {
  BarChart3,
  FileText,
  GitCompareArrows,
  Search,
  Target,
} from "lucide-react";

import { WorkflowStep } from "@/features/landing/workflow-step";

// User journey (04_UI.md navigation flow):
// Search → Compare → Analyze → Generate Report → Make Better Decisions.
const steps = [
  {
    step: 1,
    icon: Search,
    title: "Search",
    description: "Describe what you're looking for and let AI surface the best matches.",
  },
  {
    step: 2,
    icon: GitCompareArrows,
    title: "Compare",
    description: "Stage properties and compare them side by side.",
  },
  {
    step: 3,
    icon: BarChart3,
    title: "Analyze",
    description: "Run price, rental, valuation, risk and growth analysis.",
  },
  {
    step: 4,
    icon: FileText,
    title: "Generate Report",
    description: "Create a professional AI report and share it in one flow.",
  },
  {
    step: 5,
    icon: Target,
    title: "Make Better Decisions",
    description: "Invest with confidence backed by clear, data-driven insights.",
  },
];

export function HowItWorks() {
  return (
    <section className="border-y bg-muted/40">
      <div className="mx-auto max-w-6xl px-4 py-20 lg:px-6 lg:py-28">
        <div className="mx-auto mb-12 max-w-2xl text-center">
          <h2 className="font-heading text-3xl font-semibold tracking-tight sm:text-4xl">
            How EstateMind works
          </h2>
          <p className="mt-3 text-muted-foreground text-pretty">
            From search to decision in a few simple steps.
          </p>
        </div>

        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
          {steps.map((step) => (
            <WorkflowStep key={step.step} {...step} />
          ))}
        </div>
      </div>
    </section>
  );
}
