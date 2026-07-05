// "How It Works" mirrors the user journey in 01_Project.md:
// Search → Analyze → Decide.
const steps = [
  {
    step: "01",
    title: "Search properties",
    description:
      "Describe what you're looking for and let AI surface the best matches.",
  },
  {
    step: "02",
    title: "Analyze with AI",
    description:
      "Get price prediction, valuation, rental analysis, and risk insights.",
  },
  {
    step: "03",
    title: "Decide & report",
    description:
      "Compare options, generate a report, and share it — all in one flow.",
  },
];

export function HowItWorks() {
  return (
    <section className="border-y bg-muted/40">
      <div className="mx-auto max-w-6xl px-4 py-20 lg:px-6">
        <div className="mx-auto mb-12 max-w-2xl text-center">
          <h2 className="font-heading text-3xl font-semibold tracking-tight">
            How it works
          </h2>
          <p className="mt-3 text-muted-foreground text-pretty">
            From search to decision in three simple steps.
          </p>
        </div>

        <div className="grid gap-8 sm:grid-cols-3">
          {steps.map(({ step, title, description }) => (
            <div key={step} className="flex flex-col gap-3">
              <span className="font-heading text-3xl font-semibold text-primary/30">
                {step}
              </span>
              <h3 className="font-heading text-lg font-semibold">{title}</h3>
              <p className="text-sm text-muted-foreground text-pretty">
                {description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
