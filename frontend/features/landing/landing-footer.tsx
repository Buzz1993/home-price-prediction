import { Building2 } from "lucide-react";

export function LandingFooter() {
  return (
    <footer className="border-t">
      <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-4 px-4 py-8 text-sm text-muted-foreground sm:flex-row lg:px-6">
        <div className="flex items-center gap-2 font-heading font-semibold text-foreground">
          <span className="flex size-7 items-center justify-center rounded-md bg-primary text-primary-foreground">
            <Building2 className="size-4" />
          </span>
          EstateMind
        </div>
        <p>AI-powered real estate copilot.</p>
      </div>
    </footer>
  );
}
