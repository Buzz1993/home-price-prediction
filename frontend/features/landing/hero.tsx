import Link from "next/link";
import { ArrowRight, Sparkles } from "lucide-react";

import { Button } from "@/components/ui/button";

export function Hero() {
  return (
    <section className="mx-auto flex max-w-6xl flex-col items-center gap-6 px-4 py-20 text-center lg:px-6 lg:py-28">
      <span className="inline-flex items-center gap-2 rounded-full border bg-muted/50 px-3 py-1 text-xs font-medium text-muted-foreground">
        <Sparkles className="size-3.5" />
        AI-powered real estate copilot
      </span>

      <h1 className="max-w-3xl font-heading text-4xl font-semibold tracking-tight text-balance sm:text-5xl lg:text-6xl">
        Make smarter property decisions with AI
      </h1>

      <p className="max-w-2xl text-lg text-muted-foreground text-pretty">
        Search, compare, and analyze residential properties using intelligent
        insights — price prediction, valuation, rental analysis, and investment
        advice, all in one place.
      </p>

      <div className="flex flex-col gap-3 sm:flex-row">
        <Button asChild size="lg">
          <Link href="/signup">
            Get started free
            <ArrowRight />
          </Link>
        </Button>
        <Button asChild variant="outline" size="lg">
          <Link href="/login">Log in</Link>
        </Button>
      </div>
    </section>
  );
}
