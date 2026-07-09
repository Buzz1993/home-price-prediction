import Link from "next/link";
import { ArrowRight, LogIn } from "lucide-react";

import { Button } from "@/components/ui/button";

export function CallToAction() {
  return (
    <section className="mx-auto max-w-6xl px-4 py-20 lg:px-6 lg:py-28">
      <div className="group relative flex flex-col items-center gap-6 overflow-hidden rounded-3xl border bg-card px-6 py-16 text-center shadow-sm">
        {/* Subtle animated accent that eases in on hover. */}
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0 bg-gradient-to-br from-primary/5 via-transparent to-primary/10 opacity-0 transition-opacity duration-500 group-hover:opacity-100"
        />

        <div className="relative z-10 flex flex-col items-center gap-6">
          <h2 className="max-w-2xl font-heading text-3xl font-semibold tracking-tight text-balance sm:text-4xl">
            Ready to find your next property?
          </h2>
          <p className="max-w-xl text-muted-foreground text-pretty">
            Create a free account and start exploring AI-powered property insights
            today.
          </p>

          <div className="flex w-full flex-col gap-3 sm:w-auto sm:flex-row">
            <Button asChild size="lg" className="transition-transform hover:-translate-y-0.5">
              <Link href="/signup">
                Sign Up
                <ArrowRight />
              </Link>
            </Button>
            <Button
              asChild
              size="lg"
              variant="outline"
              className="transition-transform hover:-translate-y-0.5"
            >
              <Link href="/login">
                <LogIn />
                Sign In
              </Link>
            </Button>
          </div>
        </div>
      </div>
    </section>
  );
}
