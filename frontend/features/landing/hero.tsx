import Link from "next/link";
import { ArrowRight, Building2, ChevronDown, LogIn, Sparkles } from "lucide-react";

import { Button } from "@/components/ui/button";

// Fullscreen scenic hero. A premium real-estate background image (served from
// the local public/ folder so no per-domain Next image config is required) sits
// under a dark gradient overlay for readability. All content is white and
// centered.
const HERO_IMAGE = "/images/landing/hero.jpg";

export function Hero() {
  return (
    <section className="relative flex min-h-[calc(100dvh-4rem)] items-center justify-center overflow-hidden">
      {/* Scenic background image. */}
      <div
        aria-hidden
        className="absolute inset-0 bg-slate-900 bg-cover bg-center"
        style={{ backgroundImage: `url(${HERO_IMAGE})` }}
      />
      {/* Dark overlay for contrast. */}
      <div
        aria-hidden
        className="absolute inset-0 bg-gradient-to-b from-black/70 via-black/55 to-black/75"
      />

      <div className="relative z-10 mx-auto flex max-w-4xl flex-col items-center gap-6 px-4 py-24 text-center text-white lg:px-6">
        {/* EstateMind Copilot branding. */}
        <span className="inline-flex items-center gap-2 rounded-full border border-white/20 bg-white/10 px-4 py-1.5 text-sm font-medium backdrop-blur">
          <Building2 className="size-4" />
          EstateMind Copilot
        </span>

        <h1 className="max-w-3xl font-heading text-4xl font-semibold tracking-tight text-balance sm:text-5xl lg:text-6xl">
          Welcome to EstateMind Copilot
        </h1>

        <p className="inline-flex items-center gap-2 text-lg font-medium text-white/90 sm:text-xl">
          <Sparkles className="size-5" />
          AI-Powered Real Estate Intelligence
        </p>

        <p className="max-w-2xl text-base text-white/80 text-pretty sm:text-lg">
          Search, compare, analyze and invest with confidence — price prediction,
          valuation, rental analysis and investment advice, all in one place.
        </p>

        {/* Primary actions: Sign In, Sign Up, Learn More. */}
        <div className="mt-2 flex w-full flex-col gap-3 sm:w-auto sm:flex-row">
          <Button asChild size="lg">
            <Link href="/signup">
              Sign Up
              <ArrowRight />
            </Link>
          </Button>
          <Button asChild size="lg" variant="secondary">
            <Link href="/login">
              <LogIn />
              Sign In
            </Link>
          </Button>
          <Button
            asChild
            size="lg"
            variant="ghost"
            className="text-white hover:bg-white/10 hover:text-white"
          >
            <a href="#features">Learn More</a>
          </Button>
        </div>
      </div>

      {/* Scroll-to-features indicator. */}
      <a
        href="#features"
        aria-label="Scroll to features"
        className="absolute inset-x-0 bottom-6 z-10 mx-auto flex w-fit items-center justify-center rounded-full border border-white/20 bg-white/10 p-2 text-white backdrop-blur transition-colors hover:bg-white/20"
      >
        <ChevronDown className="size-5 animate-bounce" />
      </a>
    </section>
  );
}
