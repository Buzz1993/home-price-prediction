import Link from "next/link";
import { ArrowRight } from "lucide-react";

import { Button } from "@/components/ui/button";

export function CallToAction() {
  return (
    <section className="mx-auto max-w-6xl px-4 py-20 lg:px-6">
      <div className="flex flex-col items-center gap-6 rounded-2xl border bg-card px-6 py-14 text-center shadow-sm">
        <h2 className="max-w-2xl font-heading text-3xl font-semibold tracking-tight text-balance">
          Ready to find your next property?
        </h2>
        <p className="max-w-xl text-muted-foreground text-pretty">
          Create a free account and start exploring AI-powered property insights
          today.
        </p>
        <Button asChild size="lg">
          <Link href="/signup">
            Get started free
            <ArrowRight />
          </Link>
        </Button>
      </div>
    </section>
  );
}
