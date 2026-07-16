import Link from "next/link";
import { Building2 } from "lucide-react";

import { cn } from "@/lib/utils";

export function Brand({ className }: { className?: string }) {
  return (
    <Link
      href="/dashboard"
      className={cn(
        "flex items-center gap-2 font-heading text-lg font-semibold",
        className
      )}
    >
      <span className="bg-brand-gradient shadow-brand-glow flex size-8 items-center justify-center rounded-lg text-primary-foreground">
        <Building2 className="size-5" />
      </span>
      EstateMind
    </Link>
  );
}
