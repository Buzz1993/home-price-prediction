import Link from "next/link";
import { Building2 } from "lucide-react";

// Centered layout for the public auth pages (login, signup). Phase 18.2: a
// soft lavender-tinted gradient canvas with faint purple glow blobs so the
// floating auth card reads premium — decoration only, no functional changes.
export default function AuthLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="relative flex min-h-dvh flex-col items-center justify-center gap-6 overflow-hidden bg-background p-4">
      {/* Soft purple glow blobs behind the card. */}
      <div
        aria-hidden
        className="pointer-events-none absolute -top-32 left-1/2 size-[28rem] -translate-x-1/2 rounded-full bg-primary/10 blur-3xl"
      />
      <div
        aria-hidden
        className="pointer-events-none absolute -bottom-40 right-[12%] size-[24rem] rounded-full bg-brand-accent/10 blur-3xl"
      />

      <Link
        href="/"
        className="relative z-10 flex items-center gap-2 font-heading text-lg font-semibold"
      >
        <span className="bg-brand-gradient shadow-brand-glow flex size-8 items-center justify-center rounded-lg text-primary-foreground">
          <Building2 className="size-5" />
        </span>
        EstateMind
      </Link>
      <div className="relative z-10 flex w-full flex-col items-center">
        {children}
      </div>
    </div>
  );
}
