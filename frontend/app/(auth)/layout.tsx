import Link from "next/link";
import { Building2 } from "lucide-react";

// Centered layout for the public auth pages (login, signup).
export default function AuthLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="flex min-h-dvh flex-col items-center justify-center gap-6 bg-muted/40 p-4">
      <Link
        href="/"
        className="flex items-center gap-2 font-heading text-lg font-semibold"
      >
        <span className="flex size-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
          <Building2 className="size-5" />
        </span>
        EstateMind
      </Link>
      {children}
    </div>
  );
}
