"use client";

// Route guard for the protected (dashboard) route group (Phase 18.18).
// Implements the documented authentication rule: protected pages require
// login, and unauthenticated visitors are redirected to the Login page.
//
// It waits for the AuthProvider to hydrate the stored session (isReady) before
// deciding, so a hard refresh of a logged-in user never bounces to /login.

import { useEffect } from "react";
import { useRouter } from "next/navigation";

import { useAuth } from "@/components/providers/auth-provider";

export function AuthGuard({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const { isAuthenticated, isReady } = useAuth();

  useEffect(() => {
    if (isReady && !isAuthenticated) {
      router.replace("/login");
    }
  }, [isReady, isAuthenticated, router]);

  // Until the stored session is read (or while redirecting), render nothing —
  // protected content must never flash for a logged-out visitor.
  if (!isReady || !isAuthenticated) return null;

  return <>{children}</>;
}
