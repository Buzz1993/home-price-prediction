"use client";

// Profile page data hooks. A read-only profile query plus a logout action.
// Report history and chat history moved to their own pages (Phase 18.9) — the
// Reports page reads the local report store and the Chat History page reads the
// workspace conversations, so their old queries were removed with them.

import { useRouter } from "next/navigation";
import { useQuery, useQueryClient } from "@tanstack/react-query";

import { useAuth } from "@/components/providers/auth-provider";
import { getProfile } from "@/services/profile-service";

export function useProfile() {
  return useQuery({ queryKey: ["profile"], queryFn: getProfile });
}

// Logout clears the client-side session (see auth-provider), drops any cached
// user data, and returns to the Login page. No backend auth call is made — the
// task scopes this page to the GET /profile, /chat-history and /reports endpoints,
// and the session is stored locally.
export function useLogout() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const { clearSession } = useAuth();

  return () => {
    clearSession();
    queryClient.clear();
    router.push("/login");
  };
}
