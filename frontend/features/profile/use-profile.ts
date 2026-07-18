"use client";

// Profile page data hooks. A read-only profile query plus a logout action.
// Report history and chat history moved off this page (Phase 18.9) — the
// Reports page reads the local report store and the Dashboard Recent section
// reads the workspace conversations, so their old queries were removed.
//
// Phase 18.18: GET /profile is authenticated — the query sends the stored
// Bearer token and only runs once a session exists, so the page can never
// show a user other than the one who logged in.

import { useRouter } from "next/navigation";
import { useQuery, useQueryClient } from "@tanstack/react-query";

import { useAuth } from "@/components/providers/auth-provider";
import { logout as logoutRequest } from "@/services/auth-service";
import { getProfile } from "@/services/profile-service";

export function useProfile() {
  const { token, isReady } = useAuth();

  return useQuery({
    queryKey: ["profile", token],
    queryFn: () => getProfile(token),
    enabled: isReady && Boolean(token),
  });
}

// Logout ends the backend session (POST /logout invalidates the token), then
// clears the client-side session (see auth-provider), drops any cached user
// data, and returns to the Login page.
export function useLogout() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const { token, clearSession } = useAuth();

  return () => {
    // Fire-and-forget: the client session is cleared regardless, so a network
    // failure here never traps the user in a logged-in UI.
    if (token) logoutRequest(token).catch(() => {});
    clearSession();
    queryClient.clear();
    router.push("/login");
  };
}
