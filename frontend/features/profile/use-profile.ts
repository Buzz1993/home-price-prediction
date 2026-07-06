"use client";

// Profile page data hooks. Three read-only queries (profile, chat history,
// reports) plus a logout action. All persistence lives in the backend; these only
// track request state and cache the responses.

import { useRouter } from "next/navigation";
import { useQuery, useQueryClient } from "@tanstack/react-query";

import { useAuth } from "@/components/providers/auth-provider";
import {
  getChatHistory,
  getProfile,
  getReports,
} from "@/services/profile-service";

export function useProfile() {
  return useQuery({ queryKey: ["profile"], queryFn: getProfile });
}

export function useChatHistory() {
  return useQuery({ queryKey: ["chat-history"], queryFn: getChatHistory });
}

export function useReports() {
  return useQuery({ queryKey: ["reports"], queryFn: getReports });
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
