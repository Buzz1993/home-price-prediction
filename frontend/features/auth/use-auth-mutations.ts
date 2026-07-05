"use client";

// TanStack Query mutations for auth. On success they store the session (context
// + localStorage) and redirect to the dashboard, matching the Streamlit flow
// Login/Sign Up → Dashboard.

import { useRouter } from "next/navigation";
import { useMutation } from "@tanstack/react-query";

import { login, signup } from "@/services/auth-service";
import { useAuth } from "@/components/providers/auth-provider";
import type { LoginPayload, SignupPayload } from "@/types/auth";

export function useLogin() {
  const router = useRouter();
  const { setSession } = useAuth();

  return useMutation({
    mutationFn: (payload: LoginPayload) => login(payload),
    onSuccess: (data) => {
      setSession(data);
      router.push("/dashboard");
    },
  });
}

export function useSignup() {
  const router = useRouter();
  const { setSession } = useAuth();

  return useMutation({
    mutationFn: (payload: SignupPayload) => signup(payload),
    onSuccess: (data) => {
      setSession(data);
      router.push("/dashboard");
    },
  });
}
