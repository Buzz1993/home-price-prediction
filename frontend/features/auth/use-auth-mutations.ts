"use client";

// TanStack Query mutations for auth. On success they store the session (context
// + localStorage) and redirect to the dashboard, matching the Streamlit flow
// Login/Sign Up → Dashboard.
// Phase 27: Added useForgotPassword and useResetPassword.

import { useRouter } from "next/navigation";
import { useMutation } from "@tanstack/react-query";

import { login, signup, forgotPassword, resetPassword } from "@/services/auth-service";
import { useAuth } from "@/components/providers/auth-provider";
import type { LoginPayload, SignupPayload, ForgotPasswordPayload, ResetPasswordPayload } from "@/types/auth";

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

export function useForgotPassword() {
  return useMutation({
    mutationFn: (payload: ForgotPasswordPayload) => forgotPassword(payload),
  });
}

export function useResetPassword() {
  const router = useRouter();

  return useMutation({
    mutationFn: (payload: ResetPasswordPayload) => resetPassword(payload),
    onSuccess: () => {
      // Redirect to login after successful password reset
      setTimeout(() => {
        router.push("/login");
      }, 2000);
    },
  });
}
