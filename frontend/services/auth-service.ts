// Auth API calls. Thin wrappers over the documented backend endpoints — no
// business logic lives here.

import { apiRequest } from "@/lib/api-client";
import type {
  AuthResponse,
  LoginPayload,
  SignupPayload,
  User,
  ForgotPasswordPayload,
  ResetPasswordPayload,
} from "@/types/auth";

export function login(payload: LoginPayload): Promise<AuthResponse> {
  return apiRequest<AuthResponse>("/login", {
    method: "POST",
    body: payload,
  });
}

export function signup(payload: SignupPayload): Promise<AuthResponse> {
  return apiRequest<AuthResponse>("/signup", {
    method: "POST",
    body: payload,
  });
}

export function getProfile(token: string): Promise<User> {
  return apiRequest<User>("/profile", { token });
}

// POST /logout — invalidates the session token on the backend (Phase 18.18).
export function logout(token: string): Promise<void> {
  return apiRequest<void>("/logout", { method: "POST", token });
}

// POST /forgot-password — Phase 27: request password reset email.
export function forgotPassword(
  payload: ForgotPasswordPayload
): Promise<{ success: boolean; message: string }> {
  return apiRequest<{ success: boolean; message: string }>(
    "/forgot-password",
    {
      method: "POST",
      body: payload,
    }
  );
}

// POST /reset-password — Phase 27: reset password with token.
export function resetPassword(
  payload: ResetPasswordPayload
): Promise<{ success: boolean; message: string }> {
  return apiRequest<{ success: boolean; message: string }>("/reset-password", {
    method: "POST",
    body: payload,
  });
}
