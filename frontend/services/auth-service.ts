// Auth API calls. Thin wrappers over the documented backend endpoints — no
// business logic lives here.

import { apiRequest } from "@/lib/api-client";
import type {
  AuthResponse,
  LoginPayload,
  SignupPayload,
  User,
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
