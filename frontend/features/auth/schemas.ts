import { z } from "zod";

// Validation schemas for the auth forms. Kept close to the documented backend
// contract (03_API.md, 04_UI.md).

export const loginSchema = z.object({
  email: z.string().min(1, "Email is required").email("Enter a valid email"),
  password: z.string().min(1, "Password is required"),
});

// Password policy (Phase 18.18) — mirrors the backend rules in
// src/api/auth_api.py so client validation never disagrees with the server:
// minimum 8 characters, one letter, one number, one special character.
// The message lists exactly the missing requirements.
const passwordSchema = z.string().superRefine((password, ctx) => {
  const missing: string[] = [];
  if (password.length < 8) missing.push("minimum 8 characters");
  if (!/[A-Za-z]/.test(password)) missing.push("one letter");
  if (!/\d/.test(password)) missing.push("one number");
  if (!/[^A-Za-z0-9]/.test(password)) missing.push("one special character");
  if (missing.length > 0) {
    ctx.addIssue({
      code: "custom",
      message: `Password must contain: ${missing.join(", ")}.`,
    });
  }
});

export const signupSchema = z
  .object({
    name: z.string().min(2, "Name must be at least 2 characters"),
    email: z.string().min(1, "Email is required").email("Enter a valid email"),
    password: passwordSchema,
    confirmPassword: z.string().min(1, "Confirm your password"),
  })
  .refine((data) => data.password === data.confirmPassword, {
    message: "Passwords do not match",
    path: ["confirmPassword"],
  });

export type LoginValues = z.infer<typeof loginSchema>;
export type SignupValues = z.infer<typeof signupSchema>;
