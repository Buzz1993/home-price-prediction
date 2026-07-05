import type { Metadata } from "next";

import { LoginForm } from "@/features/auth/login-form";

export const metadata: Metadata = {
  title: "Log in · EstateMind",
};

export default function LoginPage() {
  return <LoginForm />;
}
