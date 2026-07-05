import type { Metadata } from "next";

import { SignupForm } from "@/features/auth/signup-form";

export const metadata: Metadata = {
  title: "Sign up · EstateMind",
};

export default function SignupPage() {
  return <SignupForm />;
}
