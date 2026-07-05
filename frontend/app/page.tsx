import { LandingNavbar } from "@/features/landing/landing-navbar";
import { Hero } from "@/features/landing/hero";
import { Features } from "@/features/landing/features";
import { HowItWorks } from "@/features/landing/how-it-works";
import { CallToAction } from "@/features/landing/call-to-action";
import { LandingFooter } from "@/features/landing/landing-footer";

export default function Home() {
  return (
    <div className="flex min-h-dvh flex-col">
      <LandingNavbar />
      <main className="flex-1">
        <Hero />
        <Features />
        <HowItWorks />
        <CallToAction />
      </main>
      <LandingFooter />
    </div>
  );
}
