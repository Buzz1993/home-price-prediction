"use client";

// Phase 19 — First-time user onboarding tour.
// Custom spotlight implementation — no external library. Uses localStorage
// (estatemind:tour-seen) to record skip/finish so the tour never reappears
// automatically after the user has seen it once. `startTour()` is exposed
// via context so the Profile page can restart it on demand.

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";
import { createPortal } from "react-dom";
import { usePathname, useRouter } from "next/navigation";
import { Dialog as DialogPrimitive } from "radix-ui";
import {
  ArrowLeft,
  ArrowRight,
  CheckCircle2,
  X,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

// ─── Types ───────────────────────────────────────────────────────────────────

export type TourContextValue = {
  /** Manually start (or restart) the tour — used by the Profile page. */
  startTour: () => void;
};

const TourContext = createContext<TourContextValue>({ startTour: () => {} });

export function useTour(): TourContextValue {
  return useContext(TourContext);
}

// ─── Tour step definitions ────────────────────────────────────────────────────

type TourStep = {
  target: string;           // CSS selector for the spotlight target
  title: string;
  content: string;
  examples?: string[];      // optional bullet list shown below content
  footer?: string;          // optional closing line shown below the examples
};

const TOUR_STEPS: TourStep[] = [
  {
    target: '[data-tour="sidebar-nav"]',
    title: "Sidebar Navigation",
    content: "This is the main navigation. Everything starts from the Dashboard.",
  },
  {
    target: '[data-tour="dashboard-chat"]',
    title: "Dashboard",
    content: "Search for properties using natural language, just like chatting with an AI assistant.",
  },
  {
    target: '[data-tour="chat-input"]',
    title: "Search Box",
    content: "Describe your ideal property naturally.",
    examples: [
      "2 BHK in Thane under ₹2 Cr",
      "Luxury apartment near metro",
      "Property with gym and swimming pool",
    ],
  },
  {
    target: '[data-tour="dashboard-chat"]',
    title: "Property Cards",
    content:
      "Each recommendation includes price, project name, amenities, AI recommendation score, property details, Read More and View Listing.",
  },
  {
    target: '[data-tour="evaluation-tray"]',
    title: "Evaluation Tray",
    content:
      "Stage interesting properties here. You can later compare them or generate AI investment reports.",
  },
  {
    target: '[data-tour="property-map"]',
    title: "Property Map",
    content: "Visualize recommended properties on an interactive map.",
  },
  {
    target: '[data-tour="suggested-actions"]',
    title: "Suggested Next Steps",
    content:
      "EstateMind intelligently suggests useful follow-up actions based on your current conversation.",
  },
  {
    target: '[data-tour="nav-analysis"]',
    title: "AI Analysis",
    content:
      "Analyze any shortlisted property using EstateMind's AI engine. Run detailed investment insights before making your decision, including:",
    examples: [
      "Price Prediction",
      "Rental Yield Analysis",
      "Investment Analysis",
      "Future Growth Analysis",
      "Risk Assessment",
      "Negotiation Suggestions",
      "Property Valuation",
      "Investment Advisor",
    ],
    footer:
      "Use AI Analysis whenever you want a complete investment evaluation for a property.",
  },
  {
    target: '[data-tour="nav-reports"]',
    title: "Reports",
    content: "Generate detailed AI investment reports and export or share them.",
  },
  {
    target: '[data-tour="nav-saved"]',
    title: "Saved Properties",
    content: "Bookmark properties for future review.",
  },
  {
    target: '[data-tour="nav-compare"]',
    title: "Property Comparison",
    content:
      "Compare multiple shortlisted properties side-by-side before making an investment decision.",
  },
  {
    target: '[data-tour="nav-profile"]',
    title: "Profile",
    content: "Manage your account, preferences and application settings.",
  },
  {
    target: '[data-tour="recent-chats"]',
    title: "Recent Chats",
    content: "All of your previous AI conversations are saved here. You can:",
    examples: [
      "Resume old conversations",
      "Search chats instantly",
      "Pin important conversations",
      "Start a new chat anytime",
    ],
    footer: "Your conversation history is always available for future reference.",
  },
];

// ─── Constants ────────────────────────────────────────────────────────────────

const TOUR_SEEN_KEY = "estatemind:tour-seen";
const SPOTLIGHT_PAD = 8; // px padding around the highlighted element
const CARD_W = 340;      // approximate card width for positioning maths

// ─── Positioning helpers ──────────────────────────────────────────────────────

type CardPos = {
  top?: number; bottom?: number;
  left?: number; right?: number;
};

function getCardPosition(rect: DOMRect, vw: number, vh: number): CardPos {
  const CARD_H_EST = 220;
  const GAP = 16;
  const edge = 8;
  // The card is capped to the viewport on small phones (see TourCard's
  // `min(CARD_W, 100vw - 16px)`), so position maths must use the SAME effective
  // width — otherwise a 340px card would be pushed off a 320px screen.
  const cardW = Math.min(CARD_W, vw - 2 * edge);

  const leftClamp = Math.max(edge, Math.min(rect.left, vw - cardW - edge));

  // Prefer below
  if (rect.bottom + CARD_H_EST + GAP < vh) {
    return { top: rect.bottom + GAP, left: leftClamp };
  }
  // Above
  if (rect.top - CARD_H_EST - GAP > 0) {
    return { bottom: vh - rect.top + GAP, left: leftClamp };
  }
  // Right (sidebar elements)
  if (rect.right + cardW + GAP < vw) {
    return { top: Math.max(edge, Math.min(rect.top, vh - CARD_H_EST - edge)), left: rect.right + GAP };
  }
  // Left
  if (rect.left - cardW - GAP > 0) {
    return { top: Math.max(edge, Math.min(rect.top, vh - CARD_H_EST - edge)), right: vw - rect.left + GAP };
  }
  // Center fallback
  return { top: vh / 2 - CARD_H_EST / 2, left: Math.max(edge, vw / 2 - cardW / 2) };
}

// ─── Welcome modal ────────────────────────────────────────────────────────────

function WelcomeModal({
  open,
  onStart,
  onSkip,
}: {
  open: boolean;
  onStart: () => void;
  onSkip: () => void;
}) {
  return (
    <DialogPrimitive.Root open={open}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-[9060] bg-black/50 data-open:animate-in data-open:fade-in-0 data-closed:animate-out data-closed:fade-out-0" />
        <DialogPrimitive.Content
          onEscapeKeyDown={onSkip}
          onPointerDownOutside={onSkip}
          className={cn(
            "fixed left-1/2 top-1/2 z-[9061] w-[min(440px,calc(100vw-2rem))] -translate-x-1/2 -translate-y-1/2",
            "rounded-2xl border bg-card p-8 shadow-float",
            "data-open:animate-in data-open:fade-in-0 data-open:zoom-in-95",
            "data-closed:animate-out data-closed:fade-out-0 data-closed:zoom-out-95",
          )}
        >
          <DialogPrimitive.Title className="sr-only">Welcome to EstateMind</DialogPrimitive.Title>
          <div className="mb-6 flex items-start justify-between gap-4">
            <div className="flex size-14 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br from-primary/20 via-accent to-transparent ring-1 ring-primary/10">
              <span className="text-2xl">🏡</span>
            </div>
            <button onClick={onSkip} aria-label="Skip tour" className="rounded-lg p-1 text-muted-foreground hover:text-foreground">
              <X className="size-4" />
            </button>
          </div>
          <h2 className="font-heading text-2xl font-semibold tracking-tight">Welcome to EstateMind!</h2>
          <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
            EstateMind helps you search, compare, analyse and evaluate real estate using AI.
            Let&apos;s take a quick one-minute tour to explore the main features.
          </p>
          <div className="mt-7 flex gap-3">
            <Button onClick={onStart} className="flex-1">Start Tour</Button>
            <Button onClick={onSkip} variant="outline" className="flex-1">Skip Tour</Button>
          </div>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}

// ─── Step tooltip card ────────────────────────────────────────────────────────

function TourCard({
  step,
  total,
  pos,
  onNext,
  onPrev,
  onSkip,
}: {
  step: TourStep;
  total: number;
  pos: CardPos;
  onNext: () => void;
  onPrev: () => void;
  onSkip: () => void;
}) {
  const isLast = step === TOUR_STEPS[TOUR_STEPS.length - 1];
  const idx = TOUR_STEPS.indexOf(step);
  // The positioning maths estimates the card height, but long steps (e.g. AI
  // Analysis with its 8 bullets) can exceed it — clamp the card to the space
  // between its anchored edge and the viewport edge and scroll internally,
  // and never exceed the viewport width on small phones.
  const maxHeight =
    pos.top !== undefined
      ? `calc(100dvh - ${pos.top + 8}px)`
      : pos.bottom !== undefined
        ? `calc(100dvh - ${pos.bottom + 8}px)`
        : "80dvh";
  return (
    <div
      role="dialog"
      aria-label={step.title}
      style={{
        position: "fixed",
        width: `min(${CARD_W}px, calc(100vw - 16px))`,
        maxHeight,
        zIndex: 9062,
        ...pos,
      }}
      className="overflow-y-auto rounded-2xl border bg-card p-5 shadow-float animate-in fade-in zoom-in-95 duration-200"
    >
      {/* Header */}
      <div className="mb-3 flex items-start justify-between gap-2">
        <div>
          <p className="text-[0.65rem] font-semibold uppercase tracking-wider text-primary">
            Step {idx + 1} of {total}
          </p>
          <h3 className="mt-0.5 font-heading text-base font-semibold">{step.title}</h3>
        </div>
        <button
          onClick={onSkip}
          aria-label="Skip tour"
          className="mt-0.5 shrink-0 rounded-lg p-1 text-muted-foreground hover:text-foreground"
        >
          <X className="size-3.5" />
        </button>
      </div>
      {/* Progress bar */}
      <div className="mb-3 h-1 overflow-hidden rounded-full bg-muted">
        <div
          className="bg-brand-gradient h-full rounded-full transition-all duration-300"
          style={{ width: `${((idx + 1) / total) * 100}%` }}
        />
      </div>
      {/* Content */}
      <p className="text-sm leading-relaxed text-muted-foreground">{step.content}</p>
      {step.examples && step.examples.length > 0 && (
        <ul className="mt-2 space-y-1">
          {step.examples.map((ex) => (
            <li key={ex} className="flex items-center gap-2 text-xs text-muted-foreground">
              <span className="size-1.5 shrink-0 rounded-full bg-primary/60" />
              {ex}
            </li>
          ))}
        </ul>
      )}
      {step.footer && (
        <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{step.footer}</p>
      )}
      {/* Controls */}
      <div className="mt-4 flex items-center justify-between gap-2">
        <Button
          variant="ghost"
          size="sm"
          onClick={onSkip}
          className="text-xs text-muted-foreground hover:text-foreground"
        >
          Skip Tour
        </Button>
        <div className="flex gap-2">
          {idx > 0 && (
            <Button variant="outline" size="sm" onClick={onPrev} aria-label="Previous step">
              <ArrowLeft className="size-3.5" />
            </Button>
          )}
          <Button size="sm" onClick={onNext}>
            {isLast ? "Finish" : (
              <>Next <ArrowRight className="size-3.5" /></>
            )}
          </Button>
        </div>
      </div>
    </div>
  );
}

// ─── Completion card ──────────────────────────────────────────────────────────

function FinishCard({ onFinish }: { onFinish: () => void }) {
  return (
    <div
      role="dialog"
      aria-label="Tour complete"
      style={{ position: "fixed", zIndex: 9062, top: "50%", left: "50%", transform: "translate(-50%,-50%)", width: CARD_W }}
      className="rounded-2xl border bg-card p-8 text-center shadow-float animate-in fade-in zoom-in-95 duration-200"
    >
      <div className="mx-auto mb-4 flex size-14 items-center justify-center rounded-2xl bg-primary/10 text-primary">
        <CheckCircle2 className="size-7" />
      </div>
      <h3 className="font-heading text-2xl font-semibold">You&apos;re Ready!</h3>
      <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
        You&apos;re all set to explore EstateMind.
      </p>
      <Button onClick={onFinish} className="mt-6 w-full">Finish</Button>
    </div>
  );
}

// ─── Main TourProvider ────────────────────────────────────────────────────────

export function TourProvider({ children }: { children: React.ReactNode }) {
  const [welcome, setWelcome] = useState(false);
  const [tourActive, setTourActive] = useState(false);
  const [stepIdx, setStepIdx] = useState(0);
  const [showFinish, setShowFinish] = useState(false);
  const [rect, setRect] = useState<DOMRect | null>(null);
  const [cardPos, setCardPos] = useState<CardPos>({});
  const [mounted, setMounted] = useState(false);
  const rafRef = useRef<number | null>(null);
  const settleRef = useRef<number | null>(null);
  const router = useRouter();
  const pathname = usePathname();

  // Hydration guard — don't touch localStorage until client. Same pattern as
  // the Reports preview portal (sync mounted after first paint).
  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(() => setMounted(true), []);

  // Show welcome modal once for first-time users.
  useEffect(() => {
    if (!mounted) return;
    const seen = localStorage.getItem(TOUR_SEEN_KEY);
    if (!seen) {
      const t = setTimeout(() => setWelcome(true), 800);
      return () => clearTimeout(t);
    }
  }, [mounted]);

  const markSeen = useCallback(() => {
    try { localStorage.setItem(TOUR_SEEN_KEY, "1"); } catch { /* ignore */ }
  }, []);

  const skip = useCallback(() => {
    markSeen();
    setWelcome(false);
    setTourActive(false);
    setShowFinish(false);
  }, [markSeen]);

  const finish = useCallback(() => {
    markSeen();
    setTourActive(false);
    setShowFinish(false);
  }, [markSeen]);

  const startTour = useCallback(() => {
    setWelcome(false);
    setStepIdx(0);
    setShowFinish(false);
    // The tour highlights the Copilot workspace, so make sure we're on it
    // before spotlighting (e.g. when restarted from the Profile page).
    if (pathname !== "/dashboard" && pathname !== "/chat") {
      router.push("/dashboard");
      window.setTimeout(() => setTourActive(true), 600);
    } else {
      setTourActive(true);
    }
  }, [pathname, router]);

  // Compute the spotlight rect and tooltip position for the current step.
  const computeRect = useCallback(() => {
    if (!tourActive || showFinish) return;
    const step = TOUR_STEPS[stepIdx];
    const el = document.querySelector(step.target) as HTMLElement | null;
    if (!el) {
      setRect(null);
      setCardPos({ top: window.innerHeight / 2 - 110, left: window.innerWidth / 2 - CARD_W / 2 });
      return;
    }
    el.scrollIntoView({ behavior: "smooth", block: "nearest" });
    // Give scroll time to settle, then measure.
    rafRef.current = requestAnimationFrame(() => {
      const r = el.getBoundingClientRect();
      // Hidden element (e.g. sidebar collapsed into a sheet on mobile) —
      // fall back to a centered card without a spotlight.
      if (r.width === 0 && r.height === 0) {
        setRect(null);
        setCardPos({ top: window.innerHeight / 2 - 110, left: Math.max(8, window.innerWidth / 2 - CARD_W / 2) });
        return;
      }
      setRect(r);
      setCardPos(getCardPosition(r, window.innerWidth, window.innerHeight));
    });
    // Re-measure once the smooth scroll has settled; the CSS transition on
    // the spotlight animates any correction.
    settleRef.current = window.setTimeout(() => {
      const r = el.getBoundingClientRect();
      if (r.width > 0 || r.height > 0) {
        setRect(r);
        setCardPos(getCardPosition(r, window.innerWidth, window.innerHeight));
      }
    }, 400);
  }, [tourActive, stepIdx, showFinish]);

  // Measure asynchronously (rAF) so the DOM is settled and no setState runs
  // synchronously inside the effect body.
  useEffect(() => {
    rafRef.current = requestAnimationFrame(computeRect);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      if (settleRef.current) window.clearTimeout(settleRef.current);
    };
  }, [computeRect]);

  // Re-measure on resize.
  useEffect(() => {
    if (!tourActive) return;
    window.addEventListener("resize", computeRect);
    return () => window.removeEventListener("resize", computeRect);
  }, [tourActive, computeRect]);

  // ESC closes the tour (welcome modal, any step, and the finish card).
  useEffect(() => {
    if (!tourActive && !welcome && !showFinish) return;
    const handler = (e: KeyboardEvent) => { if (e.key === "Escape") skip(); };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [tourActive, welcome, showFinish, skip]);

  const next = useCallback(() => {
    if (stepIdx >= TOUR_STEPS.length - 1) { setTourActive(false); setShowFinish(true); }
    else setStepIdx((i) => i + 1);
  }, [stepIdx]);

  const prev = useCallback(() => setStepIdx((i) => Math.max(0, i - 1)), []);

  const spotlightStyle: React.CSSProperties | undefined = rect
    ? {
        position: "fixed",
        top: rect.top - SPOTLIGHT_PAD,
        left: rect.left - SPOTLIGHT_PAD,
        width: rect.width + SPOTLIGHT_PAD * 2,
        height: rect.height + SPOTLIGHT_PAD * 2,
        borderRadius: 12,
        boxShadow: "0 0 0 9999px rgba(0,0,0,0.55)",
        zIndex: 9051,
        pointerEvents: "none",
        transition: "top 0.25s,left 0.25s,width 0.25s,height 0.25s",
      }
    : undefined;

  return (
    <TourContext.Provider value={{ startTour }}>
      {children}

      {/* Welcome modal */}
      {mounted && (
        <WelcomeModal open={welcome} onStart={startTour} onSkip={skip} />
      )}

      {/* Tour overlay */}
      {mounted && tourActive && createPortal(
        <>
          {/* Click-blocker behind spotlight — also provides the dim layer
              when there is no spotlight rect (hidden/missing target). */}
          <div
            style={{
              position: "fixed",
              inset: 0,
              zIndex: 9050,
              background: rect ? undefined : "rgba(0,0,0,0.55)",
            }}
            aria-hidden
          />
          {/* Spotlight cutout */}
          {spotlightStyle && <div style={spotlightStyle} aria-hidden />}
          {/* Step card */}
          <TourCard
            step={TOUR_STEPS[stepIdx]}
            total={TOUR_STEPS.length}
            pos={cardPos}
            onNext={next}
            onPrev={prev}
            onSkip={skip}
          />
        </>,
        document.body
      )}

      {/* Finish card */}
      {mounted && showFinish && createPortal(
        <>
          <div style={{ position: "fixed", inset: 0, zIndex: 9050, background: "rgba(0,0,0,0.55)" }} aria-hidden />
          <FinishCard onFinish={finish} />
        </>,
        document.body
      )}
    </TourContext.Provider>
  );
}
