"use client";

// Profile → Appearance (Phase 18.4). Premium theme selector: four cards, one
// per built-in theme from lib/themes.ts. Clicking a card applies the theme
// instantly through the ThemeProvider (design tokens only — no layout or
// functionality changes) and persists it to localStorage. Implemented as a
// radio group so keyboard users can arrow between themes; the selected card
// shows a brand outline, glow and check icon.
//
// Phase 18.7: refined presentation with better spacing and visual hierarchy.

import { Check, Palette } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useTheme } from "@/components/providers/theme-provider";
import { THEMES, type Theme } from "@/lib/themes";
import { cn } from "@/lib/utils";

export function AppearanceSection() {
  const { theme, setTheme } = useTheme();

  return (
    <Card className="shadow-float">
      <CardHeader className="space-y-2 pb-4">
        <CardTitle className="flex items-center gap-2.5 text-lg">
          <div className="flex size-9 items-center justify-center rounded-xl bg-primary/10 text-primary">
            <Palette className="size-5" />
          </div>
          Appearance
        </CardTitle>
        <p className="text-sm text-muted-foreground">
          Choose your preferred EstateMind theme
        </p>
      </CardHeader>
      <CardContent className="pt-0">
        <div
          role="radiogroup"
          aria-label="EstateMind theme"
          className="grid grid-cols-1 gap-3 sm:grid-cols-2"
        >
          {THEMES.map((option) => (
            <ThemeCard
              key={option.id}
              theme={option}
              selected={option.id === theme}
              onSelect={() => setTheme(option.id)}
            />
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

type ThemeCardProps = {
  theme: Theme;
  selected: boolean;
  onSelect: () => void;
};

function ThemeCard({ theme, selected, onSelect }: ThemeCardProps) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={selected}
      aria-label={`${theme.name} theme — ${theme.description}`}
      onClick={onSelect}
      className={cn(
        "group relative w-full rounded-xl border bg-card p-5 text-left transition-all duration-200",
        "hover:-translate-y-0.5 hover:border-ring/60 hover:shadow-float-lg",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
        selected
          ? "border-primary shadow-brand-glow ring-2 ring-primary/20"
          : "border-border shadow-float",
      )}
    >
      {/* Selection indicator. */}
      <span
        aria-hidden
        className={cn(
          "absolute right-3 top-3 flex size-6 items-center justify-center rounded-full transition-all duration-200",
          selected
            ? "bg-primary text-primary-foreground opacity-100 shadow-sm"
            : "opacity-0",
        )}
      >
        <Check className="size-3.5" strokeWidth={3} />
      </span>

      {/* Color preview swatches (primary → secondary → accent). */}
      <span aria-hidden className="flex items-center gap-2">
        {theme.swatches.map((color) => (
          <span
            key={color}
            className="size-7 rounded-full border-2 border-background shadow-sm ring-1 ring-black/5"
            style={{ backgroundColor: color }}
          />
        ))}
      </span>

      <span className="mt-4 block text-base font-semibold">{theme.name}</span>
      <span className="mt-1 block text-sm text-muted-foreground">
        {theme.description}
      </span>
    </button>
  );
}
