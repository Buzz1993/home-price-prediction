"use client";

// Profile → Appearance (Phase 18.4). Premium theme selector: four cards, one
// per built-in theme from lib/themes.ts. Clicking a card applies the theme
// instantly through the ThemeProvider (design tokens only — no layout or
// functionality changes) and persists it to localStorage. Implemented as a
// radio group so keyboard users can arrow between themes; the selected card
// shows a brand outline, glow and check icon.

import { Check, Palette } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useTheme } from "@/components/providers/theme-provider";
import { THEMES, type Theme } from "@/lib/themes";
import { cn } from "@/lib/utils";

export function AppearanceSection() {
  const { theme, setTheme } = useTheme();

  return (
    <Card>
      <CardHeader className="space-y-1">
        <CardTitle className="flex items-center gap-2 text-base">
          <Palette className="size-4 text-muted-foreground" />
          Appearance
        </CardTitle>
        <p className="text-sm text-muted-foreground">
          Choose your preferred EstateMind theme.
        </p>
      </CardHeader>
      <CardContent>
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
        "group relative w-full rounded-xl border bg-card p-4 text-left transition-all duration-200",
        "hover:-translate-y-0.5 hover:border-ring/60 hover:shadow-float-lg",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
        selected
          ? "border-primary shadow-brand-glow"
          : "border-border shadow-float",
      )}
    >
      {/* Selection indicator. */}
      <span
        aria-hidden
        className={cn(
          "absolute right-3 top-3 flex size-5 items-center justify-center rounded-full transition-all duration-200",
          selected
            ? "bg-primary text-primary-foreground opacity-100"
            : "opacity-0",
        )}
      >
        <Check className="size-3" strokeWidth={3} />
      </span>

      {/* Color preview swatches (primary → secondary → accent). */}
      <span aria-hidden className="flex items-center gap-1.5">
        {theme.swatches.map((color) => (
          <span
            key={color}
            className="size-6 rounded-full border border-black/10"
            style={{ backgroundColor: color }}
          />
        ))}
      </span>

      <span className="mt-3 block text-sm font-semibold">{theme.name}</span>
      <span className="mt-0.5 block text-xs text-muted-foreground">
        {theme.description}
      </span>
    </button>
  );
}
