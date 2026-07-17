// Premium theme catalog (Phase 18.4). Each entry mirrors a token set defined
// in app/globals.css — `:root` for the Estate Green default, and a matching
// `[data-theme="…"]` block for the rest. Switching themes only swaps design
// tokens; layouts and functionality never change. The swatches here are used
// solely by the Profile → Appearance theme cards for their color previews.

export type ThemeId =
  | "estate-green"
  | "royal-purple"
  | "midnight-blue"
  | "sunset-gold";

export type Theme = {
  id: ThemeId;
  name: string;
  description: string;
  // Preview swatches shown on the theme card (primary → secondary → accent).
  swatches: [string, string, string];
};

export const DEFAULT_THEME: ThemeId = "estate-green";

export const THEME_STORAGE_KEY = "estatemind.theme";

export const THEMES: Theme[] = [
  {
    id: "estate-green",
    name: "Estate Green",
    description: "Clean, trustworthy and professional.",
    swatches: ["#15803d", "#22c55e", "#4ade80"],
  },
  {
    id: "royal-purple",
    name: "Royal Purple",
    description: "Luxury AI SaaS appearance.",
    swatches: ["#6d4aff", "#8c6dff", "#a78bfa"],
  },
  {
    id: "midnight-blue",
    name: "Midnight Blue",
    description: "Enterprise and corporate.",
    swatches: ["#2563eb", "#3b82f6", "#60a5fa"],
  },
  {
    id: "sunset-gold",
    name: "Sunset Gold",
    description: "Premium investment dashboard.",
    swatches: ["#c48a00", "#eab308", "#facc15"],
  },
];

export function isThemeId(value: unknown): value is ThemeId {
  return THEMES.some((theme) => theme.id === value);
}
