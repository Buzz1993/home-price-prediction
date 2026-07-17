"use client";

// Theme provider (Phase 18.4). Holds the selected premium theme in React
// context, applies it as `data-theme` on <html> and persists it to
// localStorage so it survives reloads. The initial attribute is set by a tiny
// inline script in app/layout.tsx BEFORE hydration, so there is no flash of
// the default theme; this provider simply reads that attribute back on mount.
// Switching adds a short-lived `.theme-switching` class so token colors
// cross-fade (~200ms) instead of flashing. Visual tokens only — no routing,
// state-management or business-logic changes.

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";

import {
  DEFAULT_THEME,
  THEME_STORAGE_KEY,
  isThemeId,
  type ThemeId,
} from "@/lib/themes";

type ThemeContextValue = {
  theme: ThemeId;
  setTheme: (theme: ThemeId) => void;
};

const ThemeContext = createContext<ThemeContextValue | null>(null);

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  // Server render and first client render must match, so start from the
  // default and sync with the boot script's attribute after mount.
  const [theme, setThemeState] = useState<ThemeId>(DEFAULT_THEME);
  const switchTimeout = useRef<number | undefined>(undefined);

  useEffect(() => {
    const applied = document.documentElement.dataset.theme;
    // Hydration sync: read the theme the boot script already applied to <html>
    // and align React state. This is the SSR hydration pattern — intentional.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (isThemeId(applied)) setThemeState(applied);
    return () => window.clearTimeout(switchTimeout.current);
  }, []);

  const setTheme = useCallback((next: ThemeId) => {
    setThemeState(next);

    // Cross-fade the token swap: the class scopes a ~200ms color transition
    // to the switch itself, so ordinary interactions keep their own timing.
    const root = document.documentElement;
    root.classList.add("theme-switching");
    if (next === DEFAULT_THEME) {
      delete root.dataset.theme;
    } else {
      root.dataset.theme = next;
    }
    window.clearTimeout(switchTimeout.current);
    switchTimeout.current = window.setTimeout(
      () => root.classList.remove("theme-switching"),
      250,
    );

    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, next);
    } catch {
      // Persistence is best-effort (e.g. private browsing); the theme still
      // applies for the current session.
    }
  }, []);

  return (
    <ThemeContext.Provider value={{ theme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme(): ThemeContextValue {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error("useTheme must be used within a ThemeProvider");
  }
  return context;
}
