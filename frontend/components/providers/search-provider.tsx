"use client";

// Lightweight global-search state (Phase 18.9). The navbar's search input
// writes the query here and the page that owns searchable content (currently
// the Reports page filtering its report history) reads it. Pure UI state —
// no backend calls, following the project's React Context rule for global UI
// state. The query resets on navigation so a filter typed on one page never
// silently constrains another.

import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import { usePathname } from "next/navigation";

type SearchContextValue = {
  query: string;
  setQuery: (query: string) => void;
};

const SearchContext = createContext<SearchContextValue | null>(null);

export function SearchProvider({ children }: { children: React.ReactNode }) {
  const [query, setQuery] = useState("");
  const pathname = usePathname();

  // Clear the query whenever the route changes.
  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(() => setQuery(""), [pathname]);

  const value = useMemo(() => ({ query, setQuery }), [query]);

  return (
    <SearchContext.Provider value={value}>{children}</SearchContext.Provider>
  );
}

export function useGlobalSearch(): SearchContextValue {
  const context = useContext(SearchContext);
  if (!context) {
    throw new Error("useGlobalSearch must be used within a SearchProvider");
  }
  return context;
}
