"use client";

// Global auth state (React Context per the project's state-management rules).
// Holds the current user + token, hydrates from localStorage on mount, and
// exposes helpers to start/end a session.

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";

import type { AuthResponse, User } from "@/types/auth";
import { clearAuth, getStoredToken, getStoredUser, storeAuth } from "@/lib/auth-storage";

type AuthContextValue = {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  // False until the stored session has been read on the client.
  isReady: boolean;
  setSession: (session: AuthResponse) => void;
  clearSession: () => void;
};

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isReady, setIsReady] = useState(false);

  // Hydrate the session from storage once, on the client. Reading localStorage
  // must happen after mount to avoid an SSR hydration mismatch, so a synchronous
  // setState here is intentional.
  useEffect(() => {
    /* eslint-disable react-hooks/set-state-in-effect */
    setToken(getStoredToken());
    setUser(getStoredUser());
    setIsReady(true);
    /* eslint-enable react-hooks/set-state-in-effect */
  }, []);

  const setSession = useCallback((session: AuthResponse) => {
    const nextUser = session.user ?? null;
    setToken(session.token);
    setUser(nextUser);
    storeAuth(session.token, nextUser);
  }, []);

  const clearSession = useCallback(() => {
    setToken(null);
    setUser(null);
    clearAuth();
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({
      user,
      token,
      isAuthenticated: Boolean(token),
      isReady,
      setSession,
      clearSession,
    }),
    [user, token, isReady, setSession, clearSession]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
