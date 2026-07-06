// Reusable error state (Phase 12.3). One shared panel for every failed API
// request: a clear, non-technical message and an optional Retry button that the
// caller wires to its TanStack Query refetch() (queries) or mutate() (mutations).
// Matches the destructive banner style already used across the app so error UI is
// consistent instead of being re-implemented inline on every page.

import { RefreshCw, TriangleAlert } from "lucide-react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

type ErrorStateProps = {
  // Short, user-facing headline. Keep it non-technical.
  title?: string;
  // One line explaining what failed and that a retry may help.
  description?: string;
  // When provided, a Retry button is shown; wire it to refetch() / mutate().
  onRetry?: () => void;
  retryLabel?: string;
  // Disables the Retry button and spins its icon while a retry is in flight.
  retrying?: boolean;
  className?: string;
};

export function ErrorState({
  title = "Something went wrong",
  description = "We couldn't complete that request. Please try again.",
  onRetry,
  retryLabel = "Try again",
  retrying = false,
  className,
}: ErrorStateProps) {
  return (
    <div
      role="alert"
      className={cn(
        "flex items-start gap-3 rounded-lg border border-destructive/30 bg-destructive/5 p-4 text-sm text-destructive",
        className
      )}
    >
      <TriangleAlert className="mt-0.5 size-4 shrink-0" />
      <div className="min-w-0 flex-1 space-y-3">
        <div className="space-y-1">
          <p className="font-medium">{title}</p>
          <p className="text-destructive/80">{description}</p>
        </div>
        {onRetry && (
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={onRetry}
            disabled={retrying}
            className="border-destructive/30 bg-transparent text-destructive hover:bg-destructive/10 hover:text-destructive"
          >
            <RefreshCw className={cn(retrying && "animate-spin")} />
            {retryLabel}
          </Button>
        )}
      </div>
    </div>
  );
}
