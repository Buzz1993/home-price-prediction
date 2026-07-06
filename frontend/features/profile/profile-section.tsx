// Reusable card section for the Profile page. Renders a titled card with an
// optional header action and switches its body between the loading, error, empty
// and content states — so each profile section (account, reports, chat history)
// handles all four states without duplicating the markup.

import type { LucideIcon } from "lucide-react";
import { TriangleAlert } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

type ProfileSectionProps = {
  title: string;
  icon: LucideIcon;
  // The endpoint powering this section, shown in the error hint.
  endpoint: string;
  isLoading: boolean;
  isError: boolean;
  error?: unknown;
  isEmpty: boolean;
  emptyMessage: string;
  // Optional header-right slot (e.g. the logout button on the account section).
  action?: React.ReactNode;
  children: React.ReactNode;
};

export function ProfileSection({
  title,
  icon: Icon,
  endpoint,
  isLoading,
  isError,
  error,
  isEmpty,
  emptyMessage,
  action,
  children,
}: ProfileSectionProps) {
  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between gap-2 space-y-0">
        <CardTitle className="flex items-center gap-2 text-base">
          <Icon className="size-4 text-muted-foreground" />
          {title}
        </CardTitle>
        {action}
      </CardHeader>
      <CardContent>
        {isLoading && (
          <div className="space-y-2">
            <div className="h-4 w-1/3 animate-pulse rounded bg-muted" />
            <div className="h-4 w-2/3 animate-pulse rounded bg-muted" />
          </div>
        )}

        {isError && (
          <div className="flex items-start gap-2 rounded-lg border border-destructive/30 bg-destructive/5 p-3 text-sm text-destructive">
            <TriangleAlert className="mt-0.5 size-4 shrink-0" />
            <span>
              Could not load {title.toLowerCase()}.{" "}
              {error instanceof Error ? error.message : "Please try again."} Make
              sure the EstateMind Copilot API (src/api/main.py) is running and the{" "}
              {endpoint} endpoint is reachable at the configured base URL.
            </span>
          </div>
        )}

        {!isLoading && !isError && isEmpty && (
          <p className="py-1 text-sm text-muted-foreground">{emptyMessage}</p>
        )}

        {!isLoading && !isError && !isEmpty && children}
      </CardContent>
    </Card>
  );
}
