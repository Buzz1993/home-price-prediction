// Reusable card section for the Profile page. Renders a titled card with an
// optional header action and switches its body between the loading, error, empty
// and content states — so each profile section (account, reports, chat history)
// handles all four states without duplicating the markup.

import type { LucideIcon } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ErrorState } from "@/components/ui/error-state";
import { Skeleton } from "@/components/ui/skeleton";

type ProfileSectionProps = {
  title: string;
  icon: LucideIcon;
  isLoading: boolean;
  isError: boolean;
  isEmpty: boolean;
  emptyMessage: string;
  // Retry the section's query (TanStack Query refetch); a spinner shows while
  // the retry is in flight.
  onRetry?: () => void;
  retrying?: boolean;
  // Optional header-right slot (e.g. the logout button on the account section).
  action?: React.ReactNode;
  children: React.ReactNode;
};

export function ProfileSection({
  title,
  icon: Icon,
  isLoading,
  isError,
  isEmpty,
  emptyMessage,
  onRetry,
  retrying,
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
            <Skeleton className="h-4 w-1/3" />
            <Skeleton className="h-4 w-2/3" />
          </div>
        )}

        {isError && (
          <ErrorState
            title={`Could not load ${title.toLowerCase()}`}
            description="Something went wrong while loading this section. Please try again."
            onRetry={onRetry}
            retrying={retrying}
          />
        )}

        {!isLoading && !isError && isEmpty && (
          <p className="py-1 text-sm text-muted-foreground">{emptyMessage}</p>
        )}

        {!isLoading && !isError && !isEmpty && children}
      </CardContent>
    </Card>
  );
}
