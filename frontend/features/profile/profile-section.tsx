// Reusable card section for the Profile page. Renders a titled card with an
// optional header action and switches its body between the loading, error, empty
// and content states — so each profile section (account, reports, chat history)
// handles all four states without duplicating the markup.
//
// Phase 18.7: premium card styling with floating shadow, better spacing.

import type { LucideIcon } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { EmptyState } from "@/components/ui/empty-state";
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
    <Card className="shadow-float">
      <CardHeader className="flex-row items-center justify-between gap-3 space-y-0 pb-4">
        <CardTitle className="flex items-center gap-2.5 text-lg">
          <div className="flex size-9 items-center justify-center rounded-xl bg-primary/10 text-primary">
            <Icon className="size-5" />
          </div>
          {title}
        </CardTitle>
        {action}
      </CardHeader>
      <CardContent className="pt-0">
        {isLoading && (
          <div className="space-y-3">
            <Skeleton className="h-5 w-1/3" />
            <Skeleton className="h-5 w-2/3" />
            <Skeleton className="h-5 w-1/2" />
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
          <EmptyState icon={Icon} description={emptyMessage} className="py-8" />
        )}

        {!isLoading && !isError && !isEmpty && children}
      </CardContent>
    </Card>
  );
}
