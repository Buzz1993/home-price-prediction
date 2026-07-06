import { cn } from "@/lib/utils";

// Reusable loading placeholder. Renders a muted, pulsing block that stands in for
// content while a request is in flight, so page/content skeletons are defined once
// instead of repeating the animate-pulse markup on every page (Phase 12.2).
function Skeleton({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="skeleton"
      className={cn("animate-pulse rounded-md bg-muted", className)}
      {...props}
    />
  );
}

export { Skeleton };
