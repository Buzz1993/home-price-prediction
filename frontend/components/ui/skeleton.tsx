import { cn } from "@/lib/utils";

// Reusable loading placeholder. Renders a muted block with a shimmer sweep
// (Phase 18.1) that stands in for content while a request is in flight, so
// page/content skeletons are defined once instead of repeating the markup on
// every page (Phase 12.2).
function Skeleton({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="skeleton"
      className={cn("shimmer rounded-md bg-muted", className)}
      {...props}
    />
  );
}

export { Skeleton };
