import { Loader2 } from "lucide-react";

import { cn } from "@/lib/utils";

// Reusable loading spinner. Wraps the lucide Loader2 icon with a spin animation so
// buttons and inline "…in progress" states share one indicator instead of
// repeating <Loader2 className="animate-spin" /> everywhere (Phase 12.2). Inside a
// Button it inherits the button's icon sizing; pass a size class for inline use.
function Spinner({ className, ...props }: React.ComponentProps<typeof Loader2>) {
  return (
    <Loader2
      role="status"
      aria-label="Loading"
      className={cn("animate-spin", className)}
      {...props}
    />
  );
}

export { Spinner };
