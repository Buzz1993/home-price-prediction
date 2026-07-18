// Reusable "Original Listing" action. Renders a link that opens the backend
// `ap_pjt_url` in a new tab. Presentational only — it performs lightweight
// client-side validation and renders nothing when the URL is missing, empty or
// unusable, so callers can drop it in unconditionally.

import { ExternalLink } from "lucide-react";

import { Button, type buttonVariants } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type { VariantProps } from "class-variance-authority";

type OriginalListingButtonProps = {
  // The backend `ap_pjt_url`. May be null/undefined/empty/invalid.
  url: string | null | undefined;
  label?: string;
  className?: string;
} & Pick<VariantProps<typeof buttonVariants>, "variant" | "size">;

// A usable listing URL is a well-formed absolute http(s) URL. Anything else
// (null, whitespace, relative or non-http) is treated as unavailable.
function usableUrl(url: string | null | undefined): string | null {
  if (typeof url !== "string") return null;
  const trimmed = url.trim();
  if (!trimmed) return null;
  try {
    const parsed = new URL(trimmed);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      return null;
    }
    return trimmed;
  } catch {
    return null;
  }
}

export function OriginalListingButton({
  url,
  label = "View Original Listing",
  variant = "default",
  size = "default",
  className,
}: OriginalListingButtonProps) {
  const href = usableUrl(url);
  if (!href) return null;

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button asChild variant={variant} size={size} className={className}>
          <a href={href} target="_blank" rel="noopener noreferrer">
            {label} <ExternalLink />
          </a>
        </Button>
      </TooltipTrigger>
      <TooltipContent className="max-w-[220px]">
        Open the original property listing.
      </TooltipContent>
    </Tooltip>
  );
}
