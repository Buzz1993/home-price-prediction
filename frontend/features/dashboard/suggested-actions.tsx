"use client";

// AI Suggestions (Phase 15.11). Renders Claude's optional follow-up actions as
// quick-action chips below a completed assistant response. Selecting a chip
// re-sends it as a chat message, so the existing chat pipeline (Phase 15.8 tool
// orchestration) routes it to the correct EXISTING backend capability — no new
// workflow is introduced here.
//
// Suggestions are optional: when the backend omits them (Claude unavailable /
// failed) nothing renders. Chips wrap gracefully, are keyboard-navigable and
// stay within the green-and-white design system via the shared Button.

import { Sparkles } from "lucide-react";

import { Button } from "@/components/ui/button";
import { useWorkspace } from "./workspace-provider";

export function SuggestedActions({ suggestions }: { suggestions?: string[] }) {
  const { sendMessage, isSending } = useWorkspace();

  const items = (suggestions ?? [])
    .map((s) => s.trim())
    .filter((s) => s.length > 0);

  if (items.length === 0) return null;

  return (
    <div className="space-y-2 pt-1">
      <p className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
        <Sparkles className="size-3.5 text-primary" />
        Suggested next steps
      </p>
      <div className="flex flex-wrap gap-2">
        {items.map((suggestion) => (
          <Button
            key={suggestion}
            variant="outline"
            size="sm"
            disabled={isSending}
            onClick={() => sendMessage(suggestion)}
          >
            {suggestion}
          </Button>
        ))}
      </div>
    </div>
  );
}
