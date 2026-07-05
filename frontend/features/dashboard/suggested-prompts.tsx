"use client";

// Empty-state suggestions shown before the first message. Clicking one sends it
// as a chat message, matching the example prompts in the Streamlit copilot.

import { Sparkles } from "lucide-react";

import { Button } from "@/components/ui/button";
import { useWorkspace } from "./workspace-provider";

const PROMPTS = [
  "2 BHK with CCTV near Goregaon railway station",
  "Affordable 3 BHK in Thane with gym and swimming pool",
  "Luxury apartments in Powai with good connectivity",
  "Spacious family homes in Navi Mumbai",
];

export function SuggestedPrompts() {
  const { sendMessage, isSending } = useWorkspace();

  return (
    <div className="mx-auto max-w-lg space-y-4 py-10 text-center">
      <div className="mx-auto flex size-12 items-center justify-center rounded-full bg-muted">
        <Sparkles className="size-6 text-muted-foreground" />
      </div>
      <div className="space-y-1">
        <h2 className="font-heading text-lg font-semibold">
          Ask EstateMind anything
        </h2>
        <p className="text-sm text-muted-foreground">
          Describe what you&apos;re looking for in plain language and stage
          properties to compare, predict prices and more.
        </p>
      </div>
      <div className="flex flex-wrap justify-center gap-2">
        {PROMPTS.map((prompt) => (
          <Button
            key={prompt}
            variant="outline"
            size="sm"
            disabled={isSending}
            onClick={() => sendMessage(prompt)}
          >
            {prompt}
          </Button>
        ))}
      </div>
    </div>
  );
}
