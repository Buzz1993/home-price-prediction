"use client";

// Renders a single conversation entry: the bubble text plus any structured
// payload routed to the right panel. Matches the Streamlit render_chat_history
// payload routing.

import { Bot, User } from "lucide-react";

import { cn } from "@/lib/utils";
import type { ChatMessage as ChatMessageType } from "@/types/dashboard";
import { SearchResultsPanel } from "./search-results-panel";
import { SearchExplanation } from "./search-explanation";
import { ComparisonResult } from "./comparison-result";
import { AnalysisTable } from "./analysis-table";
import { AdvisorCards, NegotiationCards } from "./analysis-cards";

function ResponsePayload({
  response,
}: {
  response: NonNullable<ChatMessageType["response"]>;
}) {
  switch (response.type) {
    case "search_results":
      // Claude's explanation (Phase 15.3) sits above the ranked property cards.
      // It only explains the backend results; the results render unchanged and
      // the card degrades gracefully when no explanation is available.
      return (
        <div className="space-y-3">
          {response.content.length > 0 && (
            <SearchExplanation explanation={response.ai_explanation} />
          )}
          <SearchResultsPanel results={response.content} />
        </div>
      );
    case "comparison":
      return <ComparisonResult data={response.content} />;
    case "rental":
    case "prediction":
    case "valuation":
      return <AnalysisTable rows={response.content} />;
    case "negotiation":
      return <NegotiationCards rows={response.content} />;
    case "advisor":
      return <AdvisorCards rows={response.content} />;
    default:
      return null;
  }
}

export function ChatMessage({ message }: { message: ChatMessageType }) {
  const isUser = message.role === "user";

  return (
    <div className={cn("flex gap-3", isUser && "flex-row-reverse")}>
      <div
        className={cn(
          "flex size-8 shrink-0 items-center justify-center rounded-full",
          isUser ? "bg-primary text-primary-foreground" : "bg-muted"
        )}
      >
        {isUser ? <User className="size-4" /> : <Bot className="size-4" />}
      </div>

      <div className={cn("min-w-0 max-w-full space-y-3", isUser && "items-end")}>
        {message.text && (
          <div
            className={cn(
              "inline-block max-w-full whitespace-pre-wrap break-words rounded-lg px-3 py-2 text-sm",
              isUser
                ? "bg-primary text-primary-foreground"
                : "bg-muted text-foreground"
            )}
          >
            {message.text}
          </div>
        )}

        {message.response && <ResponsePayload response={message.response} />}
      </div>
    </div>
  );
}
