"use client";

// Renders a single conversation entry: the bubble text plus any structured
// payload routed to the right panel. Matches the Streamlit render_chat_history
// payload routing.

import { Bot, User } from "lucide-react";

import { cn } from "@/lib/utils";
import type { ChatMessage as ChatMessageType } from "@/types/dashboard";
import { PropertyResultsPanel } from "./property-results-panel";
import { SearchExplanation } from "./search-explanation";
import { ComparisonResult } from "./comparison-result";
import { AnalysisTable } from "./analysis-table";
import { AdvisorCards, NegotiationCards } from "./analysis-cards";
import { SuggestedActions } from "./suggested-actions";

function ResponsePayload({
  response,
  isLatestSearch,
}: {
  response: NonNullable<ChatMessageType["response"]>;
  // True only for the newest search-results message. The accumulated Property
  // Results panel renders once, under that message; earlier search messages show
  // just their explanation + how many properties that search added.
  isLatestSearch: boolean;
}) {
  switch (response.type) {
    case "search_results":
      // Claude's explanation (Phase 15.3) sits above the property cards. It only
      // explains the backend results; the results render unchanged and the card
      // degrades gracefully when no explanation is available. The cards
      // themselves come from the conversation's ACCUMULATED collection so they
      // grow across searches (Phase 15.13) — shown under the latest search only.
      return (
        <div className="space-y-3">
          {response.content.length > 0 && (
            <SearchExplanation explanation={response.ai_explanation} />
          )}
          {isLatestSearch ? (
            <PropertyResultsPanel />
          ) : (
            response.content.length > 0 && (
              <p className="text-xs text-muted-foreground">
                Added {response.content.length}{" "}
                {response.content.length === 1 ? "property" : "properties"} to
                this conversation.
              </p>
            )
          )}
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

export function ChatMessage({
  message,
  showSuggestions = false,
  isLatestSearch = false,
}: {
  message: ChatMessageType;
  // Only the latest assistant message shows follow-up suggestions (Phase 15.11),
  // keeping earlier turns clean and avoiding repeated chips.
  showSuggestions?: boolean;
  // Marks the newest search-results message so the accumulated Property Results
  // panel renders once, under it (Phase 15.13).
  isLatestSearch?: boolean;
}) {
  const isUser = message.role === "user";
  // While Claude's text is still arriving (Phase 15.9), show a blinking cursor.
  // An assistant message that is streaming but has no text yet renders nothing
  // here — the workspace shows the thinking indicator until the first token.
  const isStreaming = message.streaming === true;
  const showBubble = Boolean(message.text) || isStreaming;

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
        {showBubble && message.text && (
          <div
            className={cn(
              "inline-block max-w-full whitespace-pre-wrap break-words rounded-lg px-3 py-2 text-sm",
              isUser
                ? "bg-primary text-primary-foreground"
                : "bg-muted text-foreground"
            )}
          >
            {message.text}
            {isStreaming && (
              <span
                aria-hidden
                className="ml-0.5 inline-block h-4 w-1.5 translate-y-0.5 animate-pulse rounded-sm bg-current align-middle"
              />
            )}
          </div>
        )}

        {message.response && (
          <ResponsePayload
            response={message.response}
            isLatestSearch={isLatestSearch}
          />
        )}

        {/* Follow-up suggestions (Phase 15.11): only on the latest assistant
            message and once its text has finished streaming. */}
        {showSuggestions && !isStreaming && (
          <SuggestedActions suggestions={message.suggestions} />
        )}
      </div>
    </div>
  );
}
