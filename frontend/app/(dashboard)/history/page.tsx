import { ChatHistoryWorkspace } from "@/features/history/chat-history-workspace";

// Chat History (Phase 18.9): a dedicated page for the AI conversation history
// that previously lived on the Profile page. Reuses the shared workspace
// provider's stored conversations — opening one resumes it in the Copilot.
export default function HistoryPage() {
  return <ChatHistoryWorkspace />;
}
