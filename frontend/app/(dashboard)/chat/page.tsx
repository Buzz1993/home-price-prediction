import { CopilotWorkspace } from "@/features/dashboard/copilot-workspace";

// AI Chat: interact with the EstateMind assistant in natural language. Reuses
// the shared Copilot workspace (chat window, suggested prompts, search results
// as property cards, and the evaluation tray) so chat logic is not duplicated.
export default function ChatPage() {
  return <CopilotWorkspace />;
}
