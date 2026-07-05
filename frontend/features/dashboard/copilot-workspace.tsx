import { WorkspaceProvider } from "./workspace-provider";
import { ChatWorkspace } from "./chat-workspace";
import { EvaluationTray } from "./evaluation-tray";

// The EstateMind Copilot workspace: a chat-first conversation column with the
// evaluation tray alongside. Both share one WorkspaceProvider so staging a
// property in chat updates the tray and vice versa. Reused by the Dashboard
// (Phase 4) and the dedicated AI Chat page (Phase 5) so the layout and chat
// logic are defined once.
export function CopilotWorkspace() {
  return (
    <WorkspaceProvider>
      <div className="grid gap-4 lg:h-[calc(100dvh-7rem)] lg:grid-cols-[minmax(0,1fr)_20rem]">
        <section className="flex h-[70dvh] min-h-0 flex-col overflow-hidden rounded-xl border bg-card lg:h-auto">
          <ChatWorkspace />
        </section>
        <aside className="min-h-0 overflow-hidden rounded-xl border bg-card">
          <EvaluationTray />
        </aside>
      </div>
    </WorkspaceProvider>
  );
}
