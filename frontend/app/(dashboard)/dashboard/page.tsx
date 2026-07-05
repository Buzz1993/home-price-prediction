import { WorkspaceProvider } from "@/features/dashboard/workspace-provider";
import { ChatWorkspace } from "@/features/dashboard/chat-workspace";
import { EvaluationTray } from "@/features/dashboard/evaluation-tray";

// Dashboard: the EstateMind Copilot workspace. A chat-first conversation column
// with an evaluation tray alongside, reproducing the Streamlit copilot layout
// (main chat + comparison tray). Both share one WorkspaceProvider so staging a
// property in chat updates the tray and vice versa.
export default function DashboardPage() {
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
