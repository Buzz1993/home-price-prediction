// The EstateMind Copilot workspace: the permanent four-column layout
// (Conversation Sidebar | Chat + Results | Property Map | Evaluation Tray).
// The shared WorkspaceProvider is mounted by the (dashboard) layout so the
// tray, conversations and accumulated properties persist across routes. Reused
// by the Dashboard (Phase 4) and the dedicated AI Chat page (Phase 5) so the
// layout and chat logic are defined once. Phase 15.13 upgraded this to the
// full four-column workspace (see WorkspaceShell).
import { WorkspaceShell } from "./workspace-shell";

export function CopilotWorkspace() {
  return <WorkspaceShell />;
}
