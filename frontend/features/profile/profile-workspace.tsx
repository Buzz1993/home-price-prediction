"use client";

// Profile page body (Phase 11). Composes three sections — account, generated
// reports and AI chat history — from the documented GET /profile, GET /reports and
// GET /chat-history endpoints, and offers a logout action. Each section handles
// its own loading, empty and error states via the reusable ProfileSection. No
// authentication or backend business logic lives here; the page only reads and
// renders what the backend returns (UI doc §15).

import { FileText, LogOut, MessageSquare, UserRound } from "lucide-react";

import { Button } from "@/components/ui/button";
import { AppearanceSection } from "./appearance-section";
import { ChatHistoryList } from "./chat-history-list";
import { ProfileSection } from "./profile-section";
import { ReportsList } from "./reports-list";
import { UserInfoCard } from "./user-info-card";
import { useChatHistory, useLogout, useProfile, useReports } from "./use-profile";

export function ProfileWorkspace() {
  const profile = useProfile();
  const reports = useReports();
  const history = useChatHistory();
  const logout = useLogout();

  const reportList = reports.data ?? [];
  const chats = history.data ?? [];

  return (
    <div className="mx-auto max-w-3xl space-y-4">
      <header className="space-y-1">
        <h1 className="font-heading text-2xl font-semibold tracking-tight">
          Profile
        </h1>
        <p className="text-sm text-muted-foreground">
          Your account, generated reports and AI chat history.
        </p>
      </header>

      <ProfileSection
        title="Account"
        icon={UserRound}
        isLoading={profile.isLoading}
        isError={profile.isError}
        onRetry={() => profile.refetch()}
        retrying={profile.isFetching}
        isEmpty={!profile.data}
        emptyMessage="No profile information available."
        action={
          <Button variant="destructive" size="sm" onClick={logout}>
            <LogOut /> Log out
          </Button>
        }
      >
        {profile.data && <UserInfoCard user={profile.data} />}
      </ProfileSection>

      <AppearanceSection />

      <ProfileSection
        title="Generated Reports"
        icon={FileText}
        isLoading={reports.isLoading}
        isError={reports.isError}
        onRetry={() => reports.refetch()}
        retrying={reports.isFetching}
        isEmpty={reportList.length === 0}
        emptyMessage="You haven't generated any reports yet. Create one from the Reports page."
      >
        <ReportsList reports={reportList} />
      </ProfileSection>

      <ProfileSection
        title="AI Chat History"
        icon={MessageSquare}
        isLoading={history.isLoading}
        isError={history.isError}
        onRetry={() => history.refetch()}
        retrying={history.isFetching}
        isEmpty={chats.length === 0}
        emptyMessage="No AI conversations yet. Start chatting from the AI Chat page."
      >
        <ChatHistoryList entries={chats} />
      </ProfileSection>
    </div>
  );
}
