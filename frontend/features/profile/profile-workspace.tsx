"use client";

// Profile page body (Phase 11). Composes three sections — account, generated
// reports and AI chat history — from the documented GET /profile, GET /reports and
// GET /chat-history endpoints, and offers a logout action. Each section handles
// its own loading, empty and error states via the reusable ProfileSection. No
// authentication or backend business logic lives here; the page only reads and
// renders what the backend returns (UI doc §15).
//
// Phase 18.7: premium presentation polish — hero header with gradient avatar,
// account stats cards, refined section cards, improved spacing and typography.

import { FileText, LogOut, MessageSquare, TrendingUp, UserRound } from "lucide-react";

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
  const user = profile.data;

  // Derive initials for the large avatar
  const initials = user?.name
    ? user.name
        .split(" ")
        .filter(Boolean)
        .slice(0, 2)
        .map((word) => word[0]?.toUpperCase())
        .join("") || "U"
    : "U";

  return (
    <div className="space-y-6">
      {/* Premium profile hero (Phase 18.7) */}
      <header className="relative overflow-hidden rounded-xl border bg-gradient-to-br from-primary/10 via-accent/5 to-transparent p-8 shadow-float">
        <div className="relative z-10 flex flex-col items-start gap-6 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-5">
            {/* Large gradient avatar */}
            <div className="flex size-20 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br from-primary via-primary/80 to-primary/60 text-2xl font-bold text-primary-foreground shadow-lg ring-4 ring-background">
              {initials}
            </div>
            <div className="space-y-2">
              <h1 className="font-heading text-3xl font-semibold tracking-tight">
                {user?.name || "Profile"}
              </h1>
              {user?.email && (
                <p className="text-sm text-muted-foreground">{user.email}</p>
              )}
              <div className="flex flex-wrap gap-2">
                <span className="rounded-full bg-primary/10 px-3 py-1 text-xs font-medium text-primary">
                  Premium Member
                </span>
                <span className="rounded-full bg-accent px-3 py-1 text-xs font-medium text-accent-foreground">
                  AI Powered
                </span>
              </div>
            </div>
          </div>
          <Button variant="destructive" size="default" onClick={logout} className="gap-2">
            <LogOut /> Log Out
          </Button>
        </div>
      </header>

      {/* Account stats cards (Phase 18.7) */}
      {!profile.isLoading && user && (
        <div className="grid gap-4 sm:grid-cols-3">
          <div className="group rounded-xl border bg-gradient-to-br from-primary/10 via-primary/5 to-transparent p-5 shadow-float transition-all duration-200 hover:-translate-y-0.5 hover:shadow-float-lg">
            <div className="flex items-center gap-3">
              <div className="flex size-12 items-center justify-center rounded-xl bg-primary/10 text-primary">
                <FileText className="size-6" />
              </div>
              <div>
                <p className="text-2xl font-bold">{reportList.length}</p>
                <p className="text-sm text-muted-foreground">Reports Generated</p>
              </div>
            </div>
          </div>
          <div className="group rounded-xl border bg-gradient-to-br from-primary/10 via-primary/5 to-transparent p-5 shadow-float transition-all duration-200 hover:-translate-y-0.5 hover:shadow-float-lg">
            <div className="flex items-center gap-3">
              <div className="flex size-12 items-center justify-center rounded-xl bg-primary/10 text-primary">
                <MessageSquare className="size-6" />
              </div>
              <div>
                <p className="text-2xl font-bold">{chats.length}</p>
                <p className="text-sm text-muted-foreground">AI Conversations</p>
              </div>
            </div>
          </div>
          <div className="group rounded-xl border bg-gradient-to-br from-primary/10 via-primary/5 to-transparent p-5 shadow-float transition-all duration-200 hover:-translate-y-0.5 hover:shadow-float-lg">
            <div className="flex items-center gap-3">
              <div className="flex size-12 items-center justify-center rounded-xl bg-primary/10 text-primary">
                <TrendingUp className="size-6" />
              </div>
              <div>
                <p className="text-2xl font-bold">Active</p>
                <p className="text-sm text-muted-foreground">Account Status</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Profile sections */}
      <div className="mx-auto max-w-4xl space-y-5">
        <ProfileSection
          title="Account Information"
          icon={UserRound}
          isLoading={profile.isLoading}
          isError={profile.isError}
          onRetry={() => profile.refetch()}
          retrying={profile.isFetching}
          isEmpty={!profile.data}
          emptyMessage="No profile information available."
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
    </div>
  );
}
