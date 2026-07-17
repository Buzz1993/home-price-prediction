"use client";

// Profile page body (Phase 11, cleaned in Phase 18.9). Shows the account hero,
// account stats, account information and appearance settings, plus a logout
// action. The placeholder "Generated Reports" cards were removed — the Reports
// page (/reports) is the single source of truth for report history, reached
// via the View Report History button. AI Chat History moved to its own
// dedicated page (/history); the stats below read the SAME local stores those
// pages use, so the numbers always match what the user finds there.

import Link from "next/link";
import { useEffect, useState } from "react";
import { FileText, LogOut, MessageSquare, TrendingUp, UserRound } from "lucide-react";

import { Button } from "@/components/ui/button";
import { useWorkspace } from "@/features/dashboard/workspace-provider";
import { loadReportHistory } from "@/features/reports/report-history";
import { AppearanceSection } from "./appearance-section";
import { ProfileSection } from "./profile-section";
import { UserInfoCard } from "./user-info-card";
import { useLogout, useProfile } from "./use-profile";

export function ProfileWorkspace() {
  const profile = useProfile();
  const logout = useLogout();
  const { conversations } = useWorkspace();

  // Local report count — read after mount (localStorage is client-only).
  const [reportCount, setReportCount] = useState(0);
  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(() => setReportCount(loadReportHistory().length), []);

  // Count only conversations that actually contain messages.
  const conversationCount = conversations.filter(
    (c) => c.messages.length > 0
  ).length;

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

      {/* Account stats cards (Phase 18.7) — counts come from the same local
          stores the Reports and Chat History pages read. */}
      {!profile.isLoading && user && (
        <div className="grid gap-4 sm:grid-cols-3">
          <div className="group rounded-xl border bg-gradient-to-br from-primary/10 via-primary/5 to-transparent p-5 shadow-float transition-all duration-200 hover:-translate-y-0.5 hover:shadow-float-lg">
            <div className="flex items-center gap-3">
              <div className="flex size-12 items-center justify-center rounded-xl bg-primary/10 text-primary">
                <FileText className="size-6" />
              </div>
              <div>
                <p className="text-2xl font-bold">{reportCount}</p>
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
                <p className="text-2xl font-bold">{conversationCount}</p>
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

        {/* Reports live on the Reports page — the single source of truth
            (Phase 18.9). */}
        <div className="flex flex-col items-start justify-between gap-4 rounded-xl border bg-gradient-to-br from-muted/30 to-transparent p-6 shadow-float sm:flex-row sm:items-center">
          <div className="flex items-center gap-4">
            <div className="flex size-12 items-center justify-center rounded-xl bg-primary/10 text-primary">
              <FileText className="size-6" />
            </div>
            <div className="space-y-0.5">
              <p className="font-semibold">Generated Reports</p>
              <p className="text-sm text-muted-foreground">
                Preview, download and share your previously generated reports.
              </p>
            </div>
          </div>
          <Button asChild variant="outline" className="gap-2">
            <Link href="/reports">
              <FileText /> View Report History
            </Link>
          </Button>
        </div>
      </div>
    </div>
  );
}
