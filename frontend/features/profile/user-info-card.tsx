// User information block for the Profile page. Presentational only — renders the
// account details returned by GET /profile as an initials avatar plus name, email
// and id.
//
// Phase 18.7: refined layout with better spacing and visual hierarchy.

import { Mail, User as UserIcon } from "lucide-react";

import type { User } from "@/types/profile";

// Derive up-to-two-letter initials from the user's name for the avatar.
function initials(name: string): string {
  return (
    name
      .split(" ")
      .filter(Boolean)
      .slice(0, 2)
      .map((word) => word[0]?.toUpperCase())
      .join("") || "U"
  );
}

export function UserInfoCard({ user }: { user: User }) {
  return (
    <div className="space-y-4 rounded-xl border bg-gradient-to-br from-muted/20 to-transparent p-5">
      <div className="flex items-center gap-5">
        <div className="flex size-16 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br from-primary via-primary/90 to-primary/70 text-xl font-bold text-primary-foreground shadow-lg ring-2 ring-background">
          {initials(user.name)}
        </div>
        <div className="min-w-0 flex-1 space-y-1.5">
          <p className="truncate text-lg font-semibold">{user.name}</p>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Mail className="size-4 shrink-0" />
            <p className="truncate">{user.email}</p>
          </div>
        </div>
      </div>
      {user.id && (
        <div className="flex items-center gap-2 rounded-lg bg-muted/50 px-3 py-2 text-xs">
          <UserIcon className="size-3.5 shrink-0 text-muted-foreground" />
          <span className="font-mono text-muted-foreground">ID: {user.id}</span>
        </div>
      )}
    </div>
  );
}
