// User information block for the Profile page. Presentational only — renders the
// account details returned by GET /profile as an initials avatar plus name, email
// and id.

import { Mail } from "lucide-react";

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
    <div className="flex items-center gap-4">
      <div className="flex size-14 shrink-0 items-center justify-center rounded-full bg-primary/10 text-lg font-semibold text-primary">
        {initials(user.name)}
      </div>
      <div className="min-w-0 space-y-0.5">
        <p className="truncate font-medium">{user.name}</p>
        <p className="flex items-center gap-1 truncate text-sm text-muted-foreground">
          <Mail className="size-4 shrink-0" />
          {user.email}
        </p>
        {user.id && (
          <p className="truncate font-mono text-xs text-muted-foreground">
            ID: {user.id}
          </p>
        )}
      </div>
    </div>
  );
}
