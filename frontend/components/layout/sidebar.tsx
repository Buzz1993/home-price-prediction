import { cn } from "@/lib/utils";
import { Brand } from "./brand";
import { SidebarNav } from "./sidebar-nav";

export function Sidebar({ className }: { className?: string }) {
  return (
    <aside className={cn("flex w-64 flex-col border-r bg-sidebar", className)}>
      <div className="flex h-16 items-center border-b px-5">
        <Brand />
      </div>
      <SidebarNav />
      <div className="border-t p-4 text-xs text-muted-foreground">
        EstateMind Copilot
      </div>
    </aside>
  );
}
