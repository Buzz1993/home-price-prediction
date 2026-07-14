import { Brand } from "./brand";

// Shared sidebar header: the EstateMind brand with a subtle product badge.
// Reused by the global app sidebar and the Dashboard conversation sidebar so
// both rails introduce the product identically.
export function SidebarHeader() {
  return (
    <div className="shrink-0 px-5 pb-4 pt-5">
      <Brand />
      <span className="mt-2 inline-flex items-center rounded-full border border-border/60 bg-muted/50 px-2.5 py-0.5 text-[0.65rem] font-medium tracking-wide text-muted-foreground">
        AI Real Estate Copilot
      </span>
    </div>
  );
}
