import { Brand } from "./brand";

// Shared sidebar header: the EstateMind brand with a subtle product badge.
// Reused by the global app sidebar and the Dashboard conversation sidebar so
// both rails introduce the product identically. Styled for the dark purple
// rail (Phase 18.2): light text, faint translucent badge.
export function SidebarHeader() {
  return (
    <div className="shrink-0 px-5 pb-4 pt-5">
      <Brand />
      <span className="mt-2 inline-flex items-center rounded-full border border-white/10 bg-white/5 px-2.5 py-0.5 text-[0.65rem] font-medium tracking-wide text-sidebar-foreground/70">
        AI Real Estate Copilot
      </span>
    </div>
  );
}
