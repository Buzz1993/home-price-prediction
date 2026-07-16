"use client";

// Copilot Workspace shell (Phase 15.13 → 15.15). The permanent three-column
// workspace:
//
//   Conversation Sidebar | Chat + Results | Right column (Map ↑ / Tray ↓)
//
// Phase 15.14 merged the former Map and Tray columns into a single right column
// that stacks the Property Map above the Evaluation Tray with a draggable
// vertical divider (see WorkspaceRightPanel). Phase 15.15 adds a draggable
// HORIZONTAL divider between the chat column and that right column, so users can
// widen the chat or the map/tray. Both resize systems are independent and work
// together; the horizontal width is clamped (right panel 25%–45% of the
// chat+right region) and persisted to localStorage. On large screens every
// column is visible at once and never reflows. Below xl, the chat is the primary
// column and the others open as slide-over sheets from a slim toolbar. Reused by
// the Dashboard (the single workspace entry point) and the retained /chat route.

import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { Map as MapIcon, PanelLeft, Scale } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { cn } from "@/lib/utils";
import { ChatWorkspace } from "./chat-workspace";
import { ConversationSidebar } from "./conversation-sidebar";
import { EvaluationTray } from "./evaluation-tray";
import { MapPanel } from "./map-panel";
import { WorkspaceRightPanel } from "./workspace-right-panel";
import { useWorkspace } from "./workspace-provider";

// Right panel horizontal bounds, as a fraction of the chat+right region (the
// fixed conversation sidebar is excluded from the measurement). Keeping the
// right panel ≤ 45% guarantees the chat keeps ≥ 55%, comfortably above its 45%
// minimum.
const RIGHT_MIN_RATIO = 0.25;
const RIGHT_MAX_RATIO = 0.45;
const RIGHT_DEFAULT_RATIO = 0.3;
const RIGHT_STORAGE_KEY = "estatemind:workspace-right-ratio";

// Clamp the right-panel pixel width to [25%, 45%] of the split container.
function clampRight(width: number, containerWidth: number): number {
  const min = containerWidth * RIGHT_MIN_RATIO;
  const max = containerWidth * RIGHT_MAX_RATIO;
  return Math.min(Math.max(width, min), max);
}

export function WorkspaceShell() {
  const { conversations, activeId } = useWorkspace();
  const [navOpen, setNavOpen] = useState(false);
  const [mapOpen, setMapOpen] = useState(false);
  const [trayOpen, setTrayOpen] = useState(false);

  // Horizontal split between the chat column and the right panel (xl only).
  const splitRef = useRef<HTMLDivElement>(null);
  const [rightWidth, setRightWidth] = useState<number | null>(null);
  const [hDragging, setHDragging] = useState(false);

  const activeTitle =
    conversations.find((c) => c.id === activeId)?.title ?? "EstateMind Copilot";

  // Seed the right-panel width from localStorage (or the default ratio) before
  // first paint, clamped to the current container width.
  useLayoutEffect(() => {
    const el = splitRef.current;
    if (!el) return;
    const width = el.getBoundingClientRect().width;
    if (width === 0) return;

    let ratio = RIGHT_DEFAULT_RATIO;
    try {
      const saved = window.localStorage.getItem(RIGHT_STORAGE_KEY);
      if (saved) ratio = Number(saved);
    } catch {
      // Ignore storage access errors (private mode, disabled storage, …).
    }
    setRightWidth(clampRight(ratio * width, width));
  }, []);

  // Re-clamp when the split container is resized (window resize, sidebar toggle)
  // so the 25%–45% bounds keep holding.
  useEffect(() => {
    const el = splitRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(() => {
      const width = el.getBoundingClientRect().width;
      if (width === 0) return;
      setRightWidth((w) => (w === null ? w : clampRight(w, width)));
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const startHDrag = useCallback((e: React.PointerEvent) => {
    e.preventDefault();
    setHDragging(true);
  }, []);

  // Follow the pointer globally while dragging. The right panel sits on the
  // right edge, so its width == distance from the pointer to the container's
  // right edge.
  useEffect(() => {
    if (!hDragging) return;

    const onMove = (e: PointerEvent) => {
      const el = splitRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      setRightWidth(clampRight(rect.right - e.clientX, rect.width));
    };

    const onUp = () => {
      setHDragging(false);
      const el = splitRef.current;
      setRightWidth((w) => {
        if (w !== null && el) {
          const width = el.getBoundingClientRect().width;
          if (width > 0) {
            try {
              window.localStorage.setItem(RIGHT_STORAGE_KEY, String(w / width));
            } catch {
              // Ignore storage write errors.
            }
          }
        }
        return w;
      });
    };

    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    const prevCursor = document.body.style.cursor;
    const prevSelect = document.body.style.userSelect;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";

    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      document.body.style.cursor = prevCursor;
      document.body.style.userSelect = prevSelect;
    };
  }, [hDragging]);

  return (
    // The workspace is locked to the viewport height. `overflow-hidden` here
    // means the browser window is NEVER the scroll container — each column owns
    // its own scroll (see ChatWorkspace / ConversationSidebar / EvaluationTray).
    // Phase 18.1: columns float as detached rounded panels on the soft slate
    // canvas (gutter padding + gaps) instead of touching border-separated rails.
    <div className="flex h-dvh w-full gap-3 overflow-hidden bg-background p-3">
      {/* Column 1 — Conversation Sidebar (inline on xl). Fixed full height;
          scrolls internally. Floating dark purple panel (the rail paints its
          own gradient — this wrapper only rounds and clips it). */}
      <div className="hidden w-[280px] shrink-0 overflow-hidden rounded-2xl border border-sidebar-border shadow-float xl:block">
        <ConversationSidebar />
      </div>

      {/* Chat + Right panel share the remaining width; the split container is
          what we measure for the horizontal resize. */}
      <div ref={splitRef} className="flex min-h-0 min-w-0 flex-1">
        {/* Column 2 — Chat + Results (always visible, primary column). A bounded
            flex column: fixed header + scrolling body + fixed input. min-h-0
            lets the inner list shrink so IT scrolls instead of the column
            growing. flex-1 absorbs whatever the right panel does not take. */}
        <section className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden rounded-2xl border bg-card shadow-float">
          {/* Fixed chat header — panel toggles on smaller screens, a clean title
              bar on xl where every column is already visible. Never scrolls. */}
          <div className="flex h-14 shrink-0 items-center gap-2 border-b px-4">
            <Button
              variant="ghost"
              size="icon"
              className="xl:hidden"
              aria-label="Open conversations"
              onClick={() => setNavOpen(true)}
            >
              <PanelLeft />
            </Button>

            <p className="min-w-0 flex-1 truncate font-heading text-sm font-semibold">
              {activeTitle}
            </p>

            <Button
              variant="ghost"
              size="icon"
              className="xl:hidden"
              aria-label="Open property map"
              onClick={() => setMapOpen(true)}
            >
              <MapIcon />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="xl:hidden"
              aria-label="Open evaluation tray"
              onClick={() => setTrayOpen(true)}
            >
              <Scale />
            </Button>
          </div>

          {/* ChatWorkspace is the flex-1 body; it owns the message/card scroll. */}
          <ChatWorkspace />
        </section>

        {/* Horizontal drag handle — col-resize cursor, highlights while
            hovering/dragging. Only shown on xl where both columns are inline.
            Rendered as a slim transparent gutter between the floating panels. */}
        <div
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize chat and right panel"
          onPointerDown={startHDrag}
          className={cn(
            "group relative hidden w-3 shrink-0 cursor-col-resize items-center justify-center transition-colors xl:flex",
            hDragging && "bg-primary/5"
          )}
        >
          <span
            className={cn(
              "h-10 w-1 rounded-full bg-border transition-colors group-hover:bg-primary/50",
              hDragging && "bg-primary"
            )}
          />
        </div>

        {/* Column 3 — Right column (inline on xl). A vertical stack: Property Map
            on top, Evaluation Tray below, with its own draggable divider. Fixed
            full height; each section scrolls internally. Width is driven by the
            horizontal resize (defaults to 24rem until measured). Transition only
            when NOT dragging so the live drag is instant and the release
            settles smoothly. */}
        <aside
          className={cn(
            "hidden shrink-0 xl:block xl:w-[24rem]",
            !hDragging && "transition-[width] duration-200 ease-out"
          )}
          style={rightWidth !== null ? { width: rightWidth } : undefined}
        >
          <WorkspaceRightPanel className="overflow-hidden rounded-2xl" />
        </aside>
      </div>

      {/* --- Slide-over panels for smaller screens ------------------------- */}
      <Sheet open={navOpen} onOpenChange={setNavOpen}>
        <SheetContent side="left" className="w-[85vw] p-0 sm:max-w-xs">
          <SheetHeader className="sr-only">
            <SheetTitle>Conversations</SheetTitle>
          </SheetHeader>
          <ConversationSidebar onNavigate={() => setNavOpen(false)} />
        </SheetContent>
      </Sheet>

      <Sheet open={mapOpen} onOpenChange={setMapOpen}>
        <SheetContent side="right" className="w-[90vw] p-0 sm:max-w-md">
          <SheetHeader className="sr-only">
            <SheetTitle>Property Map</SheetTitle>
          </SheetHeader>
          <MapPanel />
        </SheetContent>
      </Sheet>

      <Sheet open={trayOpen} onOpenChange={setTrayOpen}>
        <SheetContent side="right" className="w-[85vw] p-0 sm:max-w-sm">
          <SheetHeader className="sr-only">
            <SheetTitle>Evaluation Tray</SheetTitle>
          </SheetHeader>
          <EvaluationTray />
        </SheetContent>
      </Sheet>
    </div>
  );
}
