"use client";

// Copilot Workspace shell (Phase 15.13). The permanent four-column workspace:
//
//   Conversation Sidebar | Chat + Results | Property Map | Evaluation Tray
//
// On large screens all four columns are visible at once and never reflow — the
// map and tray update live without changing the layout. Below xl, the chat is
// the primary column and the other three open as slide-over sheets from a slim
// toolbar, so nothing overflows or stacks awkwardly on smaller screens. Reused by
// the Dashboard and the AI Chat page so the layout is defined once.

import { useState } from "react";
import { Map as MapIcon, PanelLeft, Scale } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { ChatWorkspace } from "./chat-workspace";
import { ConversationSidebar } from "./conversation-sidebar";
import { EvaluationTray } from "./evaluation-tray";
import { MapPanel } from "./map-panel";
import { useWorkspace } from "./workspace-provider";

export function WorkspaceShell() {
  const { conversations, activeId } = useWorkspace();
  const [navOpen, setNavOpen] = useState(false);
  const [mapOpen, setMapOpen] = useState(false);
  const [trayOpen, setTrayOpen] = useState(false);

  const activeTitle =
    conversations.find((c) => c.id === activeId)?.title ?? "EstateMind Copilot";

  return (
    <div className="flex h-dvh w-full overflow-hidden bg-background">
      {/* Column 1 — Conversation Sidebar (inline on xl). */}
      <div className="hidden w-72 shrink-0 border-r xl:block">
        <ConversationSidebar />
      </div>

      {/* Column 2 — Chat + Results (always visible, primary column). */}
      <section className="flex min-w-0 flex-1 flex-col">
        {/* Slim toolbar — panel toggles on smaller screens; a clean title bar
            on xl where every column is already visible. */}
        <div className="flex h-14 items-center gap-2 border-b px-3">
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

        <div className="min-h-0 flex-1">
          <ChatWorkspace />
        </div>
      </section>

      {/* Column 3 — Interactive Property Map (inline on xl). */}
      <aside className="hidden w-[22rem] shrink-0 border-l xl:block">
        <MapPanel />
      </aside>

      {/* Column 4 — Evaluation Tray (inline on xl). */}
      <aside className="hidden w-72 shrink-0 border-l xl:block">
        <EvaluationTray />
      </aside>

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
