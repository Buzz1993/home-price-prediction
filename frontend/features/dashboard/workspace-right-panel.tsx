"use client";

// Workspace right column (Phase 15.14). A single vertical stack replacing the
// former two side-by-side columns:
//
//   Property Map   (top, always above the tray)
//   ── drag handle ──
//   Evaluation Tray (bottom, fills the remaining height)
//
// A draggable divider resizes the map; the tray automatically takes whatever is
// left. Only the map height is controlled — the tray stays flex-1 so there is
// never any layout jump. The resize is pointer-driven (no React state churn per
// frame beyond a single height update), so it stays smooth. The chosen height is
// clamped to [300px, 85% of the column] and persisted to localStorage so it is
// restored on refresh. This is layout-only: map, tray, conversation and backend
// logic are untouched.

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";

import { cn } from "@/lib/utils";
import { EvaluationTray } from "./evaluation-tray";
import { MapPanel } from "./map-panel";

const MIN_MAP = 300; // px — map never shrinks below this.
const MAX_RATIO = 0.85; // map never exceeds 85% of the column height.
const DEFAULT_RATIO = 0.58; // pleasant default: map a little larger than tray.
const STORAGE_KEY = "estatemind:workspace-map-height";

// Keep the map height within [MIN_MAP, 85% of the column]. The tray absorbs the
// rest, so it is implicitly bounded to the remaining height.
function clampHeight(height: number, containerHeight: number): number {
  const max = Math.max(MIN_MAP, containerHeight * MAX_RATIO);
  return Math.min(Math.max(height, MIN_MAP), max);
}

export function WorkspaceRightPanel({ className }: { className?: string }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [mapHeight, setMapHeight] = useState<number | null>(null);
  const [dragging, setDragging] = useState(false);

  // Measure the column before first paint and seed the map height from
  // localStorage (or the default ratio), clamped to the current column size.
  useLayoutEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const containerHeight = container.getBoundingClientRect().height;
    if (containerHeight === 0) return;

    let initial = containerHeight * DEFAULT_RATIO;
    try {
      const saved = window.localStorage.getItem(STORAGE_KEY);
      if (saved) initial = Number(saved);
    } catch {
      // Ignore storage access errors (private mode, disabled storage, …).
    }
    setMapHeight(clampHeight(initial, containerHeight));
  }, []);

  // Re-clamp when the column itself is resized (window resize, sheet open, …)
  // so the map keeps respecting the 85% cap and the tray never disappears.
  useEffect(() => {
    const container = containerRef.current;
    if (!container || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(() => {
      const containerHeight = container.getBoundingClientRect().height;
      if (containerHeight === 0) return;
      setMapHeight((h) =>
        h === null ? h : clampHeight(h, containerHeight)
      );
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  const startDrag = useCallback((e: React.PointerEvent) => {
    e.preventDefault();
    setDragging(true);
  }, []);

  // While dragging, follow the pointer globally so the drag continues even if
  // the cursor leaves the thin divider. Pointer math is relative to the column
  // top, so the map height == distance from the top to the pointer.
  useEffect(() => {
    if (!dragging) return;

    const onMove = (e: PointerEvent) => {
      const container = containerRef.current;
      if (!container) return;
      const rect = container.getBoundingClientRect();
      setMapHeight(clampHeight(e.clientY - rect.top, rect.height));
    };

    const onUp = () => {
      setDragging(false);
      setMapHeight((h) => {
        if (h !== null) {
          try {
            window.localStorage.setItem(STORAGE_KEY, String(Math.round(h)));
          } catch {
            // Ignore storage write errors.
          }
        }
        return h;
      });
    };

    // Lock the cursor and disable text selection / page scroll for the whole
    // gesture so nothing flickers or scrolls while resizing.
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    const prevCursor = document.body.style.cursor;
    const prevSelect = document.body.style.userSelect;
    document.body.style.cursor = "row-resize";
    document.body.style.userSelect = "none";

    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      document.body.style.cursor = prevCursor;
      document.body.style.userSelect = prevSelect;
    };
  }, [dragging]);

  return (
    <div
      ref={containerRef}
      className={cn("flex h-full min-h-0 flex-col", className)}
    >
      {/* Map — fixed to the chosen height. Transition only when NOT dragging so
          the live drag is instant (60fps) but the release settles smoothly.
          Floating rounded card (Phase 18.1). */}
      <div
        className={cn(
          "min-h-0 shrink-0 overflow-hidden rounded-2xl border bg-card shadow-float",
          !dragging && "transition-[height] duration-200 ease-out"
        )}
        style={mapHeight !== null ? { height: mapHeight } : undefined}
      >
        <MapPanel />
      </div>

      {/* Drag handle — row-resize cursor, highlights while hovering/dragging.
          A slim transparent gutter between the two floating cards. */}
      <div
        role="separator"
        aria-orientation="horizontal"
        aria-label="Resize property map"
        onPointerDown={startDrag}
        className="group relative flex h-3 shrink-0 cursor-row-resize items-center justify-center"
      >
        <span
          className={cn(
            "h-1 w-10 rounded-full bg-border transition-colors group-hover:bg-primary/50",
            dragging && "bg-primary"
          )}
        />
      </div>

      {/* Evaluation Tray — fills the remaining height; scrolls internally.
          Floating rounded card (Phase 18.1). */}
      <div
        data-tour="evaluation-tray"
        className="min-h-0 flex-1 overflow-hidden rounded-2xl border bg-card shadow-float"
      >
        <EvaluationTray />
      </div>
    </div>
  );
}
