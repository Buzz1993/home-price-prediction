"use client";

// Evaluation Tray. Holds properties staged from search results and lets the
// user pick which to compare, remove items, clear the tray, and trigger a
// comparison. Reproduces the Streamlit render_tray_column workflow (comparison
// needs at least 2 selected properties). Phase 18.2 presentation: no
// thumbnails — each staged row shows the property id (cardid) in small gray
// monospace above the name, with the price and AI recommendation score
// (looked up from the conversation's already-accumulated backend results —
// nothing is fetched or computed here), plus a selection progress section
// above the sticky actions. Behavior is unchanged.

import { Trash2, Scale, PackageOpen } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { EmptyState } from "@/components/ui/empty-state";
import { Spinner } from "@/components/ui/spinner";
import { cn } from "@/lib/utils";
import type { SearchResult } from "@/types/dashboard";
import { formatCr, formatScore } from "./format";
import { useWorkspace } from "./workspace-provider";

type EvaluationTrayProps = {
  // Optional override for the Compare action. When provided (Property
  // Comparison page), the selection runs through POST /compare instead of the
  // chat pipeline. Defaults to the Phase 4–6 chat behavior when omitted.
  onCompare?: (ids: string[]) => void;
  isComparing?: boolean;
};

// One staged row: the property id (cardid) in monospace above the bold name,
// price and AI score when the property is known from the conversation's
// accumulated results; a tray restored from localStorage whose conversation
// results are gone still shows the bare id.
function TrayItem({
  id,
  property,
  checked,
  onCheck,
  onRemove,
}: {
  id: string;
  property?: SearchResult;
  checked: boolean;
  onCheck: (checked: boolean) => void;
  onRemove: () => void;
}) {
  const name = property?.project_name || id;

  return (
    <div
      className={cn(
        "group flex items-center gap-3 rounded-xl border bg-card p-3 transition-all duration-200 hover:bg-accent hover:shadow-float",
        checked && "border-primary/60 bg-accent"
      )}
    >
      <Checkbox
        checked={checked}
        onCheckedChange={(state) => onCheck(state === true)}
        aria-label={`Select ${name} for comparison`}
      />

      <div className="min-w-0 flex-1">
        {/* cardid — small gray monospace line above the name. */}
        <p className="truncate font-mono text-[11px] leading-tight text-muted-foreground">
          {id}
        </p>
        {property?.project_name && (
          <p className="mt-0.5 truncate text-sm font-semibold leading-tight">
            {property.project_name}
          </p>
        )}
        {property && (
          <div className="mt-1 flex items-center gap-2.5 text-xs">
            <span className="font-semibold tabular-nums">
              {formatCr(property.price)}
            </span>
            {property.search_score !== undefined && (
              <span className="font-medium text-green-600">
                AI {formatScore(property.search_score)}
              </span>
            )}
          </div>
        )}
      </div>

      <Button
        variant="ghost"
        size="icon-sm"
        aria-label={`Remove ${name}`}
        onClick={onRemove}
        className="shrink-0 self-center text-muted-foreground opacity-60 transition-opacity hover:text-destructive group-hover:opacity-100"
      >
        <Trash2 />
      </Button>
    </div>
  );
}

export function EvaluationTray({ onCompare, isComparing }: EvaluationTrayProps = {}) {
  const {
    tray,
    selected,
    properties,
    toggleSelected,
    removeFromTray,
    clearTray,
    sendMessage,
    isSending,
  } = useWorkspace();

  const busy = isSending || isComparing === true;
  const canCompare = selected.length >= 2 && !busy;

  const runComparison = () => {
    if (onCompare) {
      onCompare(selected);
      return;
    }
    sendMessage("Compare selected properties from my tray", selected);
  };

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="shrink-0 border-b p-4">
        <h2 className="font-heading text-sm font-semibold">Evaluation Tray</h2>
        <p className="text-xs text-muted-foreground">
          Properties staged for analysis
        </p>
      </div>

      {tray.length === 0 ? (
        <div className="flex flex-1 items-center justify-center p-4">
          <EmptyState
            icon={PackageOpen}
            title="Your tray is empty"
            description="Stage properties from search results to compare and analyze them."
          />
        </div>
      ) : (
        <>
          {/* The staged list is the only scroller; header + actions stay fixed. */}
          <div className="min-h-0 flex-1 space-y-2 overflow-y-auto p-3">
            {tray.map((id) => (
              <TrayItem
                key={id}
                id={id}
                property={properties.find((p) => p.id === id)}
                checked={selected.includes(id)}
                onCheck={(checked) => toggleSelected(id, checked)}
                onRemove={() => removeFromTray(id)}
              />
            ))}
          </div>

          <div className="shrink-0 space-y-3 border-t bg-card/60 p-4">
            {/* Selection progress — counter + soft purple bar. Compare unlocks
                at 2 selected, so the bar fills against that threshold. */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">
                  Selected for comparison
                </span>
                <span className="font-semibold tabular-nums">
                  {selected.length} / {tray.length}
                </span>
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-muted">
                <div
                  className="bg-brand-gradient h-full rounded-full transition-all duration-300"
                  style={{
                    width: `${tray.length > 0 ? (selected.length / tray.length) * 100 : 0}%`,
                  }}
                />
              </div>
              {selected.length < 2 && (
                <p className="text-xs text-muted-foreground">
                  Select at least 2 properties to compare.
                </p>
              )}
            </div>

            <Button
              className="w-full"
              disabled={!canCompare}
              onClick={runComparison}
            >
              {isComparing ? <Spinner /> : <Scale />}
              {isComparing ? "Comparing…" : "Compare Properties"}
            </Button>
            <Button
              variant="outline"
              className="w-full"
              onClick={clearTray}
              disabled={busy}
            >
              <Trash2 /> Clear Tray
            </Button>
          </div>
        </>
      )}
    </div>
  );
}
