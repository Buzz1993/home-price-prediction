"use client";

// Evaluation Tray. Holds properties staged from search results and lets the
// user pick which to compare, remove items, clear the tray, and trigger a
// comparison. Reproduces the Streamlit render_tray_column workflow (comparison
// needs at least 2 selected properties). Further tray actions (price
// prediction, rental, valuation, …) arrive in later phases.

import { Trash2, Scale } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { useWorkspace } from "./workspace-provider";

export function EvaluationTray() {
  const { tray, selected, toggleSelected, removeFromTray, clearTray, sendMessage, isSending } =
    useWorkspace();

  const canCompare = selected.length >= 2 && !isSending;

  const runComparison = () => {
    sendMessage("Compare selected properties from my tray", selected);
  };

  return (
    <div className="flex h-full flex-col">
      <div className="border-b p-4">
        <h2 className="font-heading text-sm font-semibold">
          Evaluation Tray
        </h2>
        <p className="text-xs text-muted-foreground">
          Properties staged for analysis
        </p>
      </div>

      {tray.length === 0 ? (
        <div className="flex flex-1 items-center justify-center p-4">
          <p className="text-center text-sm text-muted-foreground">
            Your tray is empty. Stage properties from search results to compare
            and analyze them.
          </p>
        </div>
      ) : (
        <>
          <div className="flex-1 space-y-2 overflow-y-auto p-4">
            {tray.map((id) => (
              <div
                key={id}
                className="flex items-center gap-2 rounded-lg border p-2"
              >
                <Checkbox
                  checked={selected.includes(id)}
                  onCheckedChange={(checked) =>
                    toggleSelected(id, checked === true)
                  }
                  aria-label={`Select ${id} for comparison`}
                />
                <span className="min-w-0 flex-1 truncate font-mono text-xs">
                  {id}
                </span>
                <Button
                  variant="ghost"
                  size="icon-sm"
                  aria-label={`Remove ${id}`}
                  onClick={() => removeFromTray(id)}
                >
                  <Trash2 />
                </Button>
              </div>
            ))}
          </div>

          <div className="space-y-2 border-t p-4">
            <p className="text-xs text-muted-foreground">
              Selected for comparison: {selected.length} of {tray.length}
            </p>
            <Button
              className="w-full"
              disabled={!canCompare}
              onClick={runComparison}
            >
              <Scale /> Compare Properties
            </Button>
            <Button
              variant="outline"
              className="w-full"
              onClick={clearTray}
              disabled={isSending}
            >
              <Trash2 /> Clear Tray
            </Button>
            <p className="text-xs text-muted-foreground">
              Tip: select at least 2 properties, then compare.
            </p>
          </div>
        </>
      )}
    </div>
  );
}
