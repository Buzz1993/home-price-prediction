"use client";

// Property selector for the Property Comparison workspace (Phase 17.0). Three
// dropdown slots choose which staged properties to compare — every property
// currently in the shared Evaluation Tray is a selectable option, without
// forcing the tray's own comparison tick-selection. Minimum 2, maximum 3;
// Show Comparison enables at 2. UI state only — nothing is fetched here.

import { Check, ChevronDown, Plus, Scale, X } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Spinner } from "@/components/ui/spinner";
import { useWorkspace } from "@/features/dashboard/workspace-provider";
import { cn } from "@/lib/utils";

const MAX_COMPARE = 3;
const SLOT_LABELS = [
  "Select property",
  "Select second property",
  "Select third property",
];

export function CompareSelector({
  chosen,
  onChange,
  onShow,
  isComparing,
}: {
  chosen: string[];
  onChange: (ids: string[]) => void;
  onShow: () => void;
  isComparing: boolean;
}) {
  const { tray, properties } = useWorkspace();

  const resolveName = (id: string) =>
    properties.find((p) => p.id === id)?.project_name || id;

  // Options still available for a slot (staged and not chosen elsewhere).
  const available = tray.filter((id) => !chosen.includes(id));
  const canShow = chosen.length >= 2 && !isComparing;

  const pick = (slot: number, id: string) => {
    if (slot < chosen.length) {
      onChange(chosen.map((current, i) => (i === slot ? id : current)));
    } else {
      onChange([...chosen, id]);
    }
  };

  const remove = (id: string) => onChange(chosen.filter((c) => c !== id));

  return (
    <section className="rounded-xl border bg-card p-4 shadow-float transition-shadow duration-200 hover:shadow-float-lg">
      <h2 className="flex items-center gap-2.5 font-heading text-sm font-semibold">
        <span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-primary/10">
          <Scale className="size-4 text-primary" />
        </span>
        Compare Properties
      </h2>
      <p className="mt-1 text-xs text-muted-foreground">
        Choose 2–3 properties staged in your Evaluation Tray, then show the
        comparison.
      </p>

      <div className="mt-3 flex flex-wrap items-center gap-2">
        {Array.from({ length: MAX_COMPARE }, (_, slot) => {
          const id = chosen[slot];
          const filled = id !== undefined;
          // Slots fill in order; later slots stay disabled until reached.
          const reachable = slot <= chosen.length;
          const options = available;

          return (
            <DropdownMenu key={slot}>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="outline"
                  disabled={!reachable || (!filled && options.length === 0)}
                  className={cn(
                    "h-auto min-w-52 justify-between gap-2 px-3 py-2 text-left",
                    filled && "border-primary/40 bg-primary/5"
                  )}
                >
                  <span className="flex min-w-0 items-center gap-2">
                    {filled ? (
                      <Check className="size-4 shrink-0 text-primary" />
                    ) : (
                      <Plus className="size-4 shrink-0 text-muted-foreground" />
                    )}
                    <span
                      className={cn(
                        "truncate text-sm",
                        filled ? "font-medium" : "text-muted-foreground"
                      )}
                    >
                      {filled ? resolveName(id) : SLOT_LABELS[slot]}
                    </span>
                  </span>
                  <ChevronDown className="size-4 shrink-0 text-muted-foreground" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="start" className="max-h-72 overflow-y-auto">
                {options.map((option) => (
                  <DropdownMenuItem
                    key={option}
                    onSelect={() => pick(slot, option)}
                  >
                    <span className="min-w-0 truncate">
                      {resolveName(option)}
                    </span>
                  </DropdownMenuItem>
                ))}
                {options.length === 0 && !filled && (
                  <DropdownMenuItem disabled>
                    No more staged properties
                  </DropdownMenuItem>
                )}
                {filled && (
                  <>
                    {options.length > 0 && <DropdownMenuSeparator />}
                    <DropdownMenuItem
                      variant="destructive"
                      onSelect={() => remove(id)}
                    >
                      <X /> Remove {resolveName(id)}
                    </DropdownMenuItem>
                  </>
                )}
              </DropdownMenuContent>
            </DropdownMenu>
          );
        })}

        <Button disabled={!canShow} onClick={onShow}>
          {isComparing ? <Spinner /> : <Scale />}
          {isComparing ? "Comparing…" : "Show Comparison"}
        </Button>
      </div>

      {tray.length > 0 && tray.length < 2 && (
        <p className="mt-2 text-xs text-muted-foreground">
          Stage at least 2 properties in the tray to enable the comparison.
        </p>
      )}
    </section>
  );
}
