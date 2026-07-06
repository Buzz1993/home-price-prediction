"use client";

// Comparison mutation for the dedicated Property Comparison page. Runs the
// documented POST /compare endpoint (not the chat pipeline) so the page owns a
// single, self-contained comparison request. All ranking/scoring stays in the
// backend — this only manages the request state.

import { useMutation } from "@tanstack/react-query";

import { comparePropertiesRequest } from "@/services/compare-service";

export function useComparison() {
  return useMutation({ mutationFn: comparePropertiesRequest });
}
