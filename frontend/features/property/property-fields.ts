// Dynamic field categorization for the Property Details page (Phase 14.4).
//
// The backend GET /property/{id} response is an open record — the frontend must
// render every field it returns and never assume a fixed schema. This module is
// pure logic (no JSX): it buckets backend keys into the documented logical
// sections (project_docs/04_UI.md) and leaves anything unmatched for
// "Additional Information", so the page adapts to future backend fields without
// code changes. No business logic lives here — the backend stays the source of
// truth; this only decides where each field is displayed.

export type FieldEntry = { key: string; value: unknown };

export type SectionId =
  | "pricing"
  | "overview"
  | "specifications"
  | "project"
  | "nearby"
  | "reviews"
  | "ratings"
  | "insights"
  | "location";

export type CategorizedSection = {
  id: SectionId;
  title: string;
  entries: FieldEntry[];
};

// A backend value is renderable when it is neither missing nor empty.
export function isEmptyValue(value: unknown): boolean {
  if (value === null || value === undefined) return true;
  if (typeof value === "string") return value.trim() === "";
  if (Array.isArray(value)) {
    return value.filter((v) => !isEmptyValue(v)).length === 0;
  }
  return false;
}

// Turn an arbitrary backend value into a list when it represents one (a real
// array, or a comma-separated string with more than one item), otherwise null.
export function toList(value: unknown): string[] | null {
  if (Array.isArray(value)) {
    const list = value.map((v) => String(v).trim()).filter(Boolean);
    return list.length > 0 ? list : null;
  }
  if (typeof value === "string" && value.includes(",")) {
    const list = value
      .split(",")
      .map((v) => v.trim())
      .filter(Boolean);
    return list.length > 1 ? list : null;
  }
  return null;
}

// Render a scalar backend value as display text.
export function formatValue(value: unknown): string {
  if (isEmptyValue(value)) return "—";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "number") return value.toLocaleString("en-IN");
  if (Array.isArray(value)) return value.map((v) => String(v)).join(", ");
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

// Ordered categorization rules. The first rule whose keyword is a substring of
// the (lower-cased) backend key wins, so more specific sections are listed
// first. The order here is also the display order of the sections on the page.
const SECTION_RULES: { id: SectionId; title: string; keywords: string[] }[] = [
  {
    id: "reviews",
    title: "Reviews",
    keywords: ["review", "positive", "improve", "pros", "cons", "things"],
  },
  {
    id: "ratings",
    title: "Ratings",
    keywords: ["rating"],
  },
  {
    id: "nearby",
    title: "Nearby Places",
    keywords: [
      "nearby",
      "school",
      "hospital",
      "metro",
      "railway",
      "station",
      "bus",
      "mall",
      "shopping",
      "park",
      "restaurant",
      "transport",
      "commut",
      "landmark",
      "distance",
      "connectivity",
      "market",
    ],
  },
  {
    id: "insights",
    title: "AI Insights",
    keywords: [
      "analysis",
      "insight",
      "predict",
      "risk",
      "rental",
      "growth",
      "advice",
      "advisor",
      "invest",
      "negotiat",
      "valuation",
      "forecast",
      "recommend",
      "roi",
      "yield",
    ],
  },
  {
    id: "project",
    title: "Project Information",
    keywords: [
      "builder",
      "developer",
      "project",
      "tower",
      "unit",
      "rera",
      "launch",
      "block",
      "phase",
    ],
  },
  {
    id: "pricing",
    title: "Pricing",
    keywords: [
      "price",
      "cost",
      "sqft",
      "maintenance",
      "registration",
      "booking",
      "deposit",
      "charge",
      "emi",
      "tax",
      "value",
      "budget",
      "brokerage",
      "loan",
    ],
  },
  {
    id: "overview",
    title: "Property Overview",
    keywords: [
      "property_type",
      "type",
      "status",
      "description",
      "configuration",
      "overview",
      "summary",
      "about",
      "possession",
      "transaction",
      "availability",
    ],
  },
  {
    id: "specifications",
    title: "Property Specifications",
    keywords: [
      "area",
      "floor",
      "carpet",
      "built",
      "furnish",
      "facing",
      "age",
      "construction",
      "ownership",
      "dimension",
      "bedroom",
      "bathroom",
      "balcon",
      "parking",
      "super",
      "bhk",
      "config",
      "bath",
      "bed",
    ],
  },
  {
    id: "location",
    title: "Location",
    keywords: [
      "location",
      "locality",
      "city",
      "address",
      "pincode",
      "zip",
      "region",
      "zone",
      "latitude",
      "longitude",
      "coordinate",
      "state",
    ],
  },
];

// Bucket every backend field (except those already rendered by a dedicated
// component, listed in `consumed`) into its logical section. Fields that match
// no rule are returned as `additional` so they always appear somewhere — a
// backend field is never silently dropped.
export function categorizeFields(
  record: Record<string, unknown>,
  consumed: Set<string>
): { sections: CategorizedSection[]; additional: FieldEntry[] } {
  const buckets = new Map<SectionId, FieldEntry[]>();
  const additional: FieldEntry[] = [];

  for (const [key, value] of Object.entries(record)) {
    if (consumed.has(key)) continue;
    if (isEmptyValue(value)) continue;

    const lower = key.toLowerCase();
    const rule = SECTION_RULES.find((r) =>
      r.keywords.some((word) => lower.includes(word))
    );

    if (rule) {
      const list = buckets.get(rule.id) ?? [];
      list.push({ key, value });
      buckets.set(rule.id, list);
    } else {
      additional.push({ key, value });
    }
  }

  const sections = SECTION_RULES.filter((r) => buckets.has(r.id)).map((r) => ({
    id: r.id,
    title: r.title,
    entries: buckets.get(r.id)!,
  }));

  return { sections, additional };
}
