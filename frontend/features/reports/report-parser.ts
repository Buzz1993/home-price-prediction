// Presentation-only report parser (Phase 15.17). Turns the structured
// plain-text investment report returned by the backend (divider banners, icon
// section titles, "Label: Value" lines, '•/✓/□' bullets and '|' tables) into a
// small document model that report-document.tsx renders as cards, tables and
// checklists. Every label, value and bullet is carried through VERBATIM — no
// report content is created, computed or altered here. When the text does not
// match the expected structure (e.g. the non-enhanced backend fallback report)
// parseReport returns null and the preview falls back to raw text.

export type ReportPair = { label: string; value: string };

export type ReportBlock =
  | { type: "bullets"; items: { marker: string; text: string }[] }
  | { type: "table"; header: string[]; rows: string[][] }
  | { type: "pairs"; pairs: ReportPair[] }
  | { type: "label"; text: string }
  | { type: "text"; lines: string[] };

export type ReportSubsection = {
  icon: string | null;
  title: string;
  blocks: ReportBlock[];
};

export type PropertySection = {
  kind: "property";
  number: number;
  name: string;
  location: string;
  subsections: ReportSubsection[];
};

export type GenericSection = {
  kind: "generic";
  icon: string | null;
  title: string;
  blocks: ReportBlock[];
};

export type ReportSection = PropertySection | GenericSection;

export type ReportModel = {
  title: string;
  generatedFor: string | null;
  reportDate: string | null;
  sections: ReportSection[];
};

// A banner/divider line (═══ or ━━━) drawn by the report. Lenient: any run of
// box-drawing characters counts, so a stray glyph never breaks the layout.
const DIVIDER = /^[─-╿]{5,}\s*$/;

// "🏠 PROPERTY 1" banner opening a per-property block.
const PROPERTY_BANNER = /^🏠?\s*PROPERTY\s+(\d+)\b/i;

// The report's fixed icon set (see the backend report template).
const SECTION_ICONS = [
  "📌", "🏠", "📈", "💰", "🏷️", "🏷", "⚠️", "⚠",
  "🚇", "🤝", "⭐", "🏆", "📊", "📋", "✅",
];

// '•' bullets, '✓' reasons, '□' checklist items and '-' sub-bullets.
const BULLET = /^([•✓✗□☐▪–-])\s*(.+)$/;

// "Label: Value" line — short plain-word label, value after the first colon.
const PAIR = /^([A-Za-z][^:|]{0,48}?):\s*(.*)$/;

// Standalone labels the report prints on their own line (value follows below).
const STANDALONE_LABELS = new Set([
  "property",
  "reason",
  "recommendation",
  "runner up",
  "why choose it",
  "proceed with caution",
  "avoid",
  "final recommendation",
  "recommended property",
  "investment scorecard",
  "suitable for",
  "action",
]);

function matchIcon(line: string): { icon: string; title: string } | null {
  for (const icon of SECTION_ICONS) {
    if (line.startsWith(icon)) {
      const title = line.slice(icon.length).trim();
      if (title) return { icon, title };
    }
  }
  return null;
}

function splitTableRow(line: string): string[] {
  const cells = line.split("|").map((cell) => cell.trim());
  // Drop the empty edges produced by leading/trailing pipes.
  if (cells.length && cells[0] === "") cells.shift();
  if (cells.length && cells[cells.length - 1] === "") cells.pop();
  return cells;
}

function isSeparatorRow(cells: string[]): boolean {
  return cells.every((cell) => /^:?-{2,}:?$/.test(cell) || cell === "");
}

// Convert the raw content lines of one section into typed blocks, grouping
// consecutive lines of the same shape (pairs, bullets, table rows).
function linesToBlocks(lines: string[]): ReportBlock[] {
  const blocks: ReportBlock[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];
    if (!line) {
      i++;
      continue;
    }

    if (line.startsWith("|")) {
      const rows: string[][] = [];
      while (i < lines.length && lines[i].startsWith("|")) {
        const cells = splitTableRow(lines[i]);
        if (cells.length && !isSeparatorRow(cells)) rows.push(cells);
        i++;
      }
      if (rows.length) {
        blocks.push({ type: "table", header: rows[0], rows: rows.slice(1) });
      }
      continue;
    }

    const bullet = line.match(BULLET);
    if (bullet) {
      const items: { marker: string; text: string }[] = [];
      while (i < lines.length) {
        const m = lines[i].match(BULLET);
        if (!m) break;
        items.push({ marker: m[1], text: m[2] });
        i++;
      }
      blocks.push({ type: "bullets", items });
      continue;
    }

    // "✅ Strengths:" / "⚠ Weaknesses:" style mini headings.
    if (/^.{1,40}:$/.test(line)) {
      blocks.push({ type: "label", text: line.slice(0, -1).trim() });
      i++;
      continue;
    }

    if (STANDALONE_LABELS.has(line.toLowerCase())) {
      blocks.push({ type: "label", text: line });
      i++;
      continue;
    }

    const pair = line.match(PAIR);
    if (pair && pair[2]) {
      const pairs: ReportPair[] = [];
      while (i < lines.length) {
        const m = lines[i].match(PAIR);
        if (!m || !m[2]) break;
        pairs.push({ label: m[1].trim(), value: m[2].trim() });
        i++;
      }
      blocks.push({ type: "pairs", pairs });
      continue;
    }

    const texts: string[] = [];
    while (
      i < lines.length &&
      lines[i] &&
      !lines[i].startsWith("|") &&
      !BULLET.test(lines[i]) &&
      !PAIR.test(lines[i]) &&
      !STANDALONE_LABELS.has(lines[i].toLowerCase())
    ) {
      texts.push(lines[i]);
      i++;
    }
    if (texts.length) blocks.push({ type: "text", lines: texts });
    else i++; // safety: never stall
  }

  return blocks;
}

// Find "Generated for" / "Report Date" in the header and return the line below.
function valueAfterLabel(lines: string[], label: string): string | null {
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].toLowerCase().replace(/:$/, "");
    if (line === label) return lines[i + 1] ?? null;
    // Inline form, e.g. "Report Date: 15 July 2026".
    if (line.startsWith(`${label}:`)) {
      return lines[i].slice(lines[i].indexOf(":") + 1).trim() || null;
    }
  }
  return null;
}

export function parseReport(text: string): ReportModel | null {
  if (!text || !text.trim()) return null;

  const lines = text.split(/\r?\n/).map((line) => line.trim());
  let i = 0;
  while (i < lines.length && !lines[i]) i++;

  // Document header: the first divider-bounded block (title + meta labels).
  if (i >= lines.length || !DIVIDER.test(lines[i])) return null;
  i++;
  const headerLines: string[] = [];
  while (i < lines.length && !DIVIDER.test(lines[i])) {
    if (lines[i]) headerLines.push(lines[i]);
    i++;
  }
  if (i >= lines.length || !headerLines.length) return null;
  i++; // closing divider

  const title = headerLines[0];
  const generatedFor = valueAfterLabel(headerLines, "generated for");
  const reportDate = valueAfterLabel(headerLines, "report date");

  const sections: ReportSection[] = [];
  let currentGeneric: GenericSection | null = null;
  let currentProperty: PropertySection | null = null;
  let currentSub: ReportSubsection | null = null;
  let buffer: string[] = [];

  const flushBuffer = () => {
    const blocks = linesToBlocks(buffer);
    buffer = [];
    if (!blocks.length) return;
    if (currentSub) currentSub.blocks.push(...blocks);
    else if (currentProperty) {
      currentProperty.subsections.push({ icon: null, title: "", blocks });
    } else if (currentGeneric) currentGeneric.blocks.push(...blocks);
  };

  while (i < lines.length) {
    const line = lines[i];
    if (!line) {
      buffer.push(line);
      i++;
      continue;
    }

    if (DIVIDER.test(line)) {
      // A banner is divider / 1-4 lines / divider. Property banners carry the
      // name and location; section banners carry one icon+title line.
      const banner: string[] = [];
      let j = i + 1;
      while (j < lines.length && banner.length < 4 && !DIVIDER.test(lines[j])) {
        if (lines[j]) banner.push(lines[j]);
        j++;
      }
      const closed = j < lines.length && DIVIDER.test(lines[j]);

      if (closed && banner.length) {
        flushBuffer();
        const property = banner[0].match(PROPERTY_BANNER);
        if (property) {
          currentGeneric = null;
          currentSub = null;
          currentProperty = {
            kind: "property",
            number: parseInt(property[1], 10),
            name: banner[1] ?? "",
            location: banner.slice(2).join(", "),
            subsections: [],
          };
          sections.push(currentProperty);
        } else {
          const iconTitle = matchIcon(banner[0]);
          currentProperty = null;
          currentSub = null;
          currentGeneric = {
            kind: "generic",
            icon: iconTitle?.icon ?? null,
            title: iconTitle?.title ?? banner[0],
            blocks: [],
          };
          sections.push(currentGeneric);
          buffer.push(...banner.slice(1));
        }
        i = j + 1;
        continue;
      }

      i++; // stray divider — skip
      continue;
    }

    // Icon titles inside a property block start a new subsection (lines ending
    // with ':' such as "✅ Strengths:" are content, not subsection titles).
    const iconTitle = matchIcon(line);
    if (iconTitle && !line.endsWith(":") && currentProperty) {
      flushBuffer();
      currentSub = { icon: iconTitle.icon, title: iconTitle.title, blocks: [] };
      currentProperty.subsections.push(currentSub);
      i++;
      continue;
    }

    buffer.push(line);
    i++;
  }
  flushBuffer();

  if (!sections.length) return null;
  return { title, generatedFor, reportDate, sections };
}
