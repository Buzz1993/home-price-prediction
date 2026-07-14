# ===============================
# src/llm/prompts/report_prompt.py
# ===============================
#
# Prompt builder for the PROFESSIONAL INVESTMENT REPORT.
#
# Input: the existing backend-generated report (Markdown text produced by
# tools.create_property_report) plus, when provided, the structured results of
# the EXISTING backend analyses for the SAME selected properties (property
# overview, price prediction, rental, valuation, advisor — which carries the
# risk and future-growth fields — negotiation and, for 2+ properties, the
# comparison). This builder asks the LLM to re-present all of that as ONE
# structured, premium investment report.
#
# It performs no report generation, analysis or reasoning of its own:
#   - Every figure, verdict, score and flag comes from the backend data.
#   - Missing analyses are simply omitted; nothing is ever invented.
#   - The backend report pipeline (create_property_report) is untouched — this
#     only redesigns how the report is PRESENTED.

from datetime import date

from src.llm.prompts.config import PromptConfig, DEFAULT_CONFIG
from src.llm.prompts.templates import Prompt, build_prompt
from src.llm.prompts.formatting import format_records, format_value


def _indian_grouping(value: float) -> str:
    """Format a rupee amount with Indian digit grouping (12,34,567)."""
    whole = int(round(value))
    s = str(abs(whole))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts + [tail])
    return f"-₹{s}" if whole < 0 else f"₹{s}"


def _format_rent_fields(records: list[dict]) -> list[dict]:
    """
    PRESENTATION ONLY: pre-render the rupee rent figures as display strings
    (Indian grouping + the correct Lakh form) so the LLM copies them verbatim
    instead of converting units itself. No value is changed — only rendered.
    """
    formatted = []
    for record in records:
        row = dict(record)
        for key in ("monthly_rent_estimate", "monthly_rent"):
            value = row.get(key)
            if isinstance(value, (int, float)):
                row[key] = f"{_indian_grouping(value)}/month"
        annual = row.get("annual_rent")
        if isinstance(annual, (int, float)):
            lakh = annual / 100_000
            row["annual_rent"] = (
                f"{_indian_grouping(annual)}/year (≈{lakh:.2f} Lakh/year)"
            )
        formatted.append(row)
    return formatted


# The per-property analysis blocks handed to the LLM, in presentation order.
# "advisor" also carries the Risk Analysis fields (risk_label, risk_score) and
# the Future Growth fields (growth_label, growth_reason) produced during
# enrichment, so those two report sections draw from it.
REPORT_SECTIONS = (
    ("overview", "Property Overview (backend property details)"),
    ("prediction", "Price Prediction results"),
    ("rental", "Rental Analysis results"),
    ("valuation", "Property Valuation results"),
    (
        "advisor",
        "Investment Advisor results (also contains the Risk fields "
        "risk_label / risk_score and the Future Growth fields "
        "growth_label / growth_reason)",
    ),
    ("negotiation", "Negotiation Strategy results"),
    ("comparison", "Property Comparison results (winner and rankings)"),
)


# Section dividers used to build visual hierarchy WITHOUT Markdown.
DIVIDER_HEAVY = "══════════════════════════════════════"
DIVIDER_LIGHT = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# The exact document layout the report must follow — a premium consulting
# report (CBRE / JLL / Knight Frank style) rendered as PLAIN TEXT: divider
# lines, icons and whitespace create the hierarchy. The output must never
# contain Markdown syntax; it should read like a finished PDF/Word document,
# not Markdown source.
REPORT_TEMPLATE = (
    "Report structure (follow EXACTLY, in this order, as PLAIN TEXT — no "
    "Markdown anywhere):\n"
    "\n"
    f"{DIVIDER_HEAVY}\n"
    "        EstateMind Property Investment Report\n"
    "\n"
    "Generated for\n"
    "<the selected properties (project names)>\n"
    "\n"
    "Report Date\n"
    "<the date provided below>\n"
    f"{DIVIDER_HEAVY}\n"
    "\n"
    f"{DIVIDER_LIGHT}\n"
    "📌 Executive Summary\n"
    f"{DIVIDER_LIGHT}\n"
    "4-6 concise '•' bullets, drawn only from the backend results below: "
    "number of selected properties; best investment opportunity (the backend "
    "comparison winner); highest growth opportunity; highest rental "
    "opportunity; biggest risk; overall recommendation.\n"
    "\n"
    "Then, FOR EVERY property (one full block per property, in the order "
    "given — if there are N properties there must be exactly N blocks, never "
    "skip one), the block opens with:\n"
    f"{DIVIDER_HEAVY}\n"
    "🏠 PROPERTY <n>\n"
    "<Project Name>\n"
    "<Location>\n"
    f"{DIVIDER_HEAVY}\n"
    "\n"
    "…followed by these sections. Each section is its icon + plain-text "
    "title on its own line, a blank line, then compact 'Label: Value' lines "
    "or '•' bullets (labels are plain words — never wrapped in any "
    "formatting characters):\n"
    "\n"
    "🏠 Property Overview\n"
    "  Property ID / Project Name / Builder / Location / Property Type / "
    "Configuration (BHK, bed, bath) / Area / Asking Price.\n"
    "\n"
    "📈 Price Prediction\n"
    "  Status: 🟢 Undervalued / 🔴 Overpriced / 🟡 Fairly Priced (pick from "
    "the backend's own status/flag); Asking Price; Predicted Price; "
    "Difference; Recommendation: one bullet.\n"
    "\n"
    "💰 Rental Analysis\n"
    "  Monthly Rent; Annual Rent; Rental Yield; Demand; Rental Strategy; "
    "Recommendation: one bullet.\n"
    "\n"
    "🏷 Property Valuation\n"
    "  Current Price; Valuation Status (the backend analysis flag/message); "
    "Confidence (only if available, else omit); Recommendation: one bullet.\n"
    "\n"
    "⚠ Risk Analysis\n"
    "  Risk Level; Risk Score (if available); Key Concerns (from the "
    "backend); Recommendation: one bullet.\n"
    "\n"
    "🚇 Future Growth\n"
    "  Growth Potential (the backend growth label); Infrastructure Drivers "
    "(only those the backend names); Recommendation: one bullet. Never state "
    "an expected appreciation figure unless the backend provides one.\n"
    "\n"
    "🤝 Negotiation Strategy\n"
    "  Target Price; Suggested Discount; Negotiation Strength; Key Talking "
    "Points (from the backend); Recommendation: one bullet.\n"
    "\n"
    "⭐ Investment Advisor\n"
    "  Investment Rating; Verdict; then:\n"
    "  ✅ Strengths — 2-4 short bullets (only from backend values)\n"
    "  ⚠ Weaknesses — 2-4 short bullets (only from backend values)\n"
    "  Recommendation: one bullet restating the backend verdict.\n"
    "\n"
    "After ALL property blocks, ONLY when more than one property is "
    "covered:\n"
    f"{DIVIDER_LIGHT}\n"
    "📊 Comparison Summary\n"
    f"{DIVIDER_LIGHT}\n"
    "A clean aligned table (plain '|' columns are fine; no separator row of "
    "dashes) with categories as ROWS and one COLUMN per property — rows: "
    "Asking Price, Predicted Price, Rental Yield, Risk, Growth, Investment "
    "Rating, Verdict. NO explanations under the table.\n"
    "\n"
    "Investment Scorecard\n"
    "A small aligned table — one row per category (Price Value, Rental "
    "Potential, Future Growth, Risk, Negotiation Opportunity, Overall "
    "Investment), one column per property, cells are 1-5 stars (e.g. ★★★★☆). "
    "Stars must ONLY reflect the backend's relative results (the property "
    "the backend rates better on a category gets more stars; the backend "
    "winner gets the higher Overall Investment).\n"
    "\n"
    f"{DIVIDER_LIGHT}\n"
    "🏆 Best Overall Investment\n"
    f"{DIVIDER_LIGHT}\n"
    "Property\n"
    "<the backend comparison winner>\n"
    "\n"
    "Reason\n"
    "'✓ …' bullets using only backend results (score, valuation, rental, "
    "growth, risk — include only those where the winner is actually "
    "better).\n"
    "\n"
    "Recommendation\n"
    "Proceed with this property.\n"
    "\n"
    "Runner Up\n"
    "Property name; 'Suitable For:' 1-3 bullets derived only from backend "
    "fields; 'Reason:' 1-2 short bullets on why it ranked second.\n"
    "\n"
    "Finish with (always):\n"
    f"{DIVIDER_HEAVY}\n"
    "🏆 Final Investment Decision\n"
    f"{DIVIDER_HEAVY}\n"
    "Recommended Property\n"
    "<backend winner>\n"
    "\n"
    "Why Choose It\n"
    "2-4 '✓ …' bullets (backend values only).\n"
    "\n"
    "Proceed With Caution\n"
    "<other property> with 1-3 '• …' reason bullets where the backend flags "
    "concerns.\n"
    "\n"
    "Avoid\n"
    "ONLY if the backend's own verdict says to avoid; otherwise omit this "
    "label entirely.\n"
    "\n"
    "Final Recommendation\n"
    "ONE concise sentence based only on backend results.\n"
    "(With a single property, base the decision on its own backend verdict "
    "instead of a comparison.)\n"
    "\n"
    f"{DIVIDER_LIGHT}\n"
    "📋 Suggested Next Steps\n"
    f"{DIVIDER_LIGHT}\n"
    "Maximum 5 checkbox bullets ('□ …'), practical actions grounded in the "
    "backend results (e.g. '□ Negotiate toward the target price of ₹X.XX "
    "Cr', '□ Visit the property', '□ Verify legal documents').\n"
)


REPORT_RULES = (
    "Report rules:\n"
    "- Every section is CONCISE bullet points ('-'), never paragraphs or "
    "essays. Maximum 4-5 bullets per section (Property Overview may list its "
    "standard fields). The whole report must be skimmable in under 2 "
    "minutes.\n"
    "- Use clear labels, e.g.: Asking Price, Predicted Price, Monthly Rent, "
    "Annual Rent, Rental Yield, Risk Level, Growth Potential, Target Price, "
    "Suggested Discount, Investment Rating, Verdict.\n"
    "- PRESERVE BACKEND UNITS EXACTLY:\n"
    "  - Property prices are already in Crores: write \"₹2.15 Cr\".\n"
    "  - Monthly Rent and Annual Rent are ALREADY PRE-FORMATTED in the "
    "backend data as display strings (e.g. \"₹81,037/month\", "
    "\"₹9,72,444/year (≈9.72 Lakh/year)\"). COPY THEM CHARACTER-FOR-"
    "CHARACTER — never reformat, recompute or convert them, and NEVER "
    "express any rent in Crores.\n"
    "  - Percentages stay exactly as the backend returned them.\n"
    "  - Never convert Lakh into Crore, never convert Crore into Rupees, "
    "never guess a unit.\n"
    "- Every per-section Recommendation must restate or directly follow the "
    "backend's own verdict/flags — never a new judgement.\n"
    "- The comparison table, Best Overall Investment, Runner Up and Final "
    "Investment Decision must follow the backend comparison result (winner "
    "and overall scores) — never re-rank, never override the backend winner, "
    "never invent a better property.\n"
    "- Never invent values, builders, market trends, appreciation figures or "
    "risks. Never praise or criticise a builder/developer's reputation. "
    "Never infer anything not returned by the backend.\n"
    "- If a value or analysis is missing, write \"Not available\" instead of "
    "inventing it. 'Confidence' in the Valuation section may appear ONLY if "
    "the backend data literally contains a confidence field — otherwise omit "
    "that line entirely (do not write High/Medium/Low).\n"
    "- Risk Analysis uses the advisor's risk fields; Future Growth uses the "
    "advisor's growth fields.\n"
    "- No conversational filler, no first person, no introductions beyond "
    "the header block. Style the report like a CBRE / JLL / Knight Frank "
    "investment report, NOT a chat response, README or Markdown source.\n"
    "- Use the section icons exactly as shown in the structure (📌 🏠 📈 💰 "
    "🏷 ⚠ 🚇 🤝 ⭐ ✅ 🏆 📊 📋) and no others.\n"
    "- If any paragraph would exceed 3 lines, rewrite it as bullets.\n"
    "- ABSOLUTELY NO MARKDOWN SYNTAX in the output. Never produce the "
    "characters '#', '*', '_' (as emphasis), or '`' anywhere. No '#'-style "
    "headings, no '**bold**', no '*italic*', no inline code. Labels are "
    "plain words (write 'Property ID', never '**Property ID**').\n"
    "- Visual hierarchy comes ONLY from: the divider lines shown in the "
    "structure ('══…' for the header / property banners / final decision, "
    "'━━…' for section banners), icons, blank lines and indentation.\n"
    "- Divider lines must be copied EXACTLY as shown (38 identical "
    "characters, nothing else on the line). A property banner uses the "
    "'══…' line both ABOVE and BELOW the property name block — never mix "
    "'══…' and '━━…' within one banner.\n"
    "- Tables contain a header row and data rows ONLY. NEVER emit a "
    "Markdown separator row such as '|----|----|' — no row made of dashes "
    "anywhere.\n"
    "- Use whitespace generously — a blank line between every section — so "
    "the report reads like a finished PDF/Word consulting document."
)


def build_report_prompt(
    report: str,
    analyses: dict | None = None,
    config: PromptConfig = DEFAULT_CONFIG,
) -> Prompt:
    """
    Build a prompt that asks the LLM to present the backend report and the
    existing backend analyses as ONE professional investment report.

    Args:
        report   : The existing report content generated by the backend
                   (create_property_report). Kept as supporting context.
        analyses : Optional mapping of analysis key -> backend result for the
                   SAME selected properties, e.g. { "overview": [...],
                   "prediction": [...], "rental": [...], "valuation": [...],
                   "advisor": [...], "negotiation": [...],
                   "comparison": {...} }. Only what the backend actually
                   produced is included; missing keys are omitted.
        config   : Shared prompt configuration.

    Returns:
        Prompt: A built prompt (system + user text). No LLM call is made.
    """

    analyses = analyses or {}

    blocks = []
    for key, label in REPORT_SECTIONS:
        data = analyses.get(key)
        if not data:
            continue
        if isinstance(data, dict):
            # The comparison result is a mapping (winner + rankings). Name the
            # backend winner explicitly and unmissably — the report's Best
            # Overall Investment and Final Decision must follow it.
            winner = data.get("winner") or {}
            winner_id = winner.get("id", "unknown")
            body = (
                f"THE BACKEND COMPARISON WINNER IS: {winner_id}. The report's "
                "Executive Summary, Best Overall Investment and Final "
                "Investment Decision MUST name this property as the best "
                "investment — no other property.\n\n"
                + format_records([winner], item_label="Winner")
                + "\n\n"
                + format_records(data.get("rankings") or [], item_label="Rank")
            )
        else:
            if key == "rental":
                # Rents arrive in rupees; render them as display strings so
                # the LLM copies them verbatim (no unit conversion errors).
                data = _format_rent_fields(data)
            body = format_records(data, item_label="Property")
        blocks.append(f"{label}:\n{body}")

    # The backend narrative report is supporting context only; the structured
    # analyses above are the primary source when present.
    blocks.append(
        "Existing backend-generated report (supporting context):\n"
        "-----\n"
        f"{format_value(report)}\n"
        "-----"
    )

    backend_data = "\n\n".join(blocks)

    property_count = len(analyses.get("overview") or analyses.get("advisor") or [])
    report_date = date.today().strftime("%d %B %Y")

    # Surface the backend winner at the TOP of the instructions — burying it
    # inside the data block alone proved unreliable with small models.
    comparison = analyses.get("comparison") or {}
    winner_id = (comparison.get("winner") or {}).get("id")
    winner_line = (
        f"THE BACKEND COMPARISON WINNER IS: {winner_id}. The Executive "
        "Summary, Best Overall Investment and Final Investment Decision MUST "
        "all name this property (and no other) as the best investment, and "
        "the 'highest overall score' claim may only be made about it.\n\n"
        if winner_id
        else ""
    )

    task_instructions = (
        "Rewrite everything above as ONE professional property investment "
        "report for the SELECTED properties only (never a generic project or "
        "locality overview). Summarize ONLY the backend results shown above — "
        "comparison, price prediction, rental, valuation, risk, future "
        "growth, investment advice and negotiation. Never invent a fact, "
        "figure, unit or conclusion.\n\n"
        + winner_line
        + f"Report Date to print in the header: {report_date}\n\n"
        + (
            f"The report covers {property_count} selected "
            f"propert{'y' if property_count == 1 else 'ies'}: produce one "
            "complete property block per property"
            + (
                ", then the Comparison Summary, Investment Scorecard, Best "
                "Overall Investment and Runner Up sections"
                if property_count > 1
                else " and skip the Comparison Summary, Investment "
                "Scorecard, Best Overall Investment and Runner Up sections"
            )
            + ".\n\n"
            if property_count
            else ""
        )
        + REPORT_TEMPLATE
        + "\n"
        + REPORT_RULES
    )

    expected_output = (
        "A premium, visually clean PLAIN-TEXT investment report with ZERO "
        "Markdown syntax: the exact divider/icon structure above, short "
        "labelled lines throughout, backend units preserved, nothing "
        "invented, and no conversational tone. It should immediately "
        "resemble a finished CBRE / JLL / Knight Frank PDF page — never "
        "Markdown source, a README or a chat response."
    )

    return build_prompt(
        user_intent=(
            "The user wants a professional investment report for their "
            "selected properties."
        ),
        backend_data=backend_data,
        task_instructions=task_instructions,
        expected_output=expected_output,
        config=config,
    )
