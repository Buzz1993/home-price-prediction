# ===============================
# src/llm/prompts/comparison_report_prompt.py
# ===============================
#
# Prompt builder for the COMPARISON REPORT (Phase 17.3, redesigned in 17.4).
#
# The Property Comparison page (/compare) shares a report that mirrors the
# comparison view itself — Best Overall Investment hero, Overall Comparison
# Score (win tally), one side-by-side table per analysis with 🏆 winner
# cells, Final Scoreboard and Final Recommendation — instead of the standard
# per-property investment report (report_prompt.py), which stays untouched
# for the Reports page.
#
# Input: the structured results of the EXISTING backend analyses for the
# compared properties (property overview, price prediction, rental,
# valuation, advisor — which carries the risk and future-growth fields —
# negotiation and the comparison winner/rankings). This builder only asks the
# LLM to re-present that data in the comparison layout.
#
# It performs no comparison, analysis or reasoning of its own:
#   - Every figure, verdict, score, winner and flag comes from the backend.
#   - Missing analyses are simply omitted; nothing is ever invented.
#   - The layout conventions (dividers, icons, plain text — no Markdown) are
#     identical to report_prompt.py, so the existing report preview, print
#     route and single Chromium PDF pipeline render it unchanged.

from datetime import date

from src.llm.prompts.config import PromptConfig, DEFAULT_CONFIG
from src.llm.prompts.templates import Prompt, build_prompt
from src.llm.prompts.formatting import format_records, format_value

# Reuse the exact divider style and rupee-rent pre-formatting of the standard
# report so both reports share one visual language and unit discipline.
from src.llm.prompts.report_prompt import (
    DIVIDER_HEAVY,
    DIVIDER_LIGHT,
    REPORT_RULES,
    _format_rent_fields,
)


# The analysis blocks handed to the LLM, in presentation order. Comparison
# comes FIRST — it is the spine of this report. "advisor" also carries the
# Risk fields (risk_label, risk_score) and Future Growth fields (growth_label,
# growth_reason) produced during enrichment.
COMPARISON_REPORT_SECTIONS = (
    ("comparison", "Property Comparison results (winner and rankings)"),
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
)


# The exact document layout — the /compare page itself rendered as PLAIN TEXT
# (Phase 17.4). Every analysis is a side-by-side '|' table with one column per
# property (never stacked blocks, never paragraphs), the winning cell of each
# row carries a '🏆 ' prefix, and the page's own section order is kept:
# Executive Winner → Overall Comparison Score → comparison tables → Final
# Scoreboard → Final Recommendation. Same conventions as REPORT_TEMPLATE
# (dividers, icons, 'Label: Value' lines, no table separator rows, zero
# Markdown).
COMPARISON_REPORT_TEMPLATE = (
    "Report structure (follow EXACTLY, in this order, as PLAIN TEXT — no "
    "Markdown anywhere). The document is the Property Comparison page "
    "printed: comparison-first, side-by-side, almost no prose.\n"
    "\n"
    f"{DIVIDER_HEAVY}\n"
    "        EstateMind Property Comparison Report\n"
    "\n"
    "Generated for\n"
    "<the compared properties (project names), joined with ' vs '>\n"
    "\n"
    "Report Date\n"
    "<the date provided below>\n"
    f"{DIVIDER_HEAVY}\n"
    "\n"
    f"{DIVIDER_HEAVY}\n"
    "🏆 Best Overall Investment\n"
    f"{DIVIDER_HEAVY}\n"
    "Property\n"
    "<the backend comparison winner (project name)>\n"
    "\n"
    "Overall Score: <the winner's backend overall score>\n"
    "Verdict: <the winner's backend verdict>\n"
    "Category Wins: <X> of <Y> categories\n"
    "(X = how many Final Scoreboard categories below the winner wins, Y = "
    "the number of categories listed there — count them consistently)\n"
    "\n"
    "Reason\n"
    "3-5 '✓ …' bullets using only backend results (score, valuation, rental, "
    "growth, risk — include only those where the winner is actually "
    "better).\n"
    "\n"
    f"{DIVIDER_LIGHT}\n"
    "📊 Overall Comparison Score\n"
    f"{DIVIDER_LIGHT}\n"
    "Two 'Label: Value' lines per compared property, winner first, then the "
    "others in backend ranking order:\n"
    "<Project Name>: <n> Wins\n"
    "Categories Won: <the Final Scoreboard categories it wins, comma-"
    "separated, or 'None'>\n"
    "The win counts MUST match the Final Scoreboard section exactly.\n"
    "\n"
    f"{DIVIDER_LIGHT}\n"
    "🏠 Basic Information\n"
    f"{DIVIDER_LIGHT}\n"
    "ONE '|' table — header row 'Metric | <Property A> | <Property B> …' "
    "(one column per property, ALWAYS in the same property order in every "
    "table of this report), then rows: Location, Property Type, "
    "Configuration, Area, Asking Price, plus any other overview fields the "
    "backend provides for all properties (Builder, Floor, Age…). Prefix the "
    "winning cell of the Asking Price row (lowest price) with '🏆 '.\n"
    "\n"
    f"{DIVIDER_LIGHT}\n"
    "📈 Price Prediction\n"
    f"{DIVIDER_LIGHT}\n"
    "ONE '|' table, same header — rows: Asking Price, Predicted Price, "
    "Difference (predicted minus asking), Status (the backend's own flag "
    "with 🟢/🟡/🔴 matching it). Prefix the best-value cells (largest "
    "positive difference) with '🏆 '.\n"
    "Winner: <Project Name> — <the backend figure that wins it>\n"
    "\n"
    f"{DIVIDER_LIGHT}\n"
    "💰 Rental Analysis\n"
    f"{DIVIDER_LIGHT}\n"
    "ONE '|' table — rows: Monthly Rent, Annual Rent, Rental Yield, Demand, "
    "Rental Rating (★ stars exactly as the backend rates it, if it does). "
    "Prefix the highest Rental Yield cell with '🏆 '.\n"
    "Winner: <Project Name> — <highest backend rental yield>\n"
    "\n"
    f"{DIVIDER_LIGHT}\n"
    "⚠ Risk Analysis\n"
    f"{DIVIDER_LIGHT}\n"
    "ONE '|' table — rows: Risk Level, Risk Score (if available), Concerns "
    "Flagged (the COUNT of backend-listed concerns), Key Concerns (a very "
    "short comma-separated list; 'None' when the backend lists none). "
    "Prefix the best (lowest) Risk Level cell with '🏆 '.\n"
    "Winner: <Project Name> — <the backend risk level>\n"
    "\n"
    f"{DIVIDER_LIGHT}\n"
    "🚇 Future Growth\n"
    f"{DIVIDER_LIGHT}\n"
    "ONE '|' table — rows: Growth Potential, Growth Score (if available), "
    "Key Drivers (only those the backend names), Infrastructure (only if "
    "the backend provides it). Prefix the best Growth Potential cell with "
    "'🏆 ' when the backend values differ.\n"
    "Winner: <Project Name> — <the backend growth outlook> (omit this line "
    "when the backend growth values are equal)\n"
    "\n"
    f"{DIVIDER_LIGHT}\n"
    "🏷 Property Valuation\n"
    f"{DIVIDER_LIGHT}\n"
    "ONE '|' table — rows: Valuation Status (backend flag), Price per sqft "
    "(if available), Assessment (one short backend phrase). Prefix the best "
    "valuation cell (undervalued beats fair beats overpriced) with '🏆 '.\n"
    "Winner: <Project Name> — <the backend valuation flag>\n"
    "\n"
    f"{DIVIDER_LIGHT}\n"
    "⭐ Investment Advisor\n"
    f"{DIVIDER_LIGHT}\n"
    "ONE '|' table — rows: Overall Score, Investment Rating (★ stars from "
    "the backend verdict), Verdict, Suitable For. Prefix the highest "
    "Overall Score cell with '🏆 '.\n"
    "Winner: <Project Name> — Overall score <the backend comparison "
    "winner's score> (this is ALWAYS the backend comparison winner)\n"
    "\n"
    f"{DIVIDER_LIGHT}\n"
    "🤝 Negotiation\n"
    f"{DIVIDER_LIGHT}\n"
    "ONE '|' table — rows: Target Price, Suggested Discount, Negotiation "
    "Power. Prefix the largest Suggested Discount cell with '🏆 '.\n"
    "Winner: <Project Name> — <the backend discount / power that wins it>\n"
    "\n"
    f"{DIVIDER_LIGHT}\n"
    "📋 Final Scoreboard\n"
    f"{DIVIDER_LIGHT}\n"
    "'Label: Value' lines naming which property wins each category, ONLY "
    "where the backend data actually distinguishes them (omit a category "
    "when the backend values are equal or missing): Price Winner (lowest "
    "asking price); Value Winner (best asking vs. predicted price gap); "
    "Rental Winner (highest rental yield); Risk Winner (best risk level); "
    "Growth Winner (best growth potential); Valuation Winner (best backend "
    "valuation flag); Negotiation Winner (largest suggested discount); "
    "Investment Winner (the backend comparison winner, always shown). Each "
    "line: 'Category: <Project Name> — <the backend figure that wins it>'. "
    "These categories are the source of the win counts and '🏆 ' cells "
    "above — keep them consistent.\n"
    "\n"
    f"{DIVIDER_HEAVY}\n"
    "🏆 Final Recommendation\n"
    f"{DIVIDER_HEAVY}\n"
    "Recommended Property\n"
    "<backend winner>\n"
    "\n"
    "Overall Score: <the winner's backend overall score>\n"
    "Verdict: <the winner's backend verdict>\n"
    "Category Wins: <X> of <Y> categories\n"
    "\n"
    "Why Choose It\n"
    "3-4 '✓ …' bullets (backend values only).\n"
    "\n"
    "Runner Up: <the second-ranked property> — <ONE short backend-grounded "
    "phrase on why it ranked second>\n"
    "\n"
    "Recommended Action: ONE concise sentence from the winner's backend "
    "negotiation strategy (e.g. proceed and negotiate toward the backend "
    "target price of ₹X.XX Cr).\n"
    "\n"
    f"{DIVIDER_LIGHT}\n"
    "📋 Suggested Next Steps\n"
    f"{DIVIDER_LIGHT}\n"
    "Maximum 5 checkbox bullets ('□ …'), practical actions grounded in the "
    "backend results (e.g. '□ Negotiate toward the target price of ₹X.XX "
    "Cr', '□ Visit the winning property', '□ Verify legal documents')."
)

# Comparison-specific layout rules appended after the shared REPORT_RULES.
COMPARISON_REPORT_EXTRA_RULES = (
    "\nComparison layout rules (Phase 17.4 — these override any conflicting "
    "style above):\n"
    "- NO paragraphs and NO per-property prose blocks anywhere. Every "
    "analysis is ONE side-by-side '|' table with one column per property — "
    "never one property stacked below another.\n"
    "- Every table uses the SAME header ('Metric | <Property A> | "
    "<Property B> …') with the properties in the SAME order everywhere. "
    "Every table line — header and data rows — STARTS with '|' and ENDS "
    "with '|' (e.g. '| Metric | Royal Palms | Shree Heights |'), with no "
    "separator row of dashes anywhere.\n"
    "- Winner cells: prefix the winning VALUE with '🏆 ' (trophy + one "
    "space) INSIDE its table cell — at most one '🏆 ' per row, and only "
    "when the backend values actually differ. Never add the trophy to the "
    "Metric column or to equal values.\n"
    "- Each analysis table is followed by its single 'Winner: <Project "
    "Name> — <backend figure>' line as specified.\n"
    "- A row whose value is missing for a property shows 'Not available' "
    "in that cell; never drop a property column.\n"
    "- The '🏆 ' cell prefixes, the per-table 'Winner:' lines, the Overall "
    "Comparison Score win counts and the Final Scoreboard MUST all tell the "
    "same story — one consistent set of category winners."
)


def build_comparison_report_prompt(
    analyses: dict,
    report: str | None = None,
    config: PromptConfig = DEFAULT_CONFIG,
) -> Prompt:
    """
    Build a prompt that asks the LLM to present the existing backend
    comparison and analyses as ONE comparison report matching the /compare
    page layout.

    Args:
        analyses : Mapping of analysis key -> backend result for the compared
                   properties, e.g. { "comparison": {...}, "overview": [...],
                   "prediction": [...], "rental": [...], "valuation": [...],
                   "advisor": [...], "negotiation": [...] }. Only what the
                   backend actually produced is included; missing keys are
                   omitted.
        report   : Optional backend-generated narrative report
                   (create_property_report), kept as supporting context only.
        config   : Shared prompt configuration.

    Returns:
        Prompt: A built prompt (system + user text). No LLM call is made.
    """

    analyses = analyses or {}

    blocks = []
    for key, label in COMPARISON_REPORT_SECTIONS:
        data = analyses.get(key)
        if not data:
            continue
        if isinstance(data, dict):
            # The comparison result is a mapping (winner + rankings). Name the
            # backend winner explicitly and unmissably — every winner-related
            # section of this report must follow it.
            winner = data.get("winner") or {}
            winner_id = winner.get("id", "unknown")
            body = (
                f"THE BACKEND COMPARISON WINNER IS: {winner_id}. The report's "
                "Best Overall Investment, Final Scoreboard (Investment "
                "Winner) and Final Recommendation MUST name this property "
                "as the best investment — no other property.\n\n"
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

    if report and report.strip():
        # Supporting context only; the structured analyses are the source.
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
        f"THE BACKEND COMPARISON WINNER IS: {winner_id}. The Best Overall "
        "Investment, Overall Comparison Score (first entry), Final "
        "Scoreboard (Investment Winner) and Final Recommendation MUST all "
        "name this property (and no other) as the best investment, and the "
        "'highest overall score' claim may only be made about it.\n\n"
        if winner_id
        else ""
    )

    task_instructions = (
        "Rewrite everything above as ONE property COMPARISON report for the "
        "compared properties only (never a generic project or locality "
        "overview). The report is the Property Comparison page itself, "
        "printed: winner hero first, then the win tally, then side-by-side "
        "'|' comparison tables for EVERY analysis, then the Final "
        "Scoreboard and Final Recommendation. Almost no prose — a reader "
        "scans it, they never read paragraphs. Summarize ONLY the backend "
        "results shown above — comparison, price prediction, rental, "
        "valuation, risk, future growth, investment advice and negotiation. "
        "Never invent a fact, figure, unit or conclusion, and never re-rank "
        "the properties.\n\n"
        + winner_line
        + f"Report Date to print in the header: {report_date}\n\n"
        + (
            f"The report compares {property_count} properties: every table "
            "must have one 'Metric' column plus one column per property — "
            "never skip a property, never stack properties vertically.\n\n"
            if property_count
            else ""
        )
        + COMPARISON_REPORT_TEMPLATE
        + "\n"
        + REPORT_RULES
        + COMPARISON_REPORT_EXTRA_RULES
    )

    expected_output = (
        "A premium, visually clean PLAIN-TEXT property comparison report "
        "with ZERO Markdown syntax: the exact divider/icon structure above, "
        "every analysis as ONE side-by-side '|' table with one column per "
        "property, '🏆 ' prefixes on the winning cells, short labelled "
        "lines everywhere else, backend units preserved, nothing invented, "
        "and no conversational tone. It should read as the /compare "
        "dashboard printed into a consulting dossier — never Markdown "
        "source, a README or a chat response."
    )

    return build_prompt(
        user_intent=(
            "The user compared properties side-by-side on the Property "
            "Comparison page and wants that comparison as a report."
        ),
        backend_data=backend_data,
        task_instructions=task_instructions,
        expected_output=expected_output,
        config=config,
    )
