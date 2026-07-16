# Manual verification for Phase 17.4 — renders a sample comparison report
# (new template shape) through the REAL pdf_service pipeline (headless
# Chromium + /print/report), and a standard-report sample to confirm the
# Reports page path is untouched. Run with the frontend on :3000.

import sys

sys.path.insert(0, ".")

from src.services.pdf_service import generate_report_pdf

H = "═" * 38
L = "━" * 38

COMPARISON_REPORT = f"""{H}
        EstateMind Property Comparison Report

Generated for
Royal Palms vs Shree Heights

Report Date
16 July 2026
{H}

{H}
🏆 Best Overall Investment
{H}
Property
Royal Palms

Overall Score: 0.775
Verdict: Best Value
Category Wins: 5 of 6 categories

Reason
✓ Undervalued by ₹91 Lakhs vs. predicted price
✓ Highest rental yield of 6.67%
✓ Low risk with 0 concerns flagged
✓ Highest overall score of 0.775

{L}
📊 Overall Comparison Score
{L}
Royal Palms: 5 Wins
Categories Won: Price Winner, Value Winner, Rental Winner, Risk Winner, Investment Winner
Shree Heights: 1 Win
Categories Won: Negotiation Winner

{L}
🏠 Basic Information
{L}
| Metric | Royal Palms | Shree Heights |
| Location | Goregaon | Goregaon |
| Property Type | Resale | Resale |
| Configuration | 2 BHK | 2 BHK |
| Area | 880 sqft | 880 sqft |
| Asking Price | 🏆 ₹1.10 Cr | ₹2.10 Cr |

{L}
📈 Price Prediction
{L}
| Metric | Royal Palms | Shree Heights |
| Asking Price | ₹1.10 Cr | ₹2.10 Cr |
| Predicted Price | ₹2.01 Cr | ₹2.25 Cr |
| Difference | 🏆 ₹0.91 Cr | ₹0.15 Cr |
| Status | 🟢 Undervalued | 🟡 Fair |
Winner: Royal Palms — ₹0.91 Cr below predicted value

{L}
💰 Rental Analysis
{L}
| Metric | Royal Palms | Shree Heights |
| Monthly Rent | ₹61,160/month | ₹61,160/month |
| Annual Rent | ₹7,33,920/year (≈7.34 Lakh/year) | ₹7,33,920/year (≈7.34 Lakh/year) |
| Rental Yield | 🏆 6.67% | 3.49% |
| Demand | High | Medium |
| Rental Rating | ★★★★★ | ★★★★☆ |
Winner: Royal Palms — 6.67% rental yield

{L}
⚠ Risk Analysis
{L}
| Metric | Royal Palms | Shree Heights |
| Risk Level | 🏆 Low | High |
| Risk Score | 0 | 14 |
| Concerns Flagged | 0 | 7 |
| Key Concerns | None | Overpriced, high floor, low yield |
Winner: Royal Palms — Low risk

{L}
🚇 Future Growth
{L}
| Metric | Royal Palms | Shree Heights |
| Growth Potential | Moderate | Moderate |
| Key Drivers | Metro expansion | Commercial hub |
| Infrastructure | Good | Average |

{L}
🏷 Property Valuation
{L}
| Metric | Royal Palms | Shree Heights |
| Valuation Status | 🏆 🟢 Undervalued | 🟡 Fair |
| Price per sqft | ₹12,500/sqft | ₹23,863/sqft |
| Assessment | Priced below market benchmark | In line with the market |
Winner: Royal Palms — Undervalued

{L}
⭐ Investment Advisor
{L}
| Metric | Royal Palms | Shree Heights |
| Overall Score | 🏆 0.775 | 0.300 |
| Investment Rating | ★★★★★ | ★★★☆☆ |
| Verdict | Best Value | Risky |
| Suitable For | Long-term investors | Not available |
Winner: Royal Palms — Overall score 0.775

{L}
🤝 Negotiation
{L}
| Metric | Royal Palms | Shree Heights |
| Target Price | ₹1.06 Cr | ₹1.89 Cr |
| Suggested Discount | 2-5% | 🏆 8-12% |
| Negotiation Power | Low | High |
Winner: Shree Heights — 8-12% suggested discount

{L}
📋 Final Scoreboard
{L}
Price Winner: Royal Palms — ₹1.10 Cr lowest asking price
Value Winner: Royal Palms — ₹0.91 Cr below predicted value
Rental Winner: Royal Palms — 6.67% rental yield
Risk Winner: Royal Palms — Low risk
Negotiation Winner: Shree Heights — 8-12% suggested discount
Investment Winner: Royal Palms — Overall score 0.775

{H}
🏆 Final Recommendation
{H}
Recommended Property
Royal Palms

Overall Score: 0.775
Verdict: Best Value
Category Wins: 5 of 6 categories

Why Choose It
✓ Lowest risk of the compared properties
✓ Highest rental yield of 6.67%
✓ Undervalued by ₹0.91 Cr
✓ Best overall score of 0.775

Runner Up: Shree Heights — Overpriced with a higher risk score

Recommended Action: Proceed with purchase and negotiate toward the target price of ₹1.06 Cr.

{L}
📋 Suggested Next Steps
{L}
□ Negotiate toward the target price of ₹1.06 Cr
□ Visit Royal Palms in person
□ Verify legal documents and title
□ Confirm rental demand with local brokers
"""

pdf = generate_report_pdf(COMPARISON_REPORT)
with open("comparison_report_test.pdf", "wb") as f:
    f.write(pdf)
print(f"comparison PDF: {len(pdf)} bytes -> comparison_report_test.pdf")
