"""
Claude API Email Generator for Recession Probability Report

Reads the daily_summary.json from the daily report pipeline, sends it
to Claude for analysis, and generates an HTML email with embedded chart
references. Optionally sends via SendGrid.

Usage:
    python automation/generate_email.py

Required environment variables:
    ANTHROPIC_API_KEY - Claude API key
    MAIL_USERNAME     - Gmail address (optional, for sending)
    MAIL_PASSWORD     - Gmail app password (optional, for sending)
    MAIL_PORT         - SMTP port, typically 587 (optional)
    EMAIL_TO          - Recipient email address(es), comma-separated
"""

import os
import sys
import json
import base64
from datetime import datetime
from pathlib import Path

import anthropic

OUTPUT_DIR = Path(__file__).parent / "output"


def load_summary():
    """Load the daily summary JSON."""
    path = OUTPUT_DIR / "daily_summary.json"
    if not path.exists():
        print(f"ERROR: {path} not found. Run daily_report.py first.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def encode_image(path):
    """Base64-encode an image for Claude's vision API."""
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def generate_analysis(summary):
    """Send summary + charts to Claude for analysis and email generation."""
    client = anthropic.Anthropic()

    # Build the prompt
    run_date = summary.get("run_date", datetime.now().strftime("%Y-%m-%d"))
    data_date = summary.get("data_through", "unknown")

    prompt = f"""You are a senior macro strategist writing a weekly recession probability
briefing for an investment committee.

TODAY'S DATE: {run_date}
DATA THROUGH: {data_date}

IMPORTANT FORMATTING: Include a visible "Data Through: {data_date}" line at the very top
of the email body, immediately below the report title and date. This tells the reader how
current the underlying data is. Format it prominently — not buried in fine print.

IMPORTANT: All references to time must be grounded in today's date ({run_date}).
Do not reference future months that haven't happened yet. When discussing what to
watch, reference the NEXT data releases relative to {run_date} (e.g. if today is
April 2026, the next jobs report is in May 2026, not November).

Based on the model output and attached charts below,
write a substantive, visually-integrated email briefing.

You have 7 charts available. Reference them in your HTML using <img src="cid:chart_0">
through <img src="cid:chart_6"> tags. The chart order is:
- cid:chart_0 = Probability gauge (current reading)
- cid:chart_1 = Probability trend (24-month trailing, is risk rising or falling?)
- cid:chart_2 = Indicator percentile dashboard (where each indicator sits historically)
- cid:chart_3 = Model comparison (do all specifications agree?)
- cid:chart_4 = Indicator sparklines (24-month trailing trends for each feature)
- cid:chart_5 = Historical probability (full history with NBER recession shading)
- cid:chart_6 = Sensitivity analysis (which indicators move the needle most)

IMPORTANT: Embed these charts INLINE within the relevant sections of your email using
img tags with the cid: references above. Do NOT group all charts at the top or bottom.
Place each chart immediately after the section it illustrates. Add a brief caption
below each chart in small gray text.

MODEL OUTPUT:
{json.dumps(summary, indent=2)}

Write the email with these sections:
1. **Subject line** — one line. Format: "Recession Probability: [ensemble]% — Models range
   [low]% to [high]%". Do NOT include traffic-light labels (LOW/MODERATE/HIGH) in the subject
   or anywhere in the narrative text. The color-banded gauge chart handles visual classification.

2. **Investment Committee Summary** — This is the FIRST content section, immediately after the
   Data Through header. It must fit on one page and contain exactly these blocks in order:

   SNAPSHOT TABLE (4 rows, formatted as an HTML table):
   | Current Probability | [ensemble]% |
   | Direction of Change | [rising/falling/stable based on trend data] |
   | Primary Risk Driver | [the indicator closest to its warning trigger] |
   | Primary Offset      | [the indicator most strongly supporting expansion] |

   MODEL RANGE (one sentence): "Individual models range from [low]% to [high]%, reflecting
   meaningful disagreement on yield curve interpretation."

   PORTFOLIO POSITIONING (bullet format, exactly 4 bullets):
   - Equities: [stance] — [one-sentence rationale using "consistent with" framing]
   - Fixed Income: [stance] — [one-sentence rationale]
   - Credit: [stance] — [one-sentence rationale]
   - Hedging: [stance] — [one-sentence rationale]
   Frame ALL positioning as "consistent with [condition]" — never as a directive.
   Add a footer line in italics: "Positioning reflects model output only and should be
   evaluated against individual mandate constraints."

   Place chart_0 (probability gauge) after this section.

3. **Executive Summary** (3-5 sentences) — The headline figure is the ENSEMBLE probability.
   Lead with: "Our five-model ensemble estimates [X]% probability of recession in the next
   12 months — consistent with expansion-phase conditions, though model dispersion warrants
   attention." Do NOT use "LOW risk," "MODERATE risk," or "HIGH risk" as labels. Instead
   describe what the probability level is consistent with. Note consensus strength. The
   BIC-selected model is one input among five. If models diverge, state the divergence
   factually and defer the explanation to the Model Divergence section.

   Place chart_1 (probability trend) and chart_3 (model comparison) after this section.

4. **Key Indicators** — Consolidate into exactly four macro buckets. Each gets exactly two
   sentences: what the indicators currently show, and what that implies for recession risk.
   Do NOT define what indicators measure. Do NOT include historical context or background.

   **Growth** (housing starts, new home sales): [current readings] [recession risk implication]
   **Inflation** (core CPI, PPI): [current readings] [recession risk implication]
   **Policy** (fed funds rate, yield curve spread): [current readings] [recession risk implication]
   **Market Signals** (credit spreads, consumer sentiment): [current readings] [recession risk implication]

   Place chart_2 (indicator percentiles) and chart_4 (sparklines) after this section.

5. **Model Divergence Analysis** — Structure as exactly three paragraphs:

   Paragraph 1 — Why the yield curve signal still matters: The term spread has preceded every
   recession since 1968. The current un-inversion phase is historically the most dangerous
   period — recessions typically begin 6-18 months after the curve steepens from inversion.

   Paragraph 2 — Why this cycle may differ (three specific structural factors only):
   (a) QE suppressed term premium artificially, making inversion easier to achieve without
   credit tightening; (b) foreign central bank demand for Treasuries compressed long-end yields
   independent of growth expectations; (c) post-Basel III bank regulation reduced duration risk
   appetite, flattening the curve structurally.

   Paragraph 3 — Our judgment: one clear sentence stating which interpretation the ensemble
   weighting implies, followed by one sentence on what evidence would confirm or deny it.
   Write with institutional conviction while acknowledging the judgment call.

   Include this limitation note verbatim at the end of the section: "Note: the model assigns
   a negative coefficient to inflation, reflecting the historical pattern where demand-collapse
   recessions are preceded by disinflation — this may understate stagflation risk in the
   current tariff environment where inflation and growth weakness could occur simultaneously."

   Place chart_5 (historical probability) after this section.

6. **Watchlist — Trigger Levels** — Present the sensitivity data as a concrete watchlist table.
   Rank by which triggers are CLOSEST to being hit (smallest distance from current).
   For each: the indicator, current value, trigger value, distance, and what real-world event
   could cause the move. Two sentences maximum per indicator.

   Place chart_6 (sensitivity/watchlist) after this section.

7. **Adverse Scenario** — Structure as follows. Do NOT assign a numerical probability to this
   scenario — the model does not estimate conditional joint probabilities and fabricating one
   would be misleading.

   SCENARIO CLASSIFICATION: "Tail risk. Requires simultaneous deterioration across uncorrelated
   indicators — historically rare outside of systemic financial crises or external shock events."

   REAL-WORLD TRIGGERS (three bullets):
   (a) Major Fed policy error — overtightening into slowing growth, forcing rapid pivot that
       destabilizes credit markets.
   (b) Energy price shock — sustained oil above $130/barrel reigniting PPI acceleration while
       suppressing consumer demand.
   (c) Credit event — regional bank stress or sovereign contagion forcing broad credit spread
       widening and loan contraction.

   HEDGING IMPLICATION (one paragraph): Apply this framework to the CURRENT ensemble reading:
   - Below 20% ensemble with stable trend = monitoring posture only, standard rebalancing
   - 20-35% with rising trend = consider tail hedges (long vol, Treasury duration extension)
   - Above 35% = defensive repositioning warranted
   State the current reading, which bracket it falls in, and the explicit conclusion.

   WHAT TO WATCH: Two sentences on which of the three triggers above is most proximate
   given current data.

8. **What Would Change Our View** — Numbered list of exactly five items. Each includes the
   indicator, the specific threshold, and the economic implication. Use these EXACT items:

   1. Term spread falls below -1.0% — historically associated with hard landing risk; would
      trigger material upward revision to ensemble.
   2. Core CPI drops below 1.5% — signals demand destruction outpacing supply normalization;
      deflation risk inconsistent with soft landing.
   3. Housing starts decline exceeds 15% year-over-year — indicates mortgage rate transmission
      accelerating beyond stabilization phase.
   4. Initial jobless claims sustain above 300K on four-week average — early deterioration in
      labor demand before unemployment rate responds.
   5. Ensemble probability rises above 20% for two consecutive monthly updates — model
      convergence signal that would warrant defensive repositioning review.

   These thresholds are hardcoded. Do not compute them dynamically or modify the wording.

9. **Data Currency Notice** — Format as a gray bordered box. Use the "data_through" and
   "lagged_series" fields from the JSON:
   "Data Currency Notice: This briefing reflects FRED data available through [data_through].
   The following series have publication lags exceeding 30 days and will update on their next
   FRED vintage release: [comma-separated lagged_series list, or 'None' if empty]. Conditions
   may have changed materially since the data cutoff. Model probabilities will refresh
   automatically on next scheduled run."

10. **Bottom Line for the Committee** — 2-3 sentences. Lead with the ensemble figure:
    "Our five-model ensemble at [X]% — consistent with [expansion/contraction/transition]
    conditions." Reference the nearest trigger from the watchlist. State what would change
    the recommendation. Do NOT use traffic-light labels.

LANGUAGE RULES (apply globally across all sections):
- Remove any sentence that restates what a chart already shows.
- Remove any sentence that defines what an indicator measures (the committee knows).
- Replace "suggests" with "indicates" or "shows" where the evidence is clear.
- Replace "may" with "could" where appropriate.
- Eliminate all uses of "it's worth noting," "notably," "importantly," and "it is important to."
- Do not hedge every sentence. Write with institutional conviction.
- Never describe probability as "LOW risk" or "HIGH risk" — describe what it is consistent with.

Format the email body as clean HTML suitable for email clients. Use inline CSS only.
Target length: 1000-1500 words. This is an institutional investment memo, not a blog summary.
The tone should be that of a senior economist briefing the CIO — authoritative, specific,
and willing to take a view.
Do not use emojis. Use percentage signs and basis points where appropriate.

Return your response as JSON with two keys:
- "subject": the email subject line
- "html_body": the full HTML email body
"""

    # Load chart images for Claude to reference
    content = [{"type": "text", "text": prompt}]

    # Add charts if they exist
    for chart_name in summary.get("charts", []):
        chart_path = OUTPUT_DIR / chart_name
        if chart_path.exists():
            img_data = encode_image(chart_path)
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": img_data,
                },
            })
            content.append({
                "type": "text",
                "text": f"[Above image: {chart_name}]",
            })

    print("Sending to Claude API for analysis...")
    response = None
    for attempt in range(4):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=8192,
                messages=[{"role": "user", "content": content}],
            )
            break
        except Exception as e:
            if attempt < 3:
                import time
                wait = 2 ** (attempt + 1)
                print(f"  API error (attempt {attempt+1}/4): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  API failed after 4 attempts: {e}")
                raise

    # Parse response
    response_text = response.content[0].text

    # Try to extract JSON from the response
    try:
        # Handle case where Claude wraps JSON in markdown code blocks
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            json_str = response_text
        result = json.loads(json_str)
    except json.JSONDecodeError:
        # Fallback: use the raw text as the body
        result = {
            "subject": f"Recession Probability Report — {summary['bic_probability']}% ({summary['signal']})",
            "html_body": f"<html><body><pre>{response_text}</pre></body></html>",
        }

    return result


def embed_charts_in_html(html_body, summary):
    """Replace chart references with CID-based inline images for email."""
    for i, chart_name in enumerate(summary.get("charts", [])):
        cid = f"chart_{i}"
        # Add image tags where appropriate (at the end if not already present)
        if chart_name not in html_body:
            section_map = {
                "recession_probability_gauge.png": "Executive Summary",
                "recession_probability_history.png": "historical",
                "sensitivity_chart.png": "Sensitivity",
            }
            for keyword in section_map.values():
                if keyword.lower() in html_body.lower():
                    # Insert image after the relevant section
                    idx = html_body.lower().find(keyword.lower())
                    # Find the next closing tag after keyword
                    close_idx = html_body.find("</", idx + len(keyword))
                    if close_idx > 0:
                        tag_end = html_body.find(">", close_idx) + 1
                        img_tag = f'<br><img src="cid:{cid}" style="max-width:100%;height:auto;"><br>'
                        html_body = html_body[:tag_end] + img_tag + html_body[tag_end:]
                    break
    return html_body


def send_email(subject, html_body, summary):
    """Send email via Gmail SMTP with embedded chart images."""
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.image import MIMEImage

    mail_username = os.environ.get("MAIL_USERNAME")
    mail_password = os.environ.get("MAIL_PASSWORD")
    mail_port = int(os.environ.get("MAIL_PORT", "587"))
    email_to = os.environ.get("EMAIL_TO", "")

    if not mail_username or not mail_password:
        print("MAIL_USERNAME/MAIL_PASSWORD not set. Saving email to file instead.")
        save_email_to_file(subject, html_body)
        return

    if not email_to:
        print("EMAIL_TO not set. Saving email to file instead.")
        save_email_to_file(subject, html_body)
        return

    recipients = [addr.strip() for addr in email_to.split(",") if addr.strip()]

    # Build MIME message
    msg = MIMEMultipart("related")
    msg["Subject"] = subject
    msg["From"] = mail_username
    msg["To"] = ", ".join(recipients)

    # Attach HTML body
    msg.attach(MIMEText(html_body, "html"))

    # Attach charts as inline images
    for i, chart_name in enumerate(summary.get("charts", [])):
        chart_path = OUTPUT_DIR / chart_name
        if chart_path.exists():
            with open(chart_path, "rb") as f:
                img = MIMEImage(f.read(), _subtype="png")
            img.add_header("Content-ID", f"<chart_{i}>")
            img.add_header("Content-Disposition", "inline", filename=chart_name)
            msg.attach(img)

    # Send via SMTP
    try:
        with smtplib.SMTP("smtp.gmail.com", mail_port) as server:
            server.starttls()
            server.login(mail_username, mail_password)
            server.sendmail(mail_username, recipients, msg.as_string())
        print(f"Email sent to {recipients}")
    except Exception as e:
        print(f"Email send failed: {e}")
        save_email_to_file(subject, html_body)


def save_email_to_file(subject, html_body):
    """Save email as HTML file when SendGrid is not available."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    path = OUTPUT_DIR / f"email_report_{date_str}.html"

    full_html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>{subject}</title></head>
<body>
<p style="color:#999;font-size:12px;">Subject: {subject}</p>
<hr>
{html_body}
</body>
</html>"""

    with open(path, "w") as f:
        f.write(full_html)
    print(f"Email saved to {path}")


def run():
    """Main pipeline."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    print("=== Generating Email Report via Claude API ===\n")

    summary = load_summary()
    print(f"Loaded summary: {summary['run_date']}, prob={summary['bic_probability']}%")

    result = generate_analysis(summary)
    subject = result["subject"]
    html_body = result["html_body"]

    # Embed chart references
    html_body = embed_charts_in_html(html_body, summary)

    print(f"\nSubject: {subject}")
    print(f"Body length: {len(html_body)} chars")

    # Send or save
    send_email(subject, html_body, summary)

    # Also save the raw output
    with open(OUTPUT_DIR / "email_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    run()
