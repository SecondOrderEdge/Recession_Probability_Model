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
    prompt = f"""You are a senior macro strategist writing a daily recession probability
briefing for an investment committee. Based on the model output below, write a
concise, professional email briefing.

MODEL OUTPUT:
{json.dumps(summary, indent=2)}

Write the email with these sections:
1. **Subject line** — one line, include the probability and signal level
2. **Executive Summary** — 2-3 sentences with the headline probability, CI, and consensus
3. **Key Indicators** — brief table or bullet list of the BIC-selected features with current values and what they mean economically
4. **What Changed** — note any indicators at extreme percentiles (below 10th or above 90th)
5. **Sensitivity** — which 1-2 indicators would most change the outlook if they moved
6. **Adverse Scenario** — what happens if all indicators deteriorate by 1 SD
7. **Bottom Line** — one sentence actionable takeaway

Format the email body as clean HTML suitable for email clients. Use inline CSS only.
Keep the tone professional but direct — this is for a CIO, not a blog post.
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
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": content}],
    )

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
