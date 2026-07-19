# ===============================
# src/services/email_service.py
# ===============================
#
# Hybrid email service for password reset emails (Phase 27/28, hybrid
# transport in Phase 28.2). Reads configuration from environment variables
# only. The public API is a single function, send_password_reset_email();
# callers never know which provider delivered the email.
#
# Provider selection (configuration only, no code changes):
# - RESEND_API_KEY set   -> ResendProvider (HTTPS API; works on Railway,
#                           which blocks outbound SMTP ports 25/465/587).
# - SMTP_* set           -> GmailSmtpProvider (Gmail 587 STARTTLS or 465 SSL,
#                           or any standard SMTP server). Local development.
# - Neither configured   -> log a WARNING with the reset URL (local
#                           development convenience) and return False.
#                           Never fake success.

import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("estatemind.email")

RESET_TOKEN_TTL_MINUTES = 30

EMAIL_SUBJECT = "Reset Your EstateMind Password"


def _build_reset_email(reset_url: str) -> tuple[str, str]:
    """Return (plain_text, html) bodies for the password reset email."""

    text = f"""\
Hello,

You requested to reset your password for your EstateMind account.

Open the link below to choose a new password:
{reset_url}

This link expires in {RESET_TOKEN_TTL_MINUTES} minutes and can only be used once.

If you did not request a password reset, you can safely ignore this
email — your password will not be changed.

Need help? Reply to this email and our team will assist you.

Thanks,
The EstateMind Team
"""

    html = f"""\
<html>
  <body style="margin: 0; padding: 0; background-color: #f4f6f4; font-family: Arial, Helvetica, sans-serif;">
    <div style="max-width: 600px; margin: 0 auto; padding: 32px 16px;">
      <div style="background-color: #ffffff; border-radius: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); overflow: hidden;">

        <!-- Header / logo -->
        <div style="background-color: #16a34a; padding: 24px; text-align: center;">
          <span style="font-size: 22px; font-weight: bold; color: #ffffff; letter-spacing: 0.5px;">
            &#127969; EstateMind
          </span>
        </div>

        <!-- Body -->
        <div style="padding: 32px; color: #333333; line-height: 1.6;">
          <h2 style="margin-top: 0; color: #14532d;">Reset your password</h2>
          <p>Hello,</p>
          <p>We received a request to reset the password for your EstateMind account.</p>
          <p>Click the button below to choose a new password:</p>

          <p style="text-align: center; margin: 32px 0;">
            <a href="{reset_url}"
               style="background-color: #16a34a; color: #ffffff; padding: 14px 32px;
                      text-decoration: none; border-radius: 8px; display: inline-block;
                      font-weight: bold; font-size: 16px;">
              Reset Password
            </a>
          </p>

          <p style="font-size: 14px; color: #555555;">
            Or copy and paste this link into your browser:
          </p>
          <p style="word-break: break-all; font-size: 13px;">
            <a href="{reset_url}" style="color: #16a34a;">{reset_url}</a>
          </p>

          <div style="background-color: #f0fdf4; border-left: 4px solid #16a34a;
                      padding: 12px 16px; margin: 24px 0; border-radius: 4px;">
            <p style="margin: 0; font-size: 14px; color: #14532d;">
              &#9200; This link expires in <strong>{RESET_TOKEN_TTL_MINUTES} minutes</strong>
              and can only be used once.
            </p>
          </div>

          <p style="font-size: 14px; color: #666666;">
            &#128274; <strong>Didn't request this?</strong> You can safely ignore
            this email — your password will not be changed. If you're concerned
            about your account's security, please contact our support team.
          </p>
        </div>

        <!-- Footer -->
        <div style="border-top: 1px solid #e5e7eb; padding: 20px 32px; text-align: center;">
          <p style="margin: 0; font-size: 12px; color: #999999;">
            Need help? Reply to this email and our team will assist you.<br>
            &copy; EstateMind — AI-powered real estate insights
          </p>
        </div>

      </div>
    </div>
  </body>
</html>
"""
    return text, html


# -------------------------------
# Email providers
# -------------------------------
# Each provider implements send(to_email, subject, text, html) -> bool and
# reads its own configuration from the environment. Templates and callers
# are provider-agnostic.


class GmailSmtpProvider:
    """Delivers email over SMTP (Gmail 587 STARTTLS / 465 SSL). Local default."""

    name = "gmail-smtp"

    def __init__(self) -> None:
        self.host = os.getenv("SMTP_HOST", "")
        self.port = int(os.getenv("SMTP_PORT", "587"))
        self.username = os.getenv("SMTP_USERNAME", "")
        self.password = os.getenv("SMTP_PASSWORD", "")
        self.from_addr = os.getenv("SMTP_FROM", "noreply@estatemind.com")

    def is_configured(self) -> bool:
        return bool(self.host and self.username and self.password)

    def missing_config(self) -> list[str]:
        return [
            name
            for name, value in [
                ("SMTP_HOST", self.host),
                ("SMTP_USERNAME", self.username),
                ("SMTP_PASSWORD", self.password),
            ]
            if not value
        ]

    def send(self, to_email: str, subject: str, text: str, html: str) -> bool:
        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = f"EstateMind <{self.from_addr}>"
        message["To"] = to_email
        message.attach(MIMEText(text, "plain"))
        message.attach(MIMEText(html, "html"))

        try:
            if self.port == 465:
                # Implicit SSL (e.g. Gmail port 465)
                with smtplib.SMTP_SSL(self.host, self.port, timeout=30) as server:
                    server.login(self.username, self.password)
                    server.sendmail(self.from_addr, to_email, message.as_string())
            else:
                # STARTTLS (e.g. Gmail port 587)
                with smtplib.SMTP(self.host, self.port, timeout=30) as server:
                    server.starttls()
                    server.login(self.username, self.password)
                    server.sendmail(self.from_addr, to_email, message.as_string())

            logger.info("[gmail-smtp] Email sent to %s", to_email)
            return True

        except smtplib.SMTPAuthenticationError as e:
            logger.error(
                "[gmail-smtp] Authentication failed for %s@%s:%s — check "
                "SMTP_USERNAME/SMTP_PASSWORD (Gmail requires an App Password): %s",
                self.username, self.host, self.port, e,
            )
            return False
        except Exception as e:
            logger.error(
                "[gmail-smtp] Failed to send email to %s via %s:%s — %s",
                to_email, self.host, self.port, e,
            )
            return False


class ResendProvider:
    """Delivers email over the Resend HTTPS API. Used on Railway, where
    outbound SMTP ports are blocked."""

    name = "resend"

    def __init__(self) -> None:
        self.api_key = os.getenv("RESEND_API_KEY", "")
        # EMAIL_FROM is the production sender; fall back to SMTP_FROM so a
        # single configured sender keeps working across providers.
        self.from_addr = (
            os.getenv("EMAIL_FROM")
            or os.getenv("SMTP_FROM")
            or "onboarding@resend.dev"
        )

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def send(self, to_email: str, subject: str, text: str, html: str) -> bool:
        try:
            import resend
        except ImportError as e:
            logger.error(
                "[resend] RESEND_API_KEY is set but the 'resend' package is "
                "not installed — run: pip install resend (%s)", e,
            )
            return False

        try:
            resend.api_key = self.api_key
            response = resend.Emails.send(
                {
                    "from": f"EstateMind <{self.from_addr}>",
                    "to": [to_email],
                    "subject": subject,
                    "text": text,
                    "html": html,
                }
            )
            logger.info(
                "[resend] Email sent to %s (id=%s)",
                to_email, (response or {}).get("id"),
            )
            return True

        except Exception as e:
            logger.error(
                "[resend] Failed to send email to %s (from=%s) — %s",
                to_email, self.from_addr, e,
            )
            return False


def _select_provider() -> GmailSmtpProvider | ResendProvider | None:
    """
    Pick the email provider from configuration alone:
    RESEND_API_KEY present -> Resend (HTTPS, Railway-safe);
    otherwise SMTP configured -> Gmail SMTP; otherwise None.
    """
    resend_provider = ResendProvider()
    if resend_provider.is_configured():
        return resend_provider

    gmail_provider = GmailSmtpProvider()
    if gmail_provider.is_configured():
        return gmail_provider

    return None


def send_password_reset_email(email: str, reset_token: str) -> bool:
    """
    Send a password reset email containing the reset link.
    Returns True only when the email was actually accepted by a provider.
    """

    frontend_base_url = os.getenv("FRONTEND_BASE_URL", "http://localhost:3000").rstrip("/")
    reset_url = f"{frontend_base_url}/reset-password?token={reset_token}"

    provider = _select_provider()

    # No provider configured: never fake success. Log the reset URL so local
    # development without credentials remains testable, and return False.
    if provider is None:
        missing = ["RESEND_API_KEY"] + GmailSmtpProvider().missing_config()
        logger.warning(
            "Email NOT sent — no email provider configured (missing: %s). "
            "Reset URL for %s: %s",
            ", ".join(missing),
            email,
            reset_url,
        )
        return False

    logger.info("Sending password reset email to %s via %s", email, provider.name)

    text, html = _build_reset_email(reset_url)
    sent = provider.send(email, EMAIL_SUBJECT, text, html)

    if not sent:
        logger.error(
            "Password reset email FAILED for %s via %s — see provider error above",
            email, provider.name,
        )
    return sent
