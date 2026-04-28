import os
import re
import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage
from typing import Any

from google.adk.agents.llm_agent import Agent
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from pymongo import MongoClient


class Observation(BaseModel):
    """One visible finding. Use box_2d whenever something is localized so the reader knows where to look."""

    observation: str = Field(..., description='Short label, e.g. "Right lower lobe opacity".')
    notes: str = Field(..., description='What you see and why it matters.')
    confidence: float = Field(..., ge=0.0, le=1.0)
    box_2d: list[float] | None = Field(
        default=None,
        description='Region [ymin, xmin, ymax, xmax] in pixels or normalized 0–1; null only if not localized.',
    )

    @field_validator('box_2d')
    @classmethod
    def box_len(cls, value: list[float] | None) -> list[float] | None:
        if value is None:
            return None
        if len(value) != 4:
            raise ValueError('box_2d must be four numbers [ymin, xmin, ymax, xmax] or null.')
        return value


class PossibleDiagnosis(BaseModel):
    """One differential-style hypothesis, not a definitive diagnosis."""

    diagnosis: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    from_observations: list[str] = Field(
        default_factory=list,
        description='Copy the observation strings exactly as in observations[].observation.',
    )
    how_they_support: str = Field(
        ...,
        description='One short paragraph linking those observations to this diagnosis.',
    )


class MedicalReport(BaseModel):
    """Stored every time; same keys for every study."""

    version: int = Field(default=1, description='Schema version; always use 1.')
    image_context: str = Field(
        ...,
        description='What the image is (modality + body part + view if clear).',
    )
    observations: list[Observation] = Field(default_factory=list)
    possible_diagnoses: list[PossibleDiagnosis] = Field(default_factory=list)
    summary: str = Field(..., description='Overall takeaway in plain language.')
    summary_confidence: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode='after')
    def diagnosis_observations_exist(self) -> 'MedicalReport':
        titles = {o.observation for o in self.observations}
        for dx in self.possible_diagnoses:
            for name in dx.from_observations:
                if name not in titles:
                    raise ValueError(
                        f'Diagnosis "{dx.diagnosis}" references unknown observation "{name}". '
                        'Each entry must match observations[].observation exactly.'
                    )
        return self


def is_valid_email(email: str) -> bool:
    """Return True if email format looks valid."""
    return bool(re.fullmatch(r'[^@\s]+@[^@\s]+\.[^@\s]+', email))


def save_medical_image_report(
    recipient_email: str,
    report: dict[str, Any],
    patient_name: str = '',
) -> dict[str, str]:
    """Save report to MongoDB. ``report`` is a JSON object validated as MedicalReport."""
    try:
        validated = MedicalReport.model_validate(report)
    except ValidationError as exc:
        return {
            'status': 'error',
            'message': f'Invalid report: {exc}',
        }

    mongodb_uri = os.getenv('MONGODB_URI')
    db_name = os.getenv('MONGODB_DB', 'medical_agent')
    collection_name = os.getenv('MONGODB_COLLECTION', 'medical_image_reports')

    if not mongodb_uri:
        return {
            'status': 'error',
            'message': 'MONGODB_URI is not set in environment.',
        }

    document = {
        'recipient_email': recipient_email,
        'patient_name': patient_name,
        'report': validated.model_dump(),
        'created_at': datetime.now(timezone.utc).isoformat(),
    }

    client = MongoClient(mongodb_uri)
    try:
        result = client[db_name][collection_name].insert_one(document)
    finally:
        client.close()

    return {
        'status': 'ok',
        'inserted_id': str(result.inserted_id),
        'db': db_name,
        'collection': collection_name,
    }


def send_findings_email(
    recipient_email: str,
    subject: str,
    body: str,
) -> dict[str, str]:
    """Send the report over SMTP. Configure a single From identity in .env; To is always the user’s contact email."""
    host = os.getenv('SMTP_HOST', '')
    port = int(os.getenv('SMTP_PORT', '587'))
    user = os.getenv('SMTP_USER', '')
    password = os.getenv('SMTP_PASS', '')
    from_email = os.getenv('SMTP_FROM', user)
    use_tls = os.getenv('SMTP_USE_TLS', 'true').lower() == 'true'
    if not host or not from_email:
        return {
            'status': 'error',
            'message': (
                'Set SMTP in my_agent/.env: SMTP_HOST, SMTP_FROM, SMTP_USER, SMTP_PASS (e.g. '
                'Gmail App Password + smtp.gmail.com:587). Recipients are not pre-registered; '
                'To is the contact email in each request.'
            ),
        }

    message = EmailMessage()
    message['Subject'] = subject
    message['From'] = from_email
    message['To'] = recipient_email
    message.set_content(body)

    try:
        with smtplib.SMTP(host, port, timeout=30) as server:
            if use_tls:
                server.starttls()
            if user and password:
                server.login(user, password)
            server.send_message(message)
    except OSError as exc:
        return {'status': 'error', 'message': f'SMTP error: {exc}'}

    return {
        'status': 'ok',
        'to_email': recipient_email,
        'note': 'Gmail: smtp.gmail.com + App Password in SMTP_PASS. Check spam for the addressee.',
    }


def _resolve_model_name() -> str:
    """ADK_MODEL wins (e.g. ollama/qwen2.5vl:7b). Else GEMINI_MODEL. Else Gemini default."""
    return (
        os.getenv('GEMINI_MODEL', '').strip()
        or os.getenv('ADK_MODEL', '').strip()
        or 'gemini-2.5-flash'
    )


root_agent = Agent(
    # Cloud: gemini-2.5-flash (or set GEMINI_MODEL). Local Ollama: pip install litellm, then e.g.
    # ADK_MODEL=ollama/qwen2.5vl:7b
    model=_resolve_model_name(),
    name='medical_image_agent',
    description='Generic medical image triage: observations, differentials, MongoDB, email.',
    instruction=(
        'You support clinicians on medical images of any kind.\n'
        'Input: the image plus a contact email (optional patient name).\n'
        '\n'
        '1) is_valid_email on the address; if bad or missing, ask only for a valid email.\n'
        '2) Build one MedicalReport: version=1, image_context, observations (each: observation, '
        'notes, confidence 0–1, box_2d with [ymin,xmin,ymax,xmax] whenever something is localized '
        'so the reader knows where to look; null only when not a focal region), '
        'possible_diagnoses (each: diagnosis, confidence, from_observations list of strings that '
        'must exactly match observations[].observation, plus how_they_support), '
        'summary, summary_confidence.\n'
        'Use observations=[] / possible_diagnoses=[] when appropriate and say so in summary.\n'
        'This is not definitive diagnosis.\n'
        '3) save_medical_image_report(recipient_email, report as a JSON object / dict, '
        'patient_name if known) — same keys as MedicalReport; validation runs in the tool.\n'
        '4) send_findings_email: subject + body listing context, each observation with coords, '
        'each possible diagnosis with linked observations and next-step suggestions.\n'
        'After the tool returns, say the status and any note. The addressee is the contact email '
        'in the request; one SMTP session identity is configured in environment, not per recipient.\n'
        'Do not ask for file paths or base64.'
    ),
    tools=[is_valid_email, save_medical_image_report, send_findings_email],
)
