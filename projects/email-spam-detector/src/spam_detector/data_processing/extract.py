"""Extract normalized records from email messages and mailboxes."""

from email.header import decode_header, make_header
from email.utils import parsedate_to_datetime

from bs4 import BeautifulSoup


def decode_header_field(value):
    if value is None:
        return ""

    return str(make_header(decode_header(value)))


def extract_text(message):
    plain_text = ""
    html_text = ""

    for part in message.walk():
        # Do not include attachments as email text.
        if part.get_content_disposition() == "attachment":
            continue

        content_type = part.get_content_type()

        # Multipart containers have no text content of their own.
        if content_type not in ["text/plain", "text/html"]:
            continue

        payload = part.get_payload(decode=True)

        if payload is None:
            continue

        charset = part.get_content_charset() or "utf-8"

        try:
            decoded_content = payload.decode(charset, errors="replace")
        except LookupError:
            decoded_content = payload.decode("utf-8", errors="replace")

        if content_type == "text/plain" and not plain_text:
            plain_text = decoded_content.strip()
        elif content_type == "text/html" and not html_text:
            soup = BeautifulSoup(decoded_content, "html.parser")
            html_text = soup.get_text(separator="\n", strip=True)

    if plain_text:
        return plain_text

    if html_text:
        return html_text

    return ""


def extract_mail(message, label, source):
    sender = decode_header_field(message["From"])
    subject = decode_header_field(message["Subject"])
    text = extract_text(message)
    raw_date = message["Date"]

    if raw_date:
        try:
            date = parsedate_to_datetime(raw_date).isoformat()
        except (TypeError, ValueError):
            date = None
    else:
        date = None

    return {
        "sender": sender,
        "subject": subject,
        "text": text,
        "date": date,
        "label": label,
        "source": source,
        "source_split": "private",
    }


def extract_mailbox(mbox, label, source):
    records = []

    for message in mbox:
        record = extract_mail(message=message, label=label, source=source)
        records.append(record)

    return records
