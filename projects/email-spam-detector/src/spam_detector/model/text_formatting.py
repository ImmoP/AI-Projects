# Formating the modelinput
# The 3 columns: sender, subject, text are merged into one string
# and formated into a json
# => Creates text representation

"""
1. Cleans sender, subject and body;
2. Adds only existing fields;
3. Provides clear field labels;
4. ombines everything into a model text."""


def clean_value(value) -> str:
    if value is None:
        return ""

    return str(value).strip()


def format_email(sender, subject, text) -> str:
    sender = clean_value(sender)
    subject = clean_value(subject)
    text = clean_value(text)

    sections = []

    if sender:
        sections.append(f"Sender: {sender}")

    if subject:
        sections.append(f"Subject: {subject}")

    if text:
        sections.append(f"Body:\n{text}")

    return "\n\n".join(sections)