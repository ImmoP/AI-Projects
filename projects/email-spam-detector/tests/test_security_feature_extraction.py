"""
Tests for security feature extraction (SPF/DKIM/DMARC/spoofing signals)
against the synthetic .eml fixtures in tests/fixtures/eml/.

Scope: extract_message_information() / extract_security_row() only --
pure RFC822 header parsing, no ML model involved. classify_eml() (GPT-2 +
Security V1/V2 + temporal reputation history) is not exercised here; those
three model/data artifacts are private and deliberately excluded from the
repository (see .gitignore), so they cannot run in CI.

Assertions check well-formedness and that the extractor is sensitive to
each signal -- not that any downstream classifier is accurate.
"""

from email import policy
from email.parser import BytesParser
from pathlib import Path

import pytest
from spam_detector.inspection.extract_security_features import extract_security_row
from spam_detector.inspection.inspect_email_headers import extract_message_information

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "eml"
ALL_FIXTURES = sorted(p.name for p in FIXTURES_DIR.glob("*.eml"))


def load_eml(name: str):
    path = FIXTURES_DIR / name
    with path.open("rb") as f:
        return BytesParser(policy=policy.default).parse(f)


def security_row(name: str) -> dict:
    message = load_eml(name)
    return extract_security_row(
        message=message,
        mbox_path=FIXTURES_DIR / name,
        message_index=0,
        source="test_fixture",
    )


@pytest.mark.smoke
def test_ten_fixtures_present():
    assert len(ALL_FIXTURES) == 10


@pytest.mark.smoke
@pytest.mark.parametrize("filename", ALL_FIXTURES)
def test_extraction_does_not_crash_and_is_well_formed(filename):
    row = security_row(filename)

    required_keys = {
        "spf_result", "dkim_result", "dmarc_result",
        "from_domain", "from_org_domain",
        "has_authentication_results", "has_dkim_signature",
        "spf_pass", "spf_fail", "dkim_pass", "dkim_fail", "dmarc_pass", "dmarc_fail",
    }
    assert required_keys <= row.keys()

    for key in ("spf_result", "dkim_result", "dmarc_result", "from_domain"):
        assert isinstance(row[key], str)
    for key in ("has_authentication_results", "has_dkim_signature", "spf_pass", "spf_fail"):
        assert isinstance(row[key], bool)


@pytest.mark.smoke
def test_spf_pass():
    row = security_row("spf_pass.eml")
    assert row["spf_result"] == "pass"
    assert row["spf_pass"] is True
    assert row["spf_fail"] is False


@pytest.mark.smoke
def test_spf_fail():
    row = security_row("spf_fail.eml")
    assert row["spf_result"] == "fail"
    assert row["spf_fail"] is True
    assert row["spf_pass"] is False


@pytest.mark.smoke
def test_spf_missing():
    row = security_row("spf_missing.eml")
    assert row["spf_result"] == ""
    assert row["spf_pass"] is False
    assert row["spf_fail"] is False


@pytest.mark.smoke
def test_dkim_missing():
    row = security_row("dkim_missing.eml")
    assert row["has_dkim_signature"] is False
    assert row["dkim_result"] == ""


@pytest.mark.smoke
def test_dkim_invalid():
    row = security_row("dkim_invalid.eml")
    assert row["has_dkim_signature"] is True
    assert row["dkim_result"] == "fail"
    assert row["dkim_fail"] is True


@pytest.mark.smoke
def test_dmarc_alignment_break():
    row = security_row("dmarc_alignment_break.eml")
    assert row["dmarc_result"] == "fail"
    assert row["dmarc_fail"] is True
    # SPF/DKIM pass, but for the relay domain, not the From domain.
    assert row["from_domain"] == "example.com"
    assert row["from_dkim_org_match"] is False


@pytest.mark.smoke
def test_display_name_spoofing():
    message = load_eml("display_name_spoofing.eml")
    info = extract_message_information(
        message=message,
        mbox_path=FIXTURES_DIR / "display_name_spoofing.eml",
        message_index=0,
    )
    assert "Example Corp" in info["display_name"]
    assert info["from_domain"] == "example-corp-verify.example"
    assert info["from_domain"] != "example.com"


@pytest.mark.smoke
def test_homoglyph_sender_domain_is_not_ascii():
    row = security_row("homoglyph_sender.eml")
    assert row["from_domain"] != "example.com"
    assert not row["from_domain"].isascii()
    # Same visual length as example.com -- one character swapped for a
    # Cyrillic look-alike, not a structurally different domain.
    assert len(row["from_domain"]) == len("example.com")


@pytest.mark.smoke
def test_clean_ham_baseline():
    row = security_row("clean_ham.eml")
    assert row["spf_result"] == "pass"
    assert row["dkim_result"] == "pass"
    assert row["dmarc_result"] == "pass"
    assert row["from_domain"] == "example.com"


@pytest.mark.smoke
def test_no_auth_headers():
    row = security_row("no_auth_headers.eml")
    assert row["has_authentication_results"] is False
    assert row["has_dkim_signature"] is False
    assert row["spf_result"] == ""
    assert row["dkim_result"] == ""
    assert row["dmarc_result"] == ""


@pytest.mark.smoke
def test_spf_fail_feature_vector_differs_from_clean_ham():
    """
    Core acceptance criterion: an SPF-fail message must produce a
    different security feature vector than the clean baseline. This
    checks that feature extraction is sensitive to the signal -- it
    makes no claim about downstream model accuracy.
    """
    clean = security_row("clean_ham.eml")
    spf_fail = security_row("spf_fail.eml")

    assert clean != spf_fail
    assert clean["spf_result"] != spf_fail["spf_result"]
    assert clean["spf_pass"] != spf_fail["spf_pass"]
