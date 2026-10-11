"""Input checks only: these functions do not establish legal correctness."""

CUSTOMER_SEGMENTS = ("Unknown", "Medium Business", "Enterprise", "Government")
HANDLING_APPROACHES = ("Unknown", "Standard/Digital-eligible", "Assisted/Custom")
DIGITAL_ELIGIBILITY = ("Unknown", "Yes", "No")
INTAKE_FIELDS = {"customer_name", "description", "customer_segment", "handling_approach",
                 "digital_eligibility"}


def validate_intake(intake: dict) -> dict[str, str]:
    """Return field errors; an empty dictionary means the intake can be saved.

    Unknown categories are valid at submission. Saving is not reviewer approval.
    strip() checks for blank text without changing the original input.
    """
    errors = {}
    for field, label in (("customer_name", "Customer legal name"),
                         ("description", "Request description")):
        value = intake.get(field)
        if not isinstance(value, str) or not value.strip():
            errors[field] = f"{label} is required. Enter synthetic text."

    if intake.get("customer_segment") not in CUSTOMER_SEGMENTS:
        errors["customer_segment"] = "Choose one of the listed customer segments."
    if intake.get("handling_approach") not in HANDLING_APPROACHES:
        errors["handling_approach"] = "Choose one of the listed handling approaches."
    # Older cases did not have this field; retain them without inventing a value.
    if "digital_eligibility" in intake and intake["digital_eligibility"] not in DIGITAL_ELIGIBILITY:
        errors["digital_eligibility"] = "Choose Unknown, Yes, or No for digital eligibility."
    if set(intake) - INTAKE_FIELDS:
        errors["fields"] = "The intake contains unsupported fields."
    return errors
