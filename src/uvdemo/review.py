"""Reviewer confirmation checks. Routing and staff selection come later."""

from uvdemo.intake import CUSTOMER_SEGMENTS, HANDLING_APPROACHES

DIGITAL_ELIGIBILITY = ("Unknown", "Yes", "No")
REVIEW_FIELDS = {"contract_type", "customer_segment", "handling_approach", "digital_eligibility",
                 "facts_confirmed", "issues_resolved", "notes"}


def validate_review(review: dict, document_extraction: dict | None = None,
                    analysis_result: dict | None = None) -> dict[str, str]:
    """Check confirmed demo facts without assessing legal correctness.

    Government/Enterprise need a confirmed segment. Medium Business also needs
    confirmed handling and explicit digital eligibility. Contract type is
    descriptive and does not create additional routing criteria.
    """
    errors = {}
    if set(review) != REVIEW_FIELDS:
        errors["fields"] = "Review must contain exactly the listed confirmation fields."
    if type(review.get("issues_resolved")) is not bool:
        errors["issues_resolved"] = "Issue resolution must be explicitly checked or unchecked."
    segment = review.get("customer_segment")
    handling = review.get("handling_approach")
    digital = review.get("digital_eligibility")
    if segment not in CUSTOMER_SEGMENTS or segment == "Unknown":
        errors["customer_segment"] = "Confirm the customer segment before completing review."
    if handling not in HANDLING_APPROACHES:
        errors["handling_approach"] = "Choose a listed handling approach."
    if digital not in DIGITAL_ELIGIBILITY:
        errors["digital_eligibility"] = "Choose Unknown, Yes, or No for digital eligibility."
    if segment == "Medium Business":
        if handling == "Unknown":
            errors["handling_approach"] = "Confirm the handling approach for Medium Business."
        if digital == "Unknown":
            errors["digital_eligibility"] = "Confirm digital eligibility for Medium Business."
    if ((digital == "Yes" and handling == "Assisted/Custom") or
            (digital == "No" and handling == "Standard/Digital-eligible")):
        errors["digital_eligibility"] = "Digital eligibility conflicts with the handling approach."
    if review.get("facts_confirmed") is not True:
        errors["facts_confirmed"] = "Confirm that you have checked the required facts."
    if not isinstance(review.get("contract_type"), str) or not review["contract_type"].strip():
        errors["contract_type"] = "Enter a contract type or Unknown."
    if not isinstance(review.get("notes"), str) or not review["notes"].strip():
        errors["notes"] = "Record how you confirmed the facts in the review notes."

    unresolved = (analysis_result is None or
                  bool(analysis_result.get("contradictions")) or
                  bool(analysis_result.get("clarification_questions")) or
                  (document_extraction is not None and
                   document_extraction.get("status") != "Ready"))
    if unresolved and review.get("issues_resolved") is not True:
        errors["issues_resolved"] = (
            "Resolve the listed issues, or complete manual review, and acknowledge this below."
        )
    return errors
