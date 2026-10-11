import unittest

from uvdemo.review import validate_review


def confirmed_review():
    return {"contract_type": "Unknown", "customer_segment": "Government",
            "handling_approach": "Unknown", "digital_eligibility": "Unknown",
            "facts_confirmed": True, "issues_resolved": True,
            "notes": "Manually confirmed the synthetic customer's segment."}


class ReviewTests(unittest.TestCase):
    def test_government_and_enterprise_do_not_require_digital_eligibility(self):
        for segment in ("Government", "Enterprise"):
            review = confirmed_review()
            review["customer_segment"] = segment
            self.assertEqual(validate_review(review), {})

    def test_unknown_segment_blocks_confirmation(self):
        review = confirmed_review()
        review["customer_segment"] = "Unknown"
        self.assertIn("customer_segment", validate_review(review))

    def test_medium_business_requires_handling_and_digital_confirmation(self):
        review = confirmed_review()
        review["customer_segment"] = "Medium Business"
        errors = validate_review(review)
        self.assertIn("handling_approach", errors)
        self.assertIn("digital_eligibility", errors)
        review.update(handling_approach="Assisted/Custom", digital_eligibility="No")
        self.assertEqual(validate_review(review), {})

    def test_conflicting_confirmed_handling_and_eligibility_is_blocked(self):
        review = confirmed_review()
        review.update(handling_approach="Standard/Digital-eligible", digital_eligibility="No")
        self.assertIn("digital_eligibility", validate_review(review))

    def test_explicit_confirmation_and_notes_are_required(self):
        review = confirmed_review()
        review.update(facts_confirmed=False, notes=" ")
        errors = validate_review(review)
        self.assertIn("facts_confirmed", errors)
        self.assertIn("notes", errors)

    def test_ai_issues_and_unreadable_document_need_manual_resolution(self):
        review = confirmed_review()
        review["issues_resolved"] = False
        for result, document in (
            (None, None),
            ({"contradictions": ["Segment mismatch"]}, None),
            ({"clarification_questions": ["Which customer?"]}, None),
            ({"contradictions": [], "clarification_questions": []}, {"status": "Needs attention"}),
        ):
            with self.subTest(result=result, document=document):
                self.assertIn("issues_resolved", validate_review(review, document, result))


if __name__ == "__main__":
    unittest.main()
