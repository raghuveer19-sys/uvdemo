"""Offline tests: mocked provider calls never contact OpenAI."""

from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from openai import (
    APIConnectionError, APITimeoutError, AuthenticationError, BadRequestError,
    InternalServerError, NotFoundError, PermissionDeniedError, RateLimitError,
    UnprocessableEntityError,
)

from uvdemo.analysis import AnalysisError, analyze_request, validate_analysis


def intake(**changes):
    return {"customer_name": "Fictional Acorn", "description": "A synthetic request.",
            "customer_segment": "Unknown", "handling_approach": "Unknown", **changes}


def document(text="Contract type: NDA"):
    return {"filename": "synthetic.pdf", "status": "Ready", "text": text,
            "pages": [{"page": 1, "text": text}], "error": None}


class AnalysisTests(unittest.TestCase):
    def test_mock_is_labelled_deterministic_and_does_not_infer_digital_from_nda(self):
        request = intake(description="Contract type: NDA\nPlease use electronic signatures.")
        with patch("uvdemo.analysis.OpenAI") as provider:
            result = analyze_request(request)
            self.assertEqual(result, analyze_request(request))
            provider.assert_not_called()
        self.assertTrue(result["summary"].startswith("MOCK -"))
        self.assertEqual(result["contract_type"], "NDA")
        self.assertEqual(result["handling_approach"], "Unknown")
        self.assertEqual(result["digital_eligibility"], "Unknown")
        self.assertTrue(result["clarification_questions"])

    def test_explicit_document_categories_fill_unknown_intake_without_inference(self):
        uploaded = document("Customer segment: Medium Business\n"
                            "Handling approach: Standard/Digital-eligible\n"
                            "Digital eligibility: Yes")
        result = analyze_request(intake(), uploaded)
        self.assertEqual(result["customer_segment"], "Medium Business")
        self.assertEqual(result["handling_approach"], "Standard/Digital-eligible")
        self.assertEqual(result["digital_eligibility"], "Yes")
        self.assertEqual(result["contract_type"], "Unknown")
        self.assertFalse(result["contradictions"])
        self.assertFalse(result["clarification_questions"])
        self.assertTrue(all(fact["source"] == "document" for fact in result["facts"]))
        validate_analysis(result, intake(), uploaded)

    def test_medium_business_requires_explicit_digital_eligibility_and_handling(self):
        request = intake(customer_segment="Medium Business",
                         description="Contract type: NDA\nElectronic signing requested.")
        result = analyze_request(request)
        self.assertEqual(result["digital_eligibility"], "Unknown")
        self.assertEqual(result["handling_approach"], "Unknown")
        self.assertEqual(len(result["clarification_questions"]), 2)
        self.assertTrue(any("digital eligibility" in question for question in result["clarification_questions"]))
        result = analyze_request(intake(customer_segment="Medium Business",
                                        handling_approach="Standard/Digital-eligible"))
        self.assertEqual(result["digital_eligibility"], "Unknown")
        self.assertEqual(len(result["clarification_questions"]), 1)

    def test_explicit_digital_no_with_custom_handling_needs_no_question(self):
        request = intake(customer_segment="Medium Business", handling_approach="Assisted/Custom",
                         description="Digital eligibility: No")
        result = analyze_request(request)
        self.assertEqual(result["digital_eligibility"], "No")
        self.assertFalse(result["contradictions"])
        self.assertFalse(result["clarification_questions"])

    def test_explicit_intake_digital_eligibility_has_its_own_evidence(self):
        request = intake(customer_segment="Medium Business",
                         handling_approach="Standard/Digital-eligible", digital_eligibility="Yes")
        result = analyze_request(request)
        self.assertEqual(result["digital_eligibility"], "Yes")
        self.assertFalse(result["contradictions"])
        self.assertFalse(result["clarification_questions"])
        self.assertIn({"fact": "Submitter reports digital eligibility: Yes.",
                       "source": "digital_eligibility", "page": None, "quote": "Yes"}, result["facts"])
        validate_analysis(result, request)

    def test_intake_digital_claim_conflicting_with_pdf_remains_unresolved(self):
        request = intake(customer_segment="Medium Business",
                         handling_approach="Standard/Digital-eligible", digital_eligibility="Yes")
        uploaded = document("Digital eligibility: No")
        result = analyze_request(request, uploaded)
        self.assertEqual(result["digital_eligibility"], "Unknown")
        self.assertTrue(any("digital eligibility" in item for item in result["contradictions"]))
        self.assertTrue(result["clarification_questions"])
        self.assertTrue(any(fact["source"] == "digital_eligibility" for fact in result["facts"]))
        self.assertTrue(any(fact["source"] == "document" for fact in result["facts"]))
        validate_analysis(result, request, uploaded)

    def test_unknown_intake_digital_claim_can_be_resolved_by_document(self):
        request = intake(customer_segment="Medium Business",
                         handling_approach="Standard/Digital-eligible", digital_eligibility="Unknown")
        result = analyze_request(request, document("Digital eligibility: Yes"))
        self.assertEqual(result["digital_eligibility"], "Yes")
        self.assertFalse(result["contradictions"])
        self.assertFalse(result["clarification_questions"])

    def test_government_and_enterprise_do_not_require_handling_or_contract_type(self):
        for segment in ("Government", "Enterprise"):
            with self.subTest(segment=segment):
                result = analyze_request(intake(), document(f"Customer segment: {segment}"))
                self.assertEqual(result["customer_segment"], segment)
                self.assertEqual(result["contract_type"], "Unknown")
                self.assertEqual(result["handling_approach"], "Unknown")
                self.assertEqual(result["digital_eligibility"], "Unknown")
                self.assertFalse(result["clarification_questions"])

    def test_conflicting_digital_claims_or_incompatible_handling_block_suggestions(self):
        request = intake(customer_segment="Medium Business", handling_approach="Assisted/Custom",
                         description="Digital eligibility: No")
        result = analyze_request(request, document("Digital eligibility: Yes"))
        self.assertEqual(result["digital_eligibility"], "Unknown")
        self.assertTrue(result["contradictions"])
        self.assertTrue(result["clarification_questions"])
        result = analyze_request(intake(customer_segment="Medium Business", handling_approach="Assisted/Custom",
                                        description="Digital eligibility: Yes"))
        self.assertTrue(any("handling approach" in item for item in result["contradictions"]))

    def test_unknown_in_one_source_does_not_conflict_with_known_category_elsewhere(self):
        request = intake(description="Customer segment: Unknown\nDigital eligibility: Unknown")
        uploaded = document("Customer segment: Medium Business\nDigital eligibility: Yes\n"
                            "Handling approach: Standard/Digital-eligible")
        result = analyze_request(request, uploaded)
        self.assertEqual(result["customer_segment"], "Medium Business")
        self.assertEqual(result["digital_eligibility"], "Yes")
        self.assertFalse(result["contradictions"])
        self.assertFalse(result["clarification_questions"])

    def test_mock_conflicting_sources_remain_unknown_with_exact_evidence(self):
        request = intake(description="Contract type: Services agreement\nCustomer segment: Government",
                         customer_segment="Medium Business")
        uploaded = document()
        result = analyze_request(request, uploaded)
        self.assertEqual(result["customer_segment"], "Unknown")
        self.assertEqual(result["contract_type"], "Unknown")
        self.assertEqual(len(result["contradictions"]), 2)
        self.assertTrue(any(fact["source"] == "document" for fact in result["facts"]))
        validate_analysis(result, request, uploaded)

    def test_mock_flags_customer_government_indicator_without_inferring_supplier_segment(self):
        result = analyze_request(intake(customer_segment="Medium Business",
                                       description="The contracting customer is a government department."))
        self.assertEqual(result["customer_segment"], "Unknown")
        self.assertIn("Possible", result["contradictions"][0])
        result = analyze_request(intake(customer_segment="Medium Business",
                                       description="Our supplier is a government department."))
        self.assertEqual(result["customer_segment"], "Medium Business")
        self.assertFalse(result["contradictions"])
        self.assertFalse(any("government" in question.casefold() for question in result["clarification_questions"]))

    def test_invalid_input_failed_extraction_and_oversized_input_never_call_provider(self):
        inputs = [(intake(customer_segment="Huge"), None),
                  (intake(), {"status": "Needs attention"}),
                  (intake(description="x" * 100_001), None)]
        for request, uploaded in inputs:
            with self.subTest(request_length=len(request["description"])):
                with patch("uvdemo.analysis.OpenAI") as provider:
                    with self.assertRaises(AnalysisError):
                        analyze_request(request, uploaded, "live", api_key="test-key")
                    provider.assert_not_called()

    def test_missing_key_is_friendly_and_mock_needs_no_key(self):
        with patch.dict("os.environ", {}, clear=True), patch("uvdemo.analysis.OpenAI") as provider:
            with self.assertRaisesRegex(AnalysisError, "OPENAI_API_KEY"):
                analyze_request(intake(), mode="live")
            analyze_request(intake())
            provider.assert_not_called()

    def test_live_uses_one_structured_call_both_sources_no_retries_and_closes_client(self):
        request, uploaded = intake(), document()
        suggestions = analyze_request(request, uploaded)
        suggestions["summary"] = "An unconfirmed synthetic NDA request."
        with patch("uvdemo.analysis.OpenAI") as provider:
            client = provider.return_value.__enter__.return_value
            client.responses.create.return_value = SimpleNamespace(
                status="completed", output=[], output_text=json.dumps(suggestions))
            result = analyze_request(request, uploaded, "live", api_key="test-only-key", model="chosen-model")
            self.assertEqual(result, suggestions)
            provider.assert_called_once_with(api_key="test-only-key", max_retries=0, timeout=30.0)
            client.responses.create.assert_called_once()
            call = client.responses.create.call_args.kwargs
            self.assertEqual(call["model"], "chosen-model")
            self.assertFalse(call["store"])
            self.assertTrue(call["text"]["format"]["strict"])
            self.assertEqual(call["text"]["format"]["type"], "json_schema")
            schema = call["text"]["format"]["schema"]
            self.assertEqual(schema["properties"]["digital_eligibility"]["enum"], ["Unknown", "Yes", "No"])
            self.assertIn("digital_eligibility", schema["required"])
            self.assertIn("digital_eligibility", schema["properties"]["facts"]["items"]["properties"]["source"]["enum"])
            self.assertEqual(json.loads(call["input"]), {"intake": request, "document_pages": uploaded["pages"]})
            self.assertNotIn("test-only-key", json.dumps(call))
            provider.return_value.__exit__.assert_called_once()

    def test_live_failure_does_not_leak_exception_or_retry(self):
        with patch("uvdemo.analysis.OpenAI") as provider:
            call = provider.return_value.__enter__.return_value.responses.create
            call.side_effect = RuntimeError("secret-key-and-private-request")
            with self.assertRaises(AnalysisError) as raised:
                analyze_request(intake(), mode="live", api_key="test-only-key")
            self.assertNotIn("secret", str(raised.exception))
            call.assert_called_once()

    def test_provider_errors_offer_safe_specific_guidance_without_retry(self):
        secret = "secret-key-and-private-document"
        request = SimpleNamespace(url=secret, content=secret)

        def status_error(error_type, status, code=None):
            response = SimpleNamespace(request=request, status_code=status, headers={})
            return error_type(secret, response=response,
                              body={"message": secret, "code": code, "param": secret})

        cases = [
            (status_error(AuthenticationError, 401), "authentication failed"),
            (status_error(PermissionDeniedError, 403), "project permissions"),
            (status_error(NotFoundError, 404), "model is unavailable"),
            (status_error(RateLimitError, 429, "insufficient_quota"), "API quota is exhausted"),
            (status_error(RateLimitError, 429, "rate_limit_exceeded"), "too many requests"),
            (status_error(RateLimitError, 429, secret), "too many requests"),
            (APIConnectionError(message=secret, request=request), "Could not connect"),
            (APITimeoutError(request=request), "timed out"),
            (status_error(InternalServerError, 503), "temporarily unavailable"),
            (status_error(BadRequestError, 400), "could not accept"),
            (status_error(UnprocessableEntityError, 422), "could not accept"),
        ]
        for error, expected in cases:
            with self.subTest(error_type=type(error).__name__, expected=expected), patch("uvdemo.analysis.OpenAI") as provider:
                call = provider.return_value.__enter__.return_value.responses.create
                call.side_effect = error
                with self.assertRaises(AnalysisError) as raised:
                    analyze_request(intake(), mode="live", api_key="test-only-key")
                self.assertIn(expected, str(raised.exception))
                self.assertNotIn(secret, str(raised.exception))
                self.assertNotIn("test-only-key", str(raised.exception))
                self.assertTrue(raised.exception.__suppress_context__)
                call.assert_called_once()
                provider.return_value.__exit__.assert_called_once()

    def test_live_rejects_incomplete_refusal_bad_json_and_wrong_schema(self):
        refusal = SimpleNamespace(content=[SimpleNamespace(type="refusal")])
        responses = [SimpleNamespace(status="incomplete", output=[], output_text=""),
                     SimpleNamespace(status="completed", output=[refusal], output_text=""),
                     SimpleNamespace(status="completed", output=[], output_text="not-json"),
                     SimpleNamespace(status="completed", output=[], output_text='{"summary": "bad"}')]
        for response in responses:
            with self.subTest(response=response), patch("uvdemo.analysis.OpenAI") as provider:
                call = provider.return_value.__enter__.return_value.responses.create
                call.return_value = response
                with self.assertRaises(AnalysisError):
                    analyze_request(intake(), mode="live", api_key="test-only-key")
                call.assert_called_once()

    def test_validation_rejects_fabricated_quote_wrong_page_source_and_bad_shapes(self):
        request, uploaded = intake(), document()
        original = analyze_request(request, uploaded)
        mutations = [lambda r: r["facts"][0].update(quote="invented evidence"),
                     lambda r: r["facts"][0].update(page=2),
                     lambda r: r["facts"][0].update(page=True),
                     lambda r: r["facts"][0].update(source="internet"),
                     lambda r: r.update(customer_segment="Small"),
                     lambda r: r.update(digital_eligibility="Maybe"),
                     lambda r: r.update(clarification_questions="not a list"),
                     lambda r: r.update(facts=[]),
                     lambda r: r.update(extra="unsupported"),
                     lambda r: r.update(summary=None)]
        for mutate in mutations:
            result = deepcopy(original)
            mutate(result)
            with self.assertRaises(AnalysisError):
                validate_analysis(result, request, uploaded)

    def test_handling_claim_alone_does_not_support_known_digital_eligibility(self):
        request = intake(customer_segment="Medium Business", handling_approach="Standard/Digital-eligible")
        result = analyze_request(request)
        result["digital_eligibility"] = "Yes"
        with self.assertRaisesRegex(AnalysisError, "supporting evidence"):
            validate_analysis(result, request)

    def test_digital_intake_evidence_must_quote_the_supplied_value(self):
        request = intake(customer_segment="Medium Business",
                         handling_approach="Assisted/Custom", digital_eligibility="No")
        result = analyze_request(request)
        digital_fact = next(fact for fact in result["facts"] if fact["source"] == "digital_eligibility")
        digital_fact["quote"] = "Yes"
        with self.assertRaisesRegex(AnalysisError, "evidence did not match"):
            validate_analysis(result, request)


if __name__ == "__main__":
    unittest.main()
