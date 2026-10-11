"""Submission, document analysis and group allocation through the Streamlit UI.

Mock mode is configured only in the test environment. Live tests stub the SDK;
this suite never sends a document to OpenAI.
"""

from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from uvdemo.analysis import analyze_request
from uvdemo.documents import extract_document
from uvdemo.group_allocation import get_group_assignment
from uvdemo.storage import (
    get_case, get_case_by_submission, get_case_history, get_document,
    get_latest_analysis, get_latest_review,
)
from test_documents import synthetic_pdf

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
GOVERNMENT_TEXT = "Contract type: Services agreement\nCustomer segment: Government"
MEDIUM_BUSINESS_TEXT = "Contract type: Services agreement\nCustomer segment: Medium Business"


class AppTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.db = Path(temporary.name) / "cases.sqlite3"
        environment = patch.dict(os.environ, {
            "WORKBENCH_DB_PATH": str(self.db), "WORKBENCH_AI_MODE": "mock",
            "OPENAI_API_KEY": "", "OPENAI_MODEL": "",
        })
        environment.start()
        self.addCleanup(environment.stop)
        self.app = AppTest.from_file(APP_PATH, default_timeout=20).run()
        self.assertFalse(self.app.exception)

    def fill_request(self, *, content=None, segment="Unknown", handling="Unknown",
                     digital="Unknown", upload=True):
        token = self.app.query_params["submission"]
        request = {"customer_name": "Fictional Acorn Services",
                   "description": "A synthetic contract request.",
                   "customer_segment": segment, "handling_approach": handling,
                   "digital_eligibility": digital}
        self.app.text_input(key=f"name_{token}").input(request["customer_name"])
        self.app.text_area(key=f"description_{token}").input(request["description"])
        self.app.selectbox(key=f"segment_{token}").set_value(segment)
        self.app.selectbox(key=f"handling_{token}").set_value(handling)
        self.app.selectbox(key=f"digital_{token}").set_value(digital)
        if upload:
            content = content if content is not None else synthetic_pdf([GOVERNMENT_TEXT])
            self.app.file_uploader[0].set_value(("synthetic.pdf", content, "application/pdf"))
        return request, content

    def submit_example(self, **kwargs):
        request, content = self.fill_request(**kwargs)
        self.app.button(key="save_request").click().run()
        self.assertFalse(self.app.exception)
        return request, content

    def completed_response(self, request, content):
        extraction = extract_document("synthetic.pdf", content)
        suggestions = analyze_request(request, extraction)
        suggestions["summary"] = "Unconfirmed synthetic contract interpretation."
        return SimpleNamespace(status="completed", output=[], output_text=json.dumps(suggestions))

    def assert_simple_submission_controls(self):
        self.assertFalse(self.app.button(key="save_request").disabled)
        self.assertEqual(self.app.button(key="save_request").label, "Submit case")
        self.assertNotIn("openai_key", [widget.key for widget in self.app.text_input])
        self.assertNotIn("setup_live_ai", [widget.key for widget in self.app.button])
        self.assertFalse(any((widget.key or "").endswith("_mode") for widget in self.app.radio))
        self.assertFalse(any((widget.key or "").endswith("_consent") for widget in self.app.checkbox))

    def test_blank_form_requires_name_description_and_document_without_saving(self):
        token = self.app.query_params["submission"]
        with patch("uvdemo.analysis.OpenAI") as provider:
            self.app.button(key="save_request").click().run()
            provider.assert_not_called()
        self.assertFalse(self.app.exception)
        self.assertEqual(len(self.app.error), 3)
        self.assertEqual(self.app.query_params["submission"], token)
        self.assertIsNone(get_case_by_submission(self.db, token))

    def test_document_is_mandatory_even_when_text_fields_are_complete(self):
        token = self.app.query_params["submission"]
        with patch("uvdemo.analysis.OpenAI") as provider:
            self.submit_example(upload=False)
            provider.assert_not_called()
        self.assertEqual(len(self.app.error), 1)
        self.assertIn("contract PDF", self.app.error[0].value)
        self.assertIsNone(get_case_by_submission(self.db, token))

    def test_pdf_triggers_one_mock_analysis_and_group_allocation_surviving_reload(self):
        token = self.app.query_params["submission"]
        with patch("uvdemo.analysis.analyze_request", wraps=analyze_request) as analysis, \
                patch("uvdemo.analysis.OpenAI") as provider:
            request, content = self.submit_example()
            analysis.assert_called_once()
            provider.assert_not_called()
            case = get_case_by_submission(self.db, token)
            case_id = case["case_id"]
            self.assertEqual(case["status"], "Allocated")
            self.assertEqual(case["original_input"], request)
            saved_document = get_document(self.db, case_id)
            self.assertEqual(saved_document["content"], content)
            self.assertEqual(saved_document["extraction"]["status"], "Ready")
            attempt = get_latest_analysis(self.db, case_id)
            assignment = get_group_assignment(self.db, case_id)
            self.assertEqual(attempt["result"]["contract_type"], "Services agreement")
            self.assertEqual(attempt["mode"], "mock")
            self.assertEqual(assignment["queue"], "Government")
            self.assertEqual(assignment["mode"], "mock")
            self.assertIsNone(get_latest_review(self.db, case_id))
            self.assertTrue(any("MOCK AI" in warning.value for warning in self.app.warning))
            self.assertTrue(any("Government" in success.value for success in self.app.success))
            before = get_case_history(self.db, case_id)
            self.app.run()
            reloaded = AppTest.from_file(APP_PATH, default_timeout=20)
            reloaded.query_params["submission"] = token
            reloaded.run()
            self.assertFalse(reloaded.exception)
            self.assertEqual(reloaded.code[0].value, case_id)
            self.assertEqual(get_case_history(self.db, case_id), before)
            self.assertEqual(get_group_assignment(self.db, case_id), assignment)
            analysis.assert_called_once()
            reloaded.radio(key="screen").set_value("Review & Allocate").run()
            self.assertNotIn("run_analysis", [button.key for button in reloaded.button])
            self.assertNotIn("confirm_review", [button.key for button in reloaded.button])
            reloaded.radio(key="screen").set_value("Submit Request").run()
            reloaded.button(key="new_request").click().run()
            self.assertNotEqual(reloaded.query_params["submission"], token)
            self.assertEqual(reloaded.text_area[0].value, "")
        with closing(sqlite3.connect(self.db)) as connection:
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertNotIn("specialists", tables)

    def test_case_id_lookup_works_in_a_new_session(self):
        self.submit_example()
        case_id = self.app.code[0].value
        fresh = AppTest.from_file(APP_PATH, default_timeout=20).run()
        fresh.text_input(key="lookup_case_id").input(case_id)
        fresh.button(key="find_request_button").click().run()
        self.assertFalse(fresh.exception)
        self.assertEqual(fresh.code[0].value, case_id)
        self.assertTrue(any("Government" in success.value for success in fresh.success))

    def test_failed_save_preserves_form_and_can_be_retried(self):
        token = self.app.query_params["submission"]
        self.fill_request()
        with patch("uvdemo.storage.save_intake", side_effect=sqlite3.OperationalError("busy")):
            self.app.button(key="save_request").click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(len(self.app.error), 1)
        self.assertEqual(self.app.text_input(key=f"name_{token}").value, "Fictional Acorn Services")
        self.assertIsNone(get_case_by_submission(self.db, token))
        self.app.button(key="save_request").click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(get_case_by_submission(self.db, token)["status"], "Allocated")

    def test_unreadable_pdf_preserves_request_and_blocks_analysis_allocation(self):
        content = b"not a PDF"
        with patch("uvdemo.analysis.OpenAI") as provider, \
                patch("uvdemo.analysis.analyze_request", wraps=analyze_request) as analysis:
            request, _ = self.submit_example(content=content)
            provider.assert_not_called()
            analysis.assert_not_called()
        case_id = self.app.code[0].value
        self.assertEqual(get_document(self.db, case_id)["content"], content)
        self.assertEqual(get_document(self.db, case_id)["extraction"]["status"], "Needs attention")
        self.assertEqual(get_case(self.db, case_id)["status"], "Document needs attention")
        self.assertEqual(get_case(self.db, case_id)["original_input"], request)
        self.assertIsNone(get_latest_analysis(self.db, case_id))
        self.assertIsNone(get_group_assignment(self.db, case_id))
        self.app.radio(key="screen").set_value("Review & Allocate").run()
        self.assertFalse(self.app.exception)
        with patch("uvdemo.analysis.OpenAI") as provider:
            self.app.button(key="run_analysis").click().run()
            provider.assert_not_called()
        self.assertEqual(get_case(self.db, case_id)["status"], "Document needs attention")

    def test_conflict_is_automatically_flagged_and_original_input_stays_unassigned(self):
        request, _ = self.submit_example(segment="Medium Business")
        case_id = self.app.code[0].value
        attempt = get_latest_analysis(self.db, case_id)
        self.assertEqual(attempt["result"]["customer_segment"], "Unknown")
        self.assertTrue(attempt["result"]["contradictions"])
        self.assertEqual(get_case(self.db, case_id)["status"], "Clarification needed")
        self.assertEqual(get_case(self.db, case_id)["original_input"], request)
        self.assertIsNone(get_group_assignment(self.db, case_id))
        self.assertTrue(any("Clarification needed" in warning.value for warning in self.app.warning))
        self.assertTrue(any("Conflicting" in warning.value for warning in self.app.warning))
        self.app.radio(key="screen").set_value("Case Tracker").run()
        self.assertFalse(self.app.exception)
        row = self.app.dataframe[0].value.iloc[0]
        self.assertEqual(row["Status"], "Clarification needed")
        self.assertEqual(row["Group"], "Not allocated")
        self.assertNotIn("Specialist", row.index)

    def test_live_submission_has_no_key_mode_or_consent_controls(self):
        with patch.dict(os.environ, {"WORKBENCH_AI_MODE": "live", "OPENAI_API_KEY": "test-only-secret"}), \
                patch("uvdemo.analysis.OpenAI") as provider:
            self.app.run()
            self.assert_simple_submission_controls()
            displayed = " ".join(widget.value for widget in self.app.text_input)
            displayed += " ".join(widget.value for widget in self.app.caption)
            self.assertNotIn("test-only-secret", displayed)
            provider.assert_not_called()

    def test_missing_server_key_saves_pdf_then_configured_retry_analyses_once(self):
        token = self.app.query_params["submission"]
        with patch.dict(os.environ, {"WORKBENCH_AI_MODE": "live", "OPENAI_API_KEY": ""}), \
                patch("uvdemo.analysis.OpenAI") as provider, \
                patch("uvdemo.analysis.analyze_request", wraps=analyze_request) as analysis:
            self.app.run()
            self.assert_simple_submission_controls()
            request, content = self.submit_example()
            case_id = get_case_by_submission(self.db, token)["case_id"]
            self.assertEqual(get_case(self.db, case_id)["status"], "Analysis needed")
            self.assertEqual(get_case(self.db, case_id)["original_input"], request)
            self.assertEqual(get_document(self.db, case_id)["content"], content)
            self.assertIsNone(get_latest_analysis(self.db, case_id))
            self.assertIsNone(get_group_assignment(self.db, case_id))
            self.app.run()
            self.app.radio(key="screen").set_value("Review & Allocate").run()
            self.app.button(key="run_analysis").click().run()
            self.assertTrue(any("needs setup" in message.value for message in self.app.info))
            analysis.assert_not_called()
            provider.assert_not_called()
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-only-secret"}):
                call = provider.return_value.__enter__.return_value.responses.create
                call.return_value = self.completed_response(request, content)
                self.app.run()
                self.assertFalse(self.app.button(key="run_analysis").disabled)
                self.app.button(key="run_analysis").click().run()
                self.assertFalse(self.app.exception)
                analysis.assert_called_once()
                call.assert_called_once()
                self.assertEqual(get_case(self.db, case_id)["status"], "Allocated")
                self.assertEqual(get_group_assignment(self.db, case_id)["mode"], "live")
                self.assertEqual(get_case(self.db, case_id)["original_input"], request)

    def test_live_is_default_without_automatic_mock_fallback(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}), \
                patch("streamlit.secrets", {}), \
                patch("uvdemo.analysis.analyze_request", wraps=analyze_request) as analysis:
            os.environ.pop("WORKBENCH_AI_MODE", None)
            self.app.run()
            self.assert_simple_submission_controls()
            self.submit_example()
            case_id = self.app.code[0].value
            self.assertEqual(get_case(self.db, case_id)["status"], "Analysis needed")
            self.assertIsNone(get_latest_analysis(self.db, case_id))
            self.assertFalse(any("MOCK AI" in warning.value for warning in self.app.warning))
            analysis.assert_not_called()

    def test_whitespace_server_key_never_blocks_submission_or_attempts_live_analysis(self):
        with patch.dict(os.environ, {"WORKBENCH_AI_MODE": "live", "OPENAI_API_KEY": "   "}), \
                patch("uvdemo.analysis.OpenAI") as provider:
            self.app.run()
            self.assert_simple_submission_controls()
            self.submit_example()
            case_id = self.app.code[0].value
            self.assertEqual(get_case(self.db, case_id)["status"], "Analysis needed")
            self.assertIsNone(get_latest_analysis(self.db, case_id))
            provider.assert_not_called()

    def test_live_submission_sends_pdf_once_allocates_and_labels_output_correctly(self):
        token = self.app.query_params["submission"]
        with patch.dict(os.environ, {"WORKBENCH_AI_MODE": "live", "OPENAI_API_KEY": "test-only-secret"}), \
                patch("uvdemo.analysis.OpenAI") as provider:
            self.app.run()
            request, content = self.fill_request()
            call = provider.return_value.__enter__.return_value.responses.create
            call.return_value = self.completed_response(request, content)
            self.app.button(key="save_request").click().run()
            self.assertFalse(self.app.exception)
            call.assert_called_once()
            payload = json.loads(call.call_args.kwargs["input"])
            self.assertEqual(payload["intake"], request)
            self.assertIn("Customer segment: Government", payload["document_pages"][0]["text"])
            self.assertEqual(call.call_args.kwargs["text"]["format"]["type"], "json_schema")
            case_id = self.app.code[0].value
            self.assertEqual(get_group_assignment(self.db, case_id)["mode"], "live")
            self.assertEqual(get_case(self.db, case_id)["status"], "Allocated")
            self.assertTrue(any("OpenAI analysis" in caption.value for caption in self.app.caption))
            self.assertFalse(any("MOCK AI" in warning.value for warning in self.app.warning))
            saved = json.dumps([get_latest_analysis(self.db, case_id), get_case_history(self.db, case_id)])
            self.assertNotIn("test-only-secret", saved)
            self.app.run()
            fresh = AppTest.from_file(APP_PATH, default_timeout=20)
            fresh.query_params["submission"] = token
            fresh.run()
            self.assertFalse(fresh.exception)
            call.assert_called_once()

    def test_live_failure_is_saved_private_and_retries_only_when_explicitly_requested(self):
        with patch.dict(os.environ, {"WORKBENCH_AI_MODE": "live", "OPENAI_API_KEY": "test-only-secret"}), \
                patch("uvdemo.analysis.OpenAI") as provider:
            self.app.run()
            request, content = self.fill_request()
            response = self.completed_response(request, content)
            call = provider.return_value.__enter__.return_value.responses.create
            call.side_effect = RuntimeError("test-only-secret provider details")
            self.app.button(key="save_request").click().run()
            self.assertFalse(self.app.exception)
            call.assert_called_once()
            case_id = self.app.code[0].value
            attempt = get_latest_analysis(self.db, case_id)
            self.assertTrue(attempt["error"])
            self.assertIsNone(attempt["result"])
            self.assertEqual(get_case(self.db, case_id)["status"], "Analysis failed")
            self.assertEqual(get_case(self.db, case_id)["original_input"], request)
            self.assertEqual(get_document(self.db, case_id)["content"], content)
            self.assertIsNone(get_group_assignment(self.db, case_id))
            saved = json.dumps([attempt, get_case_history(self.db, case_id)])
            self.assertNotIn("test-only-secret", saved)
            self.assertNotIn("provider details", saved)
            self.assertTrue(all("test-only-secret" not in error.value for error in self.app.error))
            self.app.run()
            self.app.radio(key="screen").set_value("Review & Allocate").run()
            call.assert_called_once()
            call.side_effect = None
            call.return_value = response
            self.app.button(key="run_analysis").click().run()
            self.assertFalse(self.app.exception)
            self.assertEqual(call.call_count, 2)
            self.assertEqual(get_case(self.db, case_id)["status"], "Allocated")
            self.assertNotEqual(get_latest_analysis(self.db, case_id)["analysis_id"], attempt["analysis_id"])
            self.assertEqual(get_case(self.db, case_id)["original_input"], request)

    def test_medium_business_unknown_digital_eligibility_remains_unassigned(self):
        self.submit_example(content=synthetic_pdf([MEDIUM_BUSINESS_TEXT]),
                            segment="Medium Business", handling="Standard/Digital-eligible")
        case_id = self.app.code[0].value
        self.assertEqual(get_case(self.db, case_id)["status"], "Clarification needed")
        self.assertEqual(get_latest_analysis(self.db, case_id)["result"]["digital_eligibility"], "Unknown")
        self.assertIsNone(get_group_assignment(self.db, case_id))

    def test_medium_business_explicit_digital_eligibility_routes_to_digital_group(self):
        request, _ = self.submit_example(content=synthetic_pdf([MEDIUM_BUSINESS_TEXT]),
                                         segment="Medium Business", handling="Standard/Digital-eligible",
                                         digital="Yes")
        case_id = self.app.code[0].value
        self.assertEqual(get_case(self.db, case_id)["status"], "Allocated")
        self.assertEqual(get_case(self.db, case_id)["original_input"], request)
        self.assertEqual(get_group_assignment(self.db, case_id)["queue"], "Digital")

    def test_allocation_retry_uses_saved_analysis_after_database_failure(self):
        token = self.app.query_params["submission"]
        with patch("uvdemo.analysis.analyze_request", wraps=analyze_request) as analysis, \
                patch("uvdemo.analysis.OpenAI") as provider:
            with patch("uvdemo.group_allocation.allocate_group", side_effect=sqlite3.OperationalError("busy")):
                self.submit_example()
            case_id = get_case_by_submission(self.db, token)["case_id"]
            attempt = get_latest_analysis(self.db, case_id)
            self.assertIsNotNone(attempt)
            self.assertIsNone(get_group_assignment(self.db, case_id))
            pending_case = get_case(self.db, case_id)
            pending_history = get_case_history(self.db, case_id)
            with patch.dict(os.environ, {"WORKBENCH_AI_MODE": "live", "OPENAI_API_KEY": ""}):
                self.app.radio(key="screen").set_value("Review & Allocate").run()
                self.app.button(key="run_analysis").click().run()
                self.assertFalse(self.app.exception)
                self.assertTrue(any("needs setup" in message.value for message in self.app.info))
                self.assertIsNone(get_group_assignment(self.db, case_id))
                self.assertEqual(get_case(self.db, case_id), pending_case)
                self.assertEqual(get_case_history(self.db, case_id), pending_history)
                self.assertEqual(get_latest_analysis(self.db, case_id), attempt)
                analysis.assert_called_once()
                provider.assert_not_called()
                self.app.button(key="retry_allocation").click().run()
            self.assertFalse(self.app.exception)
            self.assertEqual(get_case(self.db, case_id)["status"], "Allocated")
            self.assertEqual(get_group_assignment(self.db, case_id)["queue"], "Government")
            self.assertEqual(get_latest_analysis(self.db, case_id), attempt)
            analysis.assert_called_once()
            provider.assert_not_called()

    def test_tracker_progress_completes_and_reopens_without_changing_allocation_or_ai_evidence(self):
        self.submit_example()
        case_id = self.app.code[0].value
        original = get_case(self.db, case_id)["original_input"]
        assignment = get_group_assignment(self.db, case_id)
        attempt = get_latest_analysis(self.db, case_id)
        document = get_document(self.db, case_id)
        initial_history = get_case_history(self.db, case_id)

        with patch("uvdemo.analysis.analyze_request") as analysis, \
                patch("uvdemo.analysis.OpenAI") as provider:
            self.app.radio(key="screen").set_value("Case Tracker").run()
            transitions = [("Allocated", "Start work", "In progress"),
                           ("In progress", "Mark completed", "Completed"),
                           ("Completed", "Reopen case", "In progress")]
            for index, (current, label, target) in enumerate(transitions, start=1):
                with self.subTest(target=target, transition=index):
                    button = self.app.button(key=f"advance_case_{case_id}_{current}")
                    self.assertEqual(button.label, label)
                    button.click().run()
                    self.assertFalse(self.app.exception)
                    self.assertEqual(get_case(self.db, case_id)["status"], target)
                    self.assertEqual(get_case(self.db, case_id)["original_input"], original)
                    self.assertEqual(get_group_assignment(self.db, case_id), assignment)
                    self.assertEqual(get_latest_analysis(self.db, case_id), attempt)
                    self.assertEqual(get_document(self.db, case_id), document)
                    history = get_case_history(self.db, case_id)
                    self.assertEqual(history[:len(initial_history)], initial_history)
                    updates = [event for event in history
                               if event["event_type"] == "Case progress updated"]
                    self.assertEqual(len(updates), index)
                    self.app.run()
                    self.assertEqual(get_case_history(self.db, case_id), history)

                    # A new browser session reads progress from SQLite, not widget state.
                    self.app = AppTest.from_file(APP_PATH, default_timeout=20).run()
                    self.app.radio(key="screen").set_value("Case Tracker").run()
                    self.assertFalse(self.app.exception)
                    row = self.app.dataframe[0].value.iloc[0]
                    self.assertEqual(row["Status"], target)
                    self.assertEqual(row["Group"], "Government")
                    self.assertTrue(any("Government" in message.value
                                        for message in self.app.success))
                    self.assertEqual(get_case_history(self.db, case_id), history)
            self.app.radio(key="screen").set_value("Review & Allocate").run()
            self.assertNotIn("run_analysis", [button.key for button in self.app.button])
            self.assertNotIn("retry_allocation", [button.key for button in self.app.button])
            analysis.assert_not_called()
            provider.assert_not_called()

    def test_tracker_combines_search_status_and_group_filters_and_handles_empty_results(self):
        saved = {}
        examples = [("Alpha Government", GOVERNMENT_TEXT, "Unknown"),
                    ("Beta Enterprise", "Contract type: Services agreement\nCustomer segment: Enterprise", "Unknown"),
                    ("Alpha Conflict", GOVERNMENT_TEXT, "Medium Business")]
        for name, text, segment in examples:
            self.fill_request(content=synthetic_pdf([text]), segment=segment)
            token = self.app.query_params["submission"]
            self.app.text_input(key=f"name_{token}").input(name)
            self.app.button(key="save_request").click().run()
            self.assertFalse(self.app.exception)
            saved[name] = self.app.code[0].value
            if name != examples[-1][0]:
                self.app.button(key="new_request").click().run()

        self.app.radio(key="screen").set_value("Case Tracker").run()
        self.assertEqual(len(self.app.dataframe[0].value), 3)
        self.app.text_input(key="tracker_search").input("ALPHA").run()
        self.assertEqual(set(self.app.dataframe[0].value["Customer"]),
                         {"Alpha Government", "Alpha Conflict"})
        self.assertEqual(len(self.app.selectbox(key="tracker_case").options), 2)
        self.app.selectbox(key="tracker_group").set_value("Government").run()
        self.assertEqual(list(self.app.dataframe[0].value["Case ID"]),
                         [saved["Alpha Government"]])
        self.assertEqual(self.app.selectbox(key="tracker_case").value, saved["Alpha Government"])
        self.assertEqual(len(self.app.selectbox(key="tracker_case").options), 1)

        self.app.selectbox(key="tracker_status").set_value("Clarification needed").run()
        self.assertFalse(self.app.exception)
        self.assertFalse(self.app.dataframe)
        self.assertNotIn("tracker_case", [widget.key for widget in self.app.selectbox])
        self.assertTrue(any(message.value == "No cases match these filters."
                            for message in self.app.info))
        self.assertFalse(any((button.key or "").startswith("advance_case_")
                             for button in self.app.button))

        self.app.selectbox(key="tracker_group").set_value("Not allocated").run()
        self.assertEqual(list(self.app.dataframe[0].value["Case ID"]), [saved["Alpha Conflict"]])
        self.assertEqual(self.app.selectbox(key="tracker_case").value, saved["Alpha Conflict"])
        self.assertFalse(any((button.key or "").startswith("advance_case_")
                             for button in self.app.button))
        self.app.selectbox(key="tracker_status").set_value("All")
        self.app.selectbox(key="tracker_group").set_value("All")
        self.app.text_input(key="tracker_search").input(saved["Beta Enterprise"].upper()).run()
        self.assertFalse(self.app.exception)
        self.assertEqual(list(self.app.dataframe[0].value["Case ID"]), [saved["Beta Enterprise"]])
        self.assertEqual(self.app.selectbox(key="tracker_case").value, saved["Beta Enterprise"])

    def test_tracker_does_not_offer_progress_actions_for_a_document_needing_attention(self):
        self.submit_example(content=b"not a PDF")
        case_id = self.app.code[0].value
        history = get_case_history(self.db, case_id)
        with patch("uvdemo.analysis.analyze_request") as analysis, \
                patch("uvdemo.analysis.OpenAI") as provider:
            self.app.radio(key="screen").set_value("Case Tracker").run()
            self.assertFalse(self.app.exception)
            self.assertEqual(self.app.selectbox(key="tracker_case").value, case_id)
            row = self.app.dataframe[0].value.iloc[0]
            self.assertEqual(row["Status"], "Document needs attention")
            self.assertEqual(row["Group"], "Not allocated")
            self.assertFalse(any((button.key or "").startswith("advance_case_")
                                 for button in self.app.button))
            self.assertEqual(get_case_history(self.db, case_id), history)
            analysis.assert_not_called()
            provider.assert_not_called()


if __name__ == "__main__":
    unittest.main()
