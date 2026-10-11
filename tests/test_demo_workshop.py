"""Rehearse the workshop fixtures through the real Streamlit submission flow.

These are offline mock-AI checks, not evidence of a live model's behaviour.
They use the same PDFs and intake values that the demonstration guide uses.
"""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from uvdemo.analysis import analyze_request
from uvdemo.group_allocation import get_group_assignment
from uvdemo.storage import (
    get_case, get_case_by_submission, get_case_history, get_document,
    get_latest_analysis,
)


PROJECT_PATH = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_PATH / "app.py"


class DemoWorkshopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scenarios = json.loads((PROJECT_PATH / "demo" / "scenarios.json").read_text(
            encoding="utf-8"
        ))
        cls.scenarios = {scenario["id"]: scenario for scenario in scenarios}

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.db = Path(temporary.name) / "workshop.sqlite3"
        environment = patch.dict(os.environ, {
            "WORKBENCH_DB_PATH": str(self.db), "WORKBENCH_AI_MODE": "mock",
            "OPENAI_API_KEY": "", "OPENAI_MODEL": "",
        })
        environment.start()
        self.addCleanup(environment.stop)
        provider = patch("uvdemo.analysis.OpenAI", side_effect=AssertionError(
            "The workshop rehearsal must never call OpenAI."
        ))
        self.provider = provider.start()
        self.addCleanup(provider.stop)
        analysis = patch("uvdemo.analysis.analyze_request", wraps=analyze_request)
        self.analysis = analysis.start()
        self.addCleanup(analysis.stop)
        self.app = AppTest.from_file(APP_PATH, default_timeout=20).run()
        self.assertFalse(self.app.exception)

    def submit_scenario(self, scenario_id):
        scenario = self.scenarios[scenario_id]
        intake = scenario["intake"]
        pdf = PROJECT_PATH / scenario["pdf"]
        content = pdf.read_bytes()
        token = self.app.query_params["submission"]
        self.app.text_input(key=f"name_{token}").input(intake["customer_name"])
        self.app.text_area(key=f"description_{token}").input(intake["description"])
        self.app.selectbox(key=f"segment_{token}").set_value(intake["customer_segment"])
        self.app.selectbox(key=f"handling_{token}").set_value(intake["handling_approach"])
        self.app.selectbox(key=f"digital_{token}").set_value(intake["digital_eligibility"])
        self.app.file_uploader[0].set_value((pdf.name, content, "application/pdf"))
        self.app.button(key="save_request").click().run()

        self.assertFalse(self.app.exception)
        self.analysis.assert_called_once()
        self.provider.assert_not_called()
        case = get_case_by_submission(self.db, token)
        self.assertIsNotNone(case)
        case_id = case["case_id"]
        self.assertEqual(case["original_input"], intake)
        self.assertEqual(case["status"], scenario["expected_status"])
        document = get_document(self.db, case_id)
        self.assertEqual(document["filename"], pdf.name)
        self.assertEqual(document["content"], content)
        self.assertEqual(document["extraction"]["status"], "Ready")
        attempt = get_latest_analysis(self.db, case_id)
        self.assertEqual(attempt["mode"], "mock")
        self.assertIsNone(attempt["error"])
        assignment = get_group_assignment(self.db, case_id)
        if scenario["expected_group"]:
            self.assertEqual(assignment["queue"], scenario["expected_group"])
        else:
            self.assertIsNone(assignment)
        self.assertTrue(any("MOCK AI" in warning.value for warning in self.app.warning))
        return case_id

    def assert_blocked_tracker(self, case_id):
        history = get_case_history(self.db, case_id)
        self.app.radio(key="screen").set_value("Case Tracker").run()
        self.assertFalse(self.app.exception)
        row = self.app.dataframe[0].value.iloc[0]
        self.assertEqual(row["Status"], "Clarification needed")
        self.assertEqual(row["Group"], "Not allocated")
        self.assertFalse(any((button.key or "").startswith("advance_case_")
                             for button in self.app.button))
        self.app.run()
        self.assertEqual(get_case_history(self.db, case_id), history)
        self.analysis.assert_called_once()
        self.provider.assert_not_called()

    def test_government_fixture_runs_from_upload_to_completion_and_persists(self):
        case_id = self.submit_scenario("government-match")
        original = get_case(self.db, case_id)["original_input"]
        document = get_document(self.db, case_id)
        attempt = get_latest_analysis(self.db, case_id)
        assignment = get_group_assignment(self.db, case_id)
        before = get_case_history(self.db, case_id)

        self.app.radio(key="screen").set_value("Case Tracker").run()
        for current, label, target in [
            ("Allocated", "Start work", "In progress"),
            ("In progress", "Mark completed", "Completed"),
        ]:
            button = self.app.button(key=f"advance_case_{case_id}_{current}")
            self.assertEqual(button.label, label)
            button.click().run()
            self.assertFalse(self.app.exception)
            self.assertEqual(get_case(self.db, case_id)["status"], target)

        history = get_case_history(self.db, case_id)
        self.assertEqual(history[:len(before)], before)
        progress = [event["details"] for event in history
                    if event["event_type"] == "Case progress updated"]
        self.assertEqual([(event["previous_status"], event["status"])
                          for event in progress],
                         [("Allocated", "In progress"), ("In progress", "Completed")])
        self.assertTrue(all(event["queue"] == "Government" for event in progress))
        self.app.run()
        fresh = AppTest.from_file(APP_PATH, default_timeout=20).run()
        fresh.radio(key="screen").set_value("Case Tracker").run()
        self.assertFalse(fresh.exception)
        row = fresh.dataframe[0].value.iloc[0]
        self.assertEqual(row["Case ID"], case_id)
        self.assertEqual(row["Status"], "Completed")
        self.assertEqual(row["Group"], "Government")
        self.assertEqual(get_case(self.db, case_id)["original_input"], original)
        self.assertEqual(get_document(self.db, case_id), document)
        self.assertEqual(get_latest_analysis(self.db, case_id), attempt)
        self.assertEqual(get_group_assignment(self.db, case_id), assignment)
        self.assertEqual(get_case_history(self.db, case_id), history)
        self.analysis.assert_called_once()
        self.provider.assert_not_called()

    def test_conflicting_segment_fixture_stays_unallocated(self):
        case_id = self.submit_scenario("segment-conflict")
        result = get_latest_analysis(self.db, case_id)["result"]
        self.assertEqual(result["customer_segment"], "Unknown")
        self.assertTrue(result["contradictions"])
        self.assert_blocked_tracker(case_id)

    def test_missing_digital_fixture_stays_unallocated(self):
        case_id = self.submit_scenario("missing-eligibility")
        result = get_latest_analysis(self.db, case_id)["result"]
        self.assertEqual(result["customer_segment"], "Medium Business")
        self.assertEqual(result["digital_eligibility"], "Unknown")
        self.assertTrue(result["clarification_questions"])
        self.assert_blocked_tracker(case_id)


if __name__ == "__main__":
    unittest.main()
