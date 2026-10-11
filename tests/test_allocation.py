"""Demo allocation tests: plain Python rules and isolated temporary SQLite files."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from threading import Barrier
import unittest

from uvdemo.allocation import (
    allocate_case, get_allocation_decision, get_assignment, initialize_allocation,
    list_specialists, route_analysis, select_specialist,
)
from uvdemo.storage import (
    get_case, get_case_history, get_document, initialize_database, save_analysis,
    save_document_extraction, save_intake, save_review,
)


def sample_result(segment="Government", handling="Unknown", digital="Unknown"):
    return {"summary": "Synthetic request", "contract_type": "Unknown",
            "customer_segment": segment, "handling_approach": handling,
            "digital_eligibility": digital,
            "facts": [{"fact": "The document identifies the customer as government.",
                       "source": "document", "page": 1,
                       "quote": "The contracting customer is a fictional government department."}],
            "contradictions": [],
            "clarification_questions": []}


class RoutingTests(unittest.TestCase):
    def test_only_the_four_configured_queue_rules_are_used(self):
        for segment, handling, digital, expected in (
            ("Government", "Unknown", "Unknown", "Government"),
            ("Enterprise", "Unknown", "Unknown", "Enterprise"),
            ("Medium Business", "Standard/Digital-eligible", "Yes", "Digital"),
            ("Medium Business", "Assisted/Custom", "No", "Medium Business"),
        ):
            with self.subTest(segment=segment, handling=handling):
                self.assertEqual(route_analysis(sample_result(segment, handling, digital))["queue"], expected)

    def test_missing_conflicting_or_unresolved_required_facts_block(self):
        invalid = [None, {}, sample_result("Unknown"), sample_result("Medium Business"),
                   sample_result("Medium Business", "Assisted/Custom", "Yes"),
                   sample_result("Medium Business", "Standard/Digital-eligible", "No"),
                   sample_result("Government", "Assisted/Custom", "Yes"),
                   sample_result("Enterprise", "Standard/Digital-eligible", "No"),
                   {**sample_result(), "contradictions": ["The customer segments disagree."]},
                   {**sample_result(), "clarification_questions": ["Which customer is contracting?"]},
                   {**sample_result(), "contradictions": None}]
        for result in invalid:
            with self.subTest(result=result):
                self.assertIsNone(route_analysis(result)["queue"])

    def test_selection_filters_before_workload_and_uses_stable_id_tie_break(self):
        def person(identity, available=True, capacity=3, count=0, queues=("Government",)):
            return {"specialist_id": identity, "available": available, "capacity": capacity,
                    "active_cases": count, "qualifications": queues}
        people = [person("00-unqualified", queues=("Digital",)), person("01-unavailable", False),
                  person("02-full", capacity=0), person("04", count=1), person("03", count=1)]
        self.assertEqual(select_specialist("Government", people)["specialist_id"], "03")
        people[-1]["active_cases"] = 2
        self.assertEqual(select_specialist("Government", people)["specialist_id"], "04")
        self.assertIsNone(select_specialist("Enterprise", people))


class AllocationStorageTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.db = Path(temporary.name) / "allocation.sqlite3"
        initialize_database(self.db)
        initialize_allocation(self.db)
        self.case_number = 0

    def execute(self, sql, parameters=()):
        with closing(sqlite3.connect(self.db)) as connection:
            with connection:
                return connection.execute(sql, parameters).fetchall()

    def create_case(self, *, ready=True, result=True):
        self.case_number += 1
        case = save_intake(self.db, f"submission-{self.case_number}",
                           {"customer_name": "Fictional Harbour Department",
                            "description": "A synthetic contract intake.",
                            "customer_segment": "Unknown", "handling_approach": "Unknown"},
                           {"filename": "synthetic.pdf", "content": b"%PDF-1.4 synthetic storage fixture"})
        if ready:
            document = get_document(self.db, case["case_id"])
            text = "The contracting customer is a fictional government department."
            save_document_extraction(self.db, case["case_id"],
                                     {"filename": document["filename"], "sha256": document["sha256"],
                                      "size_bytes": document["size_bytes"], "status": "Ready",
                                      "text": text, "pages": [{"page": 1, "text": text}], "error": None})
        analysis = save_analysis(self.db, case["case_id"], "mock", "development-mock",
                                 sample_result() if result is True else result, None) if result else None
        return case["case_id"], analysis["analysis_id"] if analysis else None

    def test_assignment_preserves_original_and_records_ai_provenance(self):
        case_id, analysis_id = self.create_case()
        result = allocate_case(self.db, case_id, analysis_id)
        self.assertEqual(result["status"], "Assigned")
        self.assertEqual(result["queue"], "Government")
        self.assertEqual(result["specialist_id"], "DEMO-01")
        self.assertIn("no reviewer confirmation", result["basis"])
        self.assertEqual(result["mode"], "mock")
        self.assertEqual(get_case(self.db, case_id)["original_input"]["customer_segment"], "Unknown")
        self.assertEqual(get_case(self.db, case_id)["status"], "Assigned")
        self.assertEqual(get_assignment(self.db, case_id), result)
        self.assertEqual(get_allocation_decision(self.db, case_id), result)
        self.assertEqual(get_case_history(self.db, case_id)[-1]["event_type"], "Automatically assigned")

    def test_duplicate_assignment_does_not_repeat_history_or_workload(self):
        case_id, analysis_id = self.create_case()
        first = allocate_case(self.db, case_id, analysis_id)
        history = get_case_history(self.db, case_id)
        self.assertEqual(allocate_case(self.db, case_id, analysis_id), first)
        # Existing assignments are never reassigned, even with stale UI parameters.
        self.assertEqual(allocate_case(self.db, case_id, None), first)
        self.assertEqual(get_case_history(self.db, case_id), history)
        self.assertEqual(list_specialists(self.db)[0]["active_cases"], 1)

    def test_fewest_active_saved_cases_wins_and_closed_case_releases_capacity(self):
        first_id, first_analysis = self.create_case()
        second_id, second_analysis = self.create_case()
        self.assertEqual(allocate_case(self.db, first_id, first_analysis)["specialist_id"], "DEMO-01")
        self.assertEqual(allocate_case(self.db, second_id, second_analysis)["specialist_id"], "DEMO-02")
        self.execute("UPDATE cases SET status = 'Closed' WHERE case_id = ?", (first_id,))
        third_id, third_analysis = self.create_case()
        self.assertEqual(allocate_case(self.db, third_id, third_analysis)["specialist_id"], "DEMO-01")
        allocate_case(self.db, first_id, first_analysis)
        self.assertEqual(get_case(self.db, first_id)["status"], "Closed")

    def test_no_capacity_stays_unassigned_then_can_retry_same_analysis(self):
        self.execute("UPDATE specialists SET capacity = 0")
        case_id, analysis_id = self.create_case()
        first = allocate_case(self.db, case_id, analysis_id)
        self.assertEqual(first["status"], "Awaiting capacity")
        self.assertIsNone(get_assignment(self.db, case_id))
        history = get_case_history(self.db, case_id)
        self.assertEqual(allocate_case(self.db, case_id, analysis_id), first)
        self.assertEqual(get_case_history(self.db, case_id), history)
        self.execute("UPDATE specialists SET capacity = 1 WHERE specialist_id = 'DEMO-02'")
        result = allocate_case(self.db, case_id, analysis_id)
        self.assertEqual(result["specialist_id"], "DEMO-02")
        self.assertEqual(result["status"], "Assigned")

    def test_availability_and_qualification_are_rechecked_at_assignment(self):
        case_id, analysis_id = self.create_case()
        self.execute("UPDATE specialists SET available = 0 WHERE specialist_id = 'DEMO-01'")
        self.execute("DELETE FROM specialist_qualifications WHERE specialist_id = 'DEMO-02'")
        result = allocate_case(self.db, case_id, analysis_id)
        self.assertEqual(result["status"], "Awaiting capacity")
        self.assertIsNone(result["specialist_id"])

    def test_concurrent_cases_cannot_take_same_last_capacity_slot(self):
        self.execute("UPDATE specialists SET capacity = 0")
        self.execute("UPDATE specialists SET capacity = 1 WHERE specialist_id = 'DEMO-01'")
        cases = [self.create_case(), self.create_case()]
        barrier = Barrier(2)

        def allocate(pair):
            barrier.wait()
            return allocate_case(self.db, *pair)

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(allocate, cases))
        self.assertCountEqual([result["status"] for result in results], ["Assigned", "Awaiting capacity"])
        self.assertEqual(list_specialists(self.db)[0]["active_cases"], 1)
        self.assertEqual(self.execute("SELECT COUNT(*) FROM case_assignments")[0][0], 1)

    def test_concurrent_retries_create_only_one_assignment_event(self):
        case_id, analysis_id = self.create_case()
        barrier = Barrier(2)

        def allocate(_):
            barrier.wait()
            return allocate_case(self.db, case_id, analysis_id)

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(allocate, range(2)))
        self.assertEqual(results[0], results[1])
        events = [event for event in get_case_history(self.db, case_id)
                  if event["event_type"] == "Automatically assigned"]
        self.assertEqual(len(events), 1)

    def test_stale_analysis_is_rejected_and_current_failed_attempt_blocks(self):
        case_id, analysis_id = self.create_case()
        failed = save_analysis(self.db, case_id, "live", "test-model", None, "Analysis failed.")
        history = get_case_history(self.db, case_id)
        with self.assertRaisesRegex(ValueError, "analysis changed"):
            allocate_case(self.db, case_id, analysis_id)
        self.assertEqual(get_case_history(self.db, case_id), history)
        result = allocate_case(self.db, case_id, failed["analysis_id"])
        self.assertEqual(result["status"], "Analysis failed")
        self.assertIsNone(get_assignment(self.db, case_id))

    def test_missing_unreadable_or_empty_extraction_blocks_allocation(self):
        for ready_value in (None, {"status": "Needs attention", "text": "", "pages": []},
                            {"status": "Ready", "text": " ", "pages": [{"page": 1, "text": " "}]}):
            with self.subTest(extraction=ready_value):
                case_id, analysis_id = self.create_case(ready=False)
                if ready_value:
                    document = get_document(self.db, case_id)
                    save_document_extraction(self.db, case_id,
                                             {"filename": document["filename"], "sha256": document["sha256"],
                                              "size_bytes": document["size_bytes"], "error": None, **ready_value})
                self.assertEqual(allocate_case(self.db, case_id, analysis_id)["status"], "Document needs attention")
                self.assertIsNone(get_assignment(self.db, case_id))

    def test_absent_document_in_legacy_case_and_absent_analysis_have_clear_statuses(self):
        case_id, _ = self.create_case(result=False)
        self.assertEqual(allocate_case(self.db, case_id, None)["status"], "Analysis needed")
        self.execute("DELETE FROM case_documents WHERE case_id = ?", (case_id,))
        self.assertEqual(allocate_case(self.db, case_id, None)["status"], "Document needs attention")

    def test_conflict_is_preserved_and_deferred_event_is_idempotent(self):
        conflict = {**sample_result(), "contradictions": ["Dropdown and document disagree."]}
        case_id, analysis_id = self.create_case(result=conflict)
        first = allocate_case(self.db, case_id, analysis_id)
        self.assertEqual(first["status"], "Clarification needed")
        history = get_case_history(self.db, case_id)
        self.assertEqual(allocate_case(self.db, case_id, analysis_id), first)
        self.assertEqual(get_case_history(self.db, case_id), history)
        self.assertIsNone(get_assignment(self.db, case_id))

    def test_legacy_schema_and_mismatched_evidence_cannot_allocate(self):
        legacy = sample_result()
        del legacy["digital_eligibility"]
        wrong_quote = sample_result()
        wrong_quote["facts"][0]["quote"] = "Invented text absent from the uploaded document."
        missing_evidence = {**sample_result(), "facts": []}
        for invalid in (legacy, wrong_quote, missing_evidence):
            with self.subTest(result=invalid):
                case_id, analysis_id = self.create_case(result=invalid)
                result = allocate_case(self.db, case_id, analysis_id)
                self.assertEqual(result["status"], "Analysis failed")
                self.assertIsNone(get_assignment(self.db, case_id))

    def test_interrupted_reanalysis_hides_old_decision_without_losing_history(self):
        self.execute("UPDATE specialists SET capacity = 0")
        case_id, analysis_id = self.create_case()
        old_decision = allocate_case(self.db, case_id, analysis_id)
        self.assertEqual(old_decision["status"], "Awaiting capacity")
        self.assertEqual(get_allocation_decision(self.db, case_id), old_decision)
        latest = save_analysis(self.db, case_id, "mock", "development-mock", sample_result(), None)
        # Simulate interruption after analysis is saved, before allocation runs.
        self.assertEqual(get_case(self.db, case_id)["status"], "Allocation pending")
        self.assertIsNone(get_allocation_decision(self.db, case_id))
        self.assertTrue(any(event["details"] == old_decision
                            for event in get_case_history(self.db, case_id)))
        current = allocate_case(self.db, case_id, latest["analysis_id"])
        self.assertEqual(get_allocation_decision(self.db, case_id), current)
        self.assertEqual(current["analysis_id"], latest["analysis_id"])

    def test_updated_document_hides_old_decision_until_allocation_runs(self):
        case_id, _ = self.create_case(ready=False, result=False)
        old_decision = allocate_case(self.db, case_id, None)
        self.assertEqual(old_decision["status"], "Document needs attention")
        document = get_document(self.db, case_id)
        text = "A readable synthetic contract."
        save_document_extraction(self.db, case_id,
                                 {"filename": document["filename"], "sha256": document["sha256"],
                                  "size_bytes": document["size_bytes"], "status": "Ready",
                                  "text": text, "pages": [{"page": 1, "text": text}], "error": None})
        self.assertEqual(get_case(self.db, case_id)["status"], "Analysis needed")
        self.assertIsNone(get_allocation_decision(self.db, case_id))
        self.assertTrue(any(event["details"] == old_decision
                            for event in get_case_history(self.db, case_id)))

    def test_assignment_and_status_roll_back_if_history_write_fails(self):
        case_id, analysis_id = self.create_case()
        self.execute("""CREATE TRIGGER fail_history BEFORE INSERT ON case_events
                     BEGIN SELECT RAISE(ABORT, 'Simulated history failure'); END""")
        with self.assertRaises(sqlite3.IntegrityError):
            allocate_case(self.db, case_id, analysis_id)
        self.assertIsNone(get_assignment(self.db, case_id))
        self.assertIsNone(get_allocation_decision(self.db, case_id))
        self.assertEqual(get_case(self.db, case_id)["status"], "Allocation pending")
        self.assertEqual(list_specialists(self.db)[0]["active_cases"], 0)

    def test_initializing_again_preserves_staff_configuration(self):
        self.execute("UPDATE specialists SET available = 0, capacity = 9 WHERE specialist_id = 'DEMO-01'")
        self.execute("DELETE FROM specialist_qualifications WHERE specialist_id = 'DEMO-01'")
        initialize_allocation(self.db)
        person = list_specialists(self.db)[0]
        self.assertFalse(person["available"])
        self.assertEqual(person["capacity"], 9)
        self.assertEqual(person["qualifications"], [])
        self.assertEqual(len(list_specialists(self.db)), 5)

    def test_assigned_case_rejects_new_analysis_and_review(self):
        case_id, analysis_id = self.create_case()
        allocate_case(self.db, case_id, analysis_id)
        history = get_case_history(self.db, case_id)
        with self.assertRaises(ValueError):
            save_analysis(self.db, case_id, "mock", "development-mock", sample_result(), None)
        with self.assertRaises(ValueError):
            save_review(self.db, case_id, {}, analysis_id)
        self.assertEqual(get_case(self.db, case_id)["status"], "Assigned")
        self.assertEqual(get_case_history(self.db, case_id), history)


if __name__ == "__main__":
    unittest.main()
