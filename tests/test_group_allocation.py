"""Group decisions use saved AI evidence and never require specialist capacity."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import copy
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from threading import Barrier
import unittest

from uvdemo.allocation import allocate_case, get_assignment, initialize_allocation
from uvdemo.group_allocation import (
    allocate_group, get_group_assignment, get_group_decision, get_legacy_assignment,
    initialize_group_allocation,
)
from uvdemo.storage import (
    get_case, get_case_history, get_document, initialize_database, save_analysis,
    save_document_extraction, save_intake, save_review,
)


class GroupAllocationTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.db = Path(temporary.name) / "groups.sqlite3"
        initialize_database(self.db)
        initialize_group_allocation(self.db)
        self.case_number = 0

    def execute(self, sql, parameters=()):
        with closing(sqlite3.connect(self.db)) as connection:
            with connection:
                return connection.execute(sql, parameters).fetchall()

    def create_case(self, segment="Government", handling="Unknown", digital="Unknown",
                    *, ready=True, analyse=True, mode="live"):
        self.case_number += 1
        case = save_intake(self.db, f"submission-{self.case_number}",
                           {"customer_name": "Fictional Harbour Customer",
                            "description": "A synthetic contract request.",
                            "customer_segment": "Unknown", "handling_approach": "Unknown"},
                           {"filename": "synthetic.pdf", "content": b"%PDF-1.4 synthetic storage fixture"})
        case_id = case["case_id"]
        text = (f"Customer segment: {segment}\nHandling approach: {handling}\n"
                f"Digital eligibility: {digital}")
        if ready:
            document = get_document(self.db, case_id)
            save_document_extraction(self.db, case_id,
                                     {"filename": document["filename"], "sha256": document["sha256"],
                                      "size_bytes": document["size_bytes"], "status": "Ready",
                                      "text": text, "pages": [{"page": 1, "text": text}], "error": None})
        result = {"summary": "Synthetic contract request.", "contract_type": "Unknown",
                  "customer_segment": segment, "handling_approach": handling,
                  "digital_eligibility": digital,
                  "facts": [{"fact": "The document reports the routing facts.", "source": "document",
                             "page": 1, "quote": text}],
                  "contradictions": [], "clarification_questions": []}
        analysis = save_analysis(self.db, case_id, mode, "test-model", result, None) if analyse else None
        return case_id, analysis["analysis_id"] if analysis else None, result

    def test_all_four_groups_route_without_creating_specialist_tables(self):
        for segment, handling, digital, expected in (
            ("Government", "Unknown", "Unknown", "Government"),
            ("Enterprise", "Unknown", "Unknown", "Enterprise"),
            ("Medium Business", "Standard/Digital-eligible", "Yes", "Digital"),
            ("Medium Business", "Assisted/Custom", "No", "Medium Business"),
        ):
            with self.subTest(segment=segment, handling=handling):
                case_id, analysis_id, _ = self.create_case(segment, handling, digital)
                decision = allocate_group(self.db, case_id, analysis_id)
                self.assertEqual(decision["queue"], expected)
                self.assertEqual(decision["status"], "Allocated")
                self.assertEqual(get_group_assignment(self.db, case_id), decision)
                self.assertEqual(get_group_decision(self.db, case_id), decision)
                self.assertNotIn("specialist_id", decision)
        self.assertEqual(self.execute("SELECT name FROM sqlite_master WHERE name = 'specialists'"), [])
        self.assertIsNone(get_legacy_assignment(self.db, case_id))
        self.assertEqual(self.execute("SELECT name FROM sqlite_master WHERE name = 'case_assignments'"), [])

    def test_original_input_document_and_analysis_provenance_are_preserved(self):
        case_id, analysis_id, _ = self.create_case()
        before = get_case(self.db, case_id)["original_input"]
        document = get_document(self.db, case_id)
        decision = allocate_group(self.db, case_id, analysis_id)
        self.assertEqual(get_case(self.db, case_id)["original_input"], before)
        self.assertEqual(get_document(self.db, case_id), document)
        self.assertEqual(decision["analysis_id"], analysis_id)
        self.assertEqual(decision["mode"], "live")
        self.assertEqual(decision["model"], "test-model")
        self.assertEqual(decision["document_sha256"], document["sha256"])
        self.assertEqual(get_case_history(self.db, case_id)[-1]["event_type"], "Allocated to group")

    def test_repeated_or_concurrent_allocation_creates_one_decision_event(self):
        case_id, analysis_id, _ = self.create_case()
        barrier = Barrier(2)

        def allocate(_):
            barrier.wait()
            return allocate_group(self.db, case_id, analysis_id)

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(allocate, range(2)))
        history = get_case_history(self.db, case_id)
        self.assertEqual(results[0], results[1])
        self.assertEqual(allocate_group(self.db, case_id, None), results[0])
        self.assertEqual(get_case_history(self.db, case_id), history)
        self.assertEqual(len([event for event in history if event["event_type"] == "Allocated to group"]), 1)
        self.assertEqual(self.execute("SELECT COUNT(*) FROM case_group_decisions")[0][0], 1)

    def test_specialist_capacity_availability_and_qualification_are_irrelevant(self):
        initialize_allocation(self.db)
        self.execute("UPDATE specialists SET capacity = 0, available = 0")
        self.execute("DELETE FROM specialist_qualifications")
        case_id, analysis_id, _ = self.create_case()
        self.assertEqual(allocate_group(self.db, case_id, analysis_id)["status"], "Allocated")
        self.assertEqual(self.execute("SELECT COUNT(*) FROM case_assignments")[0][0], 0)

    def test_conflicts_and_missing_required_facts_leave_unallocated(self):
        for change in (
            {"contradictions": ["The description and document give different customer segments."]},
            {"clarification_questions": ["Which contracting customer is correct?"]},
            {"customer_segment": "Unknown"},
            {"customer_segment": "Medium Business", "handling_approach": "Standard/Digital-eligible",
             "digital_eligibility": "Unknown"},
        ):
            with self.subTest(change=change):
                case_id, _, result = self.create_case(analyse=False)
                result.update(change)
                attempt = save_analysis(self.db, case_id, "live", "test-model", result, None)
                decision = allocate_group(self.db, case_id, attempt["analysis_id"])
                self.assertEqual(decision["status"], "Clarification needed")
                self.assertIsNone(get_group_assignment(self.db, case_id))
                history = get_case_history(self.db, case_id)
                self.assertEqual(allocate_group(self.db, case_id, attempt["analysis_id"]), decision)
                self.assertEqual(get_case_history(self.db, case_id), history)

    def test_stale_analysis_rejects_without_writing_and_latest_failure_blocks(self):
        case_id, analysis_id, _ = self.create_case()
        failed = save_analysis(self.db, case_id, "live", "test-model", None, "Analysis failed.")
        history = get_case_history(self.db, case_id)
        with self.assertRaisesRegex(ValueError, "analysis changed"):
            allocate_group(self.db, case_id, analysis_id)
        self.assertEqual(get_case_history(self.db, case_id), history)
        self.assertEqual(allocate_group(self.db, case_id, failed["analysis_id"])["status"], "Analysis failed")
        self.assertIsNone(get_group_assignment(self.db, case_id))

    def test_missing_or_unreadable_document_and_absent_analysis_block(self):
        case_id, _, _ = self.create_case(analyse=False)
        self.assertEqual(allocate_group(self.db, case_id, None)["status"], "Analysis needed")
        self.execute("DELETE FROM case_documents WHERE case_id = ?", (case_id,))
        self.assertEqual(allocate_group(self.db, case_id, None)["status"], "Document needs attention")
        case_id, analysis_id, _ = self.create_case(ready=False)
        self.assertEqual(allocate_group(self.db, case_id, analysis_id)["status"], "Document needs attention")
        document = get_document(self.db, case_id)
        save_document_extraction(self.db, case_id,
                                 {"filename": document["filename"], "sha256": document["sha256"],
                                  "size_bytes": document["size_bytes"], "status": "Needs attention",
                                  "text": "", "pages": [], "error": "No readable text."})
        self.assertEqual(allocate_group(self.db, case_id, analysis_id)["status"], "Document needs attention")
        self.assertIsNone(get_group_assignment(self.db, case_id))

    def test_invalid_schema_or_source_evidence_blocks_allocation(self):
        for defect in ("missing_field", "invented_quote", "wrong_page", "no_evidence"):
            with self.subTest(defect=defect):
                case_id, _, original = self.create_case(analyse=False)
                result = copy.deepcopy(original)
                if defect == "missing_field":
                    del result["digital_eligibility"]
                elif defect == "invented_quote":
                    result["facts"][0]["quote"] = "Invented source text."
                elif defect == "wrong_page":
                    result["facts"][0]["page"] = 2
                else:
                    result["facts"] = []
                attempt = save_analysis(self.db, case_id, "live", "test-model", result, None)
                self.assertEqual(allocate_group(self.db, case_id, attempt["analysis_id"])["status"], "Analysis failed")
                self.assertIsNone(get_group_assignment(self.db, case_id))

    def test_new_analysis_hides_old_deferred_decision_and_retains_history(self):
        case_id, _, result = self.create_case(analyse=False)
        conflict = {**result, "contradictions": ["Customer details conflict."]}
        first = save_analysis(self.db, case_id, "live", "test-model", conflict, None)
        old = allocate_group(self.db, case_id, first["analysis_id"])
        current = save_analysis(self.db, case_id, "live", "test-model", result, None)
        self.assertIsNone(get_group_decision(self.db, case_id))
        self.assertTrue(any(event["details"] == old for event in get_case_history(self.db, case_id)))
        self.assertEqual(allocate_group(self.db, case_id, current["analysis_id"])["status"], "Allocated")

    def test_saved_group_allocation_prevents_changes_to_its_evidence(self):
        case_id, analysis_id, result = self.create_case()
        original = allocate_group(self.db, case_id, analysis_id)
        history = get_case_history(self.db, case_id)
        with self.assertRaisesRegex(ValueError, "already allocated"):
            save_analysis(self.db, case_id, "live", "test-model", result, None)
        with self.assertRaisesRegex(ValueError, "already allocated"):
            save_review(self.db, case_id, {}, analysis_id)
        extraction = get_document(self.db, case_id)["extraction"]
        with self.assertRaisesRegex(ValueError, "already allocated"):
            save_document_extraction(self.db, case_id, {**extraction, "text": "Changed text."})
        # An identical extraction retry does not mutate anything and remains safe.
        self.assertEqual(save_document_extraction(self.db, case_id, extraction), extraction)
        self.assertEqual(get_case_history(self.db, case_id), history)
        self.assertEqual(get_group_assignment(self.db, case_id), original)

    def test_decision_and_status_roll_back_if_history_fails(self):
        case_id, analysis_id, _ = self.create_case()
        self.execute("""CREATE TRIGGER fail_history BEFORE INSERT ON case_events
                        BEGIN SELECT RAISE(ABORT, 'Simulated history failure'); END""")
        with self.assertRaises(sqlite3.IntegrityError):
            allocate_group(self.db, case_id, analysis_id)
        self.assertIsNone(get_group_decision(self.db, case_id))
        self.assertIsNone(get_group_assignment(self.db, case_id))
        self.assertEqual(get_case(self.db, case_id)["status"], "Allocation pending")
        self.execute("DROP TRIGGER fail_history")
        self.assertEqual(allocate_group(self.db, case_id, analysis_id)["status"], "Allocated")

    def test_historical_specialist_assignment_is_not_modified_or_converted(self):
        initialize_allocation(self.db)
        case_id, analysis_id, _ = self.create_case()
        legacy = allocate_case(self.db, case_id, analysis_id)
        before = get_case(self.db, case_id)
        history = get_case_history(self.db, case_id)
        initialize_group_allocation(self.db)
        with self.assertRaisesRegex(ValueError, "already assigned"):
            allocate_group(self.db, case_id, analysis_id)
        self.assertEqual(get_case(self.db, case_id), before)
        self.assertEqual(get_case_history(self.db, case_id), history)
        self.assertEqual(get_assignment(self.db, case_id), legacy)
        self.assertEqual(get_legacy_assignment(self.db, case_id), legacy)
        self.assertIsNone(get_group_decision(self.db, case_id))

    def test_mock_provenance_stays_explicit_without_claiming_live_ai(self):
        case_id, analysis_id, _ = self.create_case(mode="mock")
        self.assertEqual(allocate_group(self.db, case_id, analysis_id)["mode"], "mock")

    def test_unknown_case_has_no_decision_and_cannot_allocate(self):
        self.assertIsNone(get_group_decision(self.db, "missing"))
        self.assertIsNone(get_group_assignment(self.db, "missing"))
        with self.assertRaisesRegex(ValueError, "does not exist"):
            allocate_group(self.db, "missing", None)


if __name__ == "__main__":
    unittest.main()
