"""Admission rejects per-scene policy drift before loading GPU models."""
from dataclasses import asdict
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared_rules import shared_rules, validate_case_rules
from edit_trust import EditTrustDomain, EditTrustError, digest
import numpy as np


class SharedRulesTests(unittest.TestCase):
    def case(self):
        return dict(source_report="unseen-scene.json",
                    protocol_hash=digest(shared_rules()["protocol"]))

    def test_rules_do_not_depend_on_scene_metadata(self):
        a = self.case()
        b = dict(a, source_report="another-unseen-scene.json")
        self.assertEqual(validate_case_rules(a), validate_case_rules(b))
        b["rules_hash"] = validate_case_rules(a)["rules_hash"]
        self.assertTrue(validate_case_rules(b)["case_rules_hash_explicit"])

    def test_per_scene_overrides_and_stale_rules_rejected(self):
        for override in (dict(edit_policy={"trust_threshold": .1}),
                         dict(track_start_score=.1), dict(rules_hash="stale")):
            with self.assertRaises(ValueError):
                validate_case_rules(dict(self.case(), **override))

    def test_domain_with_relaxed_policy_cannot_enter_shared_runner(self):
        domain = EditTrustDomain(np.ones((4, 4)), np.full((4, 4), 3),
                                 [0, 0], 1., "sensor", "baseline")
        validate_case_rules(self.case(), domain)
        policy = asdict(domain.policy)
        policy["support_error_m"] = 1.
        domain.policy = type(domain.policy)(**policy)
        with self.assertRaisesRegex(EditTrustError, "shared edit policy"):
            validate_case_rules(self.case(), domain)


if __name__ == "__main__":
    unittest.main()
