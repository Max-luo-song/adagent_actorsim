"""Synthetic-only credential detector regressions; no real keys or API calls."""
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('_git_preflight_tested',Path(__file__).with_name('git_preflight.py'))
scanner=importlib.util.module_from_spec(spec);spec.loader.exec_module(scanner)


class CredentialRules(unittest.TestCase):
    def test_empty_dotenv_fields_do_not_consume_following_lines(self):
        data=b'PATCH_API_KEY=\nOBS_ACCESS_KEY_ID=\nOBS_SECRET_ACCESS_KEY=\n'
        self.assertIsNone(scanner.PATTERNS['dotenv_credential'].search(data))

    def test_unquoted_dotenv_value_is_detected(self):
        data=b'PATCH_API_KEY='+b'Q'*32+b'\n'
        self.assertIsNotNone(scanner.PATTERNS['dotenv_credential'].search(data))

    def test_api_marker_inside_key_does_not_make_it_a_placeholder(self):
        generated=b's'+b'k-'+b'Q'*20+b'fake'+b'Q'*20
        self.assertIsNotNone(scanner.PATTERNS['service_api_key'].search(generated))
        self.assertIsNone(scanner.PLACEHOLDER.search(generated))

    def test_fixture_markers_are_explicit(self):
        self.assertIsNotNone(scanner.PLACEHOLDER.search(b'test_fixture_secret'))


if __name__=='__main__':unittest.main()
