
# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand
Path = _HugsimPath

import copy,json,io
import numpy as np
from contextlib import redirect_stderr
import unittest
pass  # Path is provided by the portable boundary above
import test_agent_bridge as fixtures
from agent_jobs import validate_job,resolve,run_job,execution_coverage
import agent_bootstrap as bootstrap

class BatchTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.BridgeTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.bridge=self.fixture.bridge;self.root=self.fixture.root
        self.job=_hugsim_expand(json.loads((bootstrap.HERE/'examples/edit_preview.json').read_text()))
    def test_complete_public_json_example_uses_same_real_editor_and_reset(self):
        result=run_job(self.job,self.bridge,self.root/'job')
        self.assertTrue(result['passed']);self.assertEqual(result['completed_steps'],18)
        self.assertEqual(result['expected_error_count'],3)
        self.assertEqual(result['actual_renders'],4);self.assertEqual(result['scene_loads'],1)
        self.assertEqual(result['execution_backend'],'cpu_test_double');self.assertFalse(result['closed_loop_driving'])
        steps=_hugsim_expand(json.loads((self.root/'job/steps.json').read_text()))
        self.assertTrue(steps[8]['response']['replayed'])
        self.assertEqual(result['coverage']['edits'],2)
        self.assertEqual(result['coverage']['native_edits'],1)
        self.assertEqual(result['coverage']['managed_edits'],1)
        self.assertEqual(result['coverage']['replayed_requests'],1)
        self.assertEqual(result['coverage']['resets'],1)
        evidence=_hugsim_expand(json.loads((self.root/'job/verification.json').read_text()))
        self.assertTrue(all(evidence['checks'].values()))
        equal=[c for c in evidence['comparisons'] if c['relation']=='equal_rgb']
        self.assertEqual(len(equal),2)
        self.assertTrue(all(all(v==0 for v in c['rgb_changed_pixels'].values()) for c in equal))
    def test_public_simple_insert_example_runs_all_five_steps_without_replay(self):
        job=_hugsim_expand(json.loads((bootstrap.HERE/'examples/simple_insert.json').read_text()))
        result=run_job(job,self.bridge,self.root/'simple')
        self.assertTrue(result['passed']);self.assertEqual(result['completed_steps'],5)
        self.assertEqual(result['actual_renders'],2);self.assertEqual(result['expected_error_count'],0)
        steps=_hugsim_expand(json.loads((self.root/'simple/steps.json').read_text()))
        insert=steps[2]['response'];self.assertEqual(insert['result']['edit_revision'],1)
        self.assertFalse(insert['replayed'])
        self.assertTrue((self.root/'simple/preview_comparison.jpg').is_file())
        self.assertEqual(result['coverage']['resets'],0)
        self.assertIn('successful_resets_verified',result['not_exercised_checks'])

    def test_failed_or_replayed_resets_do_not_count_as_verified_new_resets(self):
        steps=[dict(action='reset',response=dict(status='error',replayed=False)),
               dict(action='reset',response=dict(status='ok',replayed=True))]
        result=execution_coverage(steps)
        self.assertEqual(result['resets'],0)
        self.assertEqual(result['rejected_requests'],1)
        self.assertEqual(result['replayed_requests'],1)

    def test_required_edits_do_not_count_replay_or_rejected_commands(self):
        job=copy.deepcopy(self.job);job['verification']['minimum_coverage']['edits']=3
        with self.assertRaises(RuntimeError):run_job(job,self.bridge,self.root/'coverage_fail')
        data=_hugsim_expand(json.loads((self.root/'coverage_fail/acceptance.json').read_text()))
        self.assertEqual(data['completed_steps'],18)
        self.assertEqual(data['coverage']['edits'],2)
        self.assertEqual(data['coverage']['replayed_requests'],1)
        self.assertEqual(data['coverage']['rejected_requests'],3)
        self.assertFalse(data['checks']['required_coverage_met'])

    def test_missing_required_rejection_fails_even_when_requests_match(self):
        job=copy.deepcopy(self.job);job['verification']['required_rejections'].append('object_not_found')
        with self.assertRaises(RuntimeError):run_job(job,self.bridge,self.root/'rejection_fail')
        data=_hugsim_expand(json.loads((self.root/'rejection_fail/verification.json').read_text()))
        self.assertFalse(data['required_rejections']['object_not_found'])
        self.assertFalse(data['checks']['required_rejections_observed'])

    def test_equal_pixels_at_different_times_do_not_prove_reset_restoration(self):
        np.savetxt(self.root/'inputs/ego_pose/081.txt',np.eye(4))
        job=copy.deepcopy(self.job)
        step=next(s for s in job['steps'] if s['request']['request_id']=='reset_preview')
        step['request']['args']['source_frame']=81
        with self.assertRaises(RuntimeError):run_job(job,self.bridge,self.root/'time_fail')
        data=_hugsim_expand(json.loads((self.root/'time_fail/verification.json').read_text()))
        pair=next(c for c in data['comparisons'] if c['second_request']=='reset_preview')
        self.assertFalse(pair['comparable']);self.assertFalse(pair['passed'])
        self.assertTrue(all(v==0 for v in pair['rgb_changed_pixels'].values()))

    def test_invalid_evidence_references_are_rejected_before_loading(self):
        job=copy.deepcopy(self.job);job['verification']['equal_rgb']=[['before_preview','insert_car']]
        with self.assertRaisesRegex(ValueError,'preview/observe'):validate_job(job)
        self.assertIsNone(self.bridge.session.backend)
        job=copy.deepcopy(self.job);job['verification']['minimum_coverage']['resets']=2
        with self.assertRaisesRegex(ValueError,'exceeds declared'):validate_job(job)
        self.assertIsNone(self.bridge.session.backend)

    def test_fresh_owner_first_request_cannot_expect_a_cached_replay(self):
        job=_hugsim_expand(json.loads((bootstrap.HERE/'examples/simple_insert.json').read_text()))
        job['steps'][2]['expect']['replayed']=True
        with self.assertRaisesRegex(ValueError,'first occurrence'):validate_job(job)
        self.assertIsNone(self.bridge.session.backend)
        self.assertEqual(self.bridge.load_count,0)

    def test_scene_switch_example_rejects_wrong_scene_placement(self):
        job=_hugsim_expand(json.loads((bootstrap.HERE/'examples/scene_switch.json').read_text()))
        result=run_job(job,self.bridge,self.root/'switch')
        self.assertTrue(result['passed']);self.assertEqual(result['scene_loads'],2)
        self.assertEqual(result['expected_error_count'],1)
    def test_jsonl_keeps_model_logs_off_the_response_stream_and_reports_parse_errors(self):
        from run_agent_json import serve_jsonl
        factory=self.bridge.session.factory
        def noisy(spec):
            print('explicit CPU loader log')
            return factory(spec)
        self.bridge.session.factory=noisy
        requests=[self.fixture.req('load_scene',{'scene_id':'552'},'stream_load'),
                  self.fixture.req('get_capabilities',identity='stream_caps'),self.fixture.req('close',identity='stream_close')]
        source=io.StringIO('{broken JSON}\n'+'\n'.join(json.dumps(r) for r in requests)+'\n')
        output=io.StringIO();errors=io.StringIO()
        with redirect_stderr(errors):serve_jsonl(self.bridge,source,output)
        replies=[_hugsim_expand(json.loads(line)) for line in output.getvalue().splitlines()]
        self.assertEqual(len(replies),4)
        self.assertEqual(replies[0]['error']['code'],'invalid_json')
        self.assertTrue(all(r['status']=='ok' for r in replies[1:]))
        self.assertIn('explicit CPU loader log',errors.getvalue())
        self.assertEqual(self.bridge.render_count,0)

    def test_unknown_or_forward_reference_rejected_before_execution(self):
        job=copy.deepcopy(self.job);job['steps'][0]['request']['args']={'bad':{'$ref':'future.result.generation'}}
        with self.assertRaisesRegex(ValueError,'forward'):validate_job(job)
        self.assertIsNone(self.bridge.session.backend)
    def test_reference_cannot_read_failed_request_and_has_no_expression_evaluation(self):
        with self.assertRaisesRegex(ValueError,'failed'):resolve({'$ref':'x.result.generation'},{'x':{'status':'error'}})
        with self.assertRaises(ValueError):resolve({'$ref':'__import__(os).system'},{})
    def test_unexpected_failure_stops_without_sending_remaining_edits(self):
        job=copy.deepcopy(self.job);job['steps'][2]['request']['args']['scene_id']='unknown'
        with self.assertRaises(RuntimeError):run_job(job,self.bridge,self.root/'failed')
        result=_hugsim_expand(json.loads((self.root/'failed/acceptance.json').read_text()))
        self.assertFalse(result['passed']);self.assertEqual(result['completed_steps'],3)
        self.assertEqual(result['scene_loads'],0)

if __name__=='__main__':unittest.main()
