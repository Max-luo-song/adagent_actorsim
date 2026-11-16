"""Source/data separation and relocation checks using synthetic fixtures only."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class PortablePathTests(unittest.TestCase):
    def test_relocated_repository_without_original_sources(self):
        actual=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix='hugsim-relocation-') as directory:
            root=Path(directory)/'checkout with spaces';shared=root/'integrations/_shared';shared.mkdir(parents=True)
            shutil.copy2(actual/'integrations/_shared/paths.py',shared/'paths.py')
            prefix='gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair'
            code=root/'integrations/chain-repair/phase26/demo';code.mkdir(parents=True)
            (code/'driver.py').write_text('VALUE = 17\n')
            data=root/'outputs'/prefix/'phase26/demo';data.mkdir(parents=True)
            (data/'weights.bin').write_bytes(b'synthetic weights')
            (data/'metadata.json').write_text('{"synthetic": true}')
            (data/'obsolete.py').write_text('SHOULD_NOT_IMPORT = True\n')
            (shared/'archived_sources.json').write_text(json.dumps({'schema':'archived_source_paths_v1','paths':['phase26/demo/obsolete.py']}))
            source='''
from integrations._shared.paths import Path,ROOT,CHAIN_DATA,CHAIN_CODE,expand_paths
p=Path(CHAIN_DATA/'phase26/demo/driver.py')
assert str(p)==str(CHAIN_CODE/'phase26/demo/driver.py')
assert p.read_text()=='VALUE = 17\\n'
assert not (CHAIN_DATA/'phase26/demo/driver.py').exists()
assert not Path(CHAIN_DATA/'phase26/demo/obsolete.py').exists()
assert Path(CHAIN_CODE/'phase26/demo/weights.bin').read_bytes()==b'synthetic weights'
assert Path(CHAIN_CODE/'phase26/demo').parent.name=='phase26'
assert sorted(x.name for x in Path(CHAIN_DATA/'phase26/demo').glob('*'))==['driver.py','metadata.json','weights.bin']
assert expand_paths({'path':'${HUGSIM_ROOT}/code/HUGSIM'})['path']==str(ROOT/'code/HUGSIM')
assert expand_paths('/inspire/hdd/global_user/previous-user/datasets/waymo').endswith('/assets/external/datasets/waymo')
print('relocation PASS')
'''
            env=dict(os.environ,HUGSIM_ROOT=str(root),PYTHONPATH=str(root))
            for key in ['HUGSIM_OUTPUT_ROOT','HUGSIM_CHAIN_CODE_ROOT','HUGSIM_EXTERNAL_ROOT']:env.pop(key,None)
            result=subprocess.run([sys.executable,'-B','-c',source],cwd='/tmp',env=env,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_root_override_shell(self):
        actual=Path(__file__).resolve().parents[2]
        env=dict(os.environ,HUGSIM_ROOT=str(actual))
        result=subprocess.run(['bash','-c','source "$1"; test "$HUGSIM_ROOT" = "$2"; test -n "$HUGSIM_PYTHON"',
                               '_',str(actual/'integrations/_shared/env.sh'),str(actual)],env=env,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
