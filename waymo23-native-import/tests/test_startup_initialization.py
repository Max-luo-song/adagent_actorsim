import ast
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

BASE=Path(__file__).resolve().parents[2]
ADAPTER=BASE/'waymo23-native-import'
sys.path.insert(0,str(ADAPTER))


def initializer(source):
    tree=ast.parse(source)
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    cut=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='cameras' for t in n.targets))
    return main.body[:cut]


class StartupTests(unittest.TestCase):
    def test_model_initialization_statements_unchanged(self):
        after=(BASE/'code/UniAD_SIM/tools/closeloop/e2e.py').read_text()
        prefix=[n for n in initializer(after) if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='_startup_mark')]
        # Canonical AST digest of the original initializer, before startup-only changes.
        self.assertEqual(hashlib.sha256(ast.dump(ast.Module(body=prefix,type_ignores=[])).encode()).hexdigest(),'0b826fbb65debf812df39a3037995f5f5b0dd2cdde58f2ce27c0edd62470543e')
    def test_visualization_is_deferred_past_resident_initializer(self):
        tree=ast.parse((BASE/'code/UniAD_SIM/tools/closeloop/e2e.py').read_text())
        modules=[n.module for n in tree.body if isinstance(n,ast.ImportFrom)]
        self.assertNotIn('tools.closeloop.visualizer',modules);self.assertNotIn('visualizer',modules)
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        later=[n.module for n in main.body[len(initializer((BASE/'code/UniAD_SIM/tools/closeloop/e2e.py').read_text())):] if isinstance(n,ast.ImportFrom)]
        self.assertIn('tools.closeloop.visualizer',later);self.assertIn('visualizer',later)
    def test_supervisor_imports_no_numerical_or_visual_packages(self):
        script="import sys;sys.path.insert(0,sys.argv[1]);import gpu_render_smoke,uniad_startup_probe;assert not set(('numpy','scipy','torch','open3d','dash')) & set(sys.modules)"
        subprocess.run([sys.executable,'-c',script,str(ADAPTER)],check=True,timeout=20)
    def test_timing_flushes_early_and_later_stages(self):
        import startup_timing as timing
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'startup.json'
            timing.configure(path);timing.mark('test_stage',detail='recorded')
            data=json.loads(path.read_text());event=data['events'][-1]
            self.assertEqual(event['stage'],'test_stage');self.assertGreaterEqual(event['delta_seconds'],0)
            self.assertEqual(event['detail'],'recorded')
        timing._REPORT=None

if __name__=='__main__':unittest.main()
