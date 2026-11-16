import hashlib
import io
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import runtime_bundle
from runtime_bundle import digest,extract_verified,prepare_checkpoint


class BundleTests(unittest.TestCase):
    def fixture(self,folder,name='torch/example.so'):
        payload=b'identical original library bytes';archive=folder/'test.tar'
        with tarfile.open(archive,'w') as tar:
            info=tarfile.TarInfo(name);info.size=len(payload);tar.addfile(info,io.BytesIO(payload))
        return archive,dict(archive_sha256=digest(archive),members=[dict(path=name,bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())])
    def test_extract_checks_original_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);archive,manifest=self.fixture(root)
            target=root/'cache';target.mkdir();extract_verified(archive,manifest,target)
            self.assertEqual(digest(target/'torch/example.so'),manifest['members'][0]['sha256'])
    def test_wrong_file_hash_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);archive,manifest=self.fixture(root);manifest['members'][0]['sha256']='0'*64
            target=root/'cache';target.mkdir()
            with self.assertRaisesRegex(ValueError,'file SHA'):extract_verified(archive,manifest,target)
    def test_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);archive,manifest=self.fixture(root,'../escape')
            target=root/'cache';target.mkdir()
            with self.assertRaisesRegex(ValueError,'Unsafe'):extract_verified(archive,manifest,target)
            self.assertFalse((root/'escape').exists())

    def test_frozen_checkpoint_copy_and_hit_verify_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'code/UniAD_SIM/ckpts/uniad_base_e2e.pth'
            source.parent.mkdir(parents=True);source.write_bytes(b'frozen model bytes')
            expected=dict(bytes=source.stat().st_size,mtime_ns=source.stat().st_mtime_ns,sha256=digest(source))
            cache=root/'cache';cache.mkdir()
            with patch.object(runtime_bundle,'BASE',root):
                first=prepare_checkpoint(expected,cache);self.assertFalse(first['cache_hit'])
                second=prepare_checkpoint(expected,cache);self.assertTrue(second['cache_hit'])
                Path(first['path']).write_bytes(b'corrupt')
                with self.assertRaisesRegex(ValueError,'Cached checkpoint changed'):prepare_checkpoint(expected,cache)
    def test_changed_source_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'code/UniAD_SIM/ckpts/uniad_base_e2e.pth'
            source.parent.mkdir(parents=True);source.write_bytes(b'frozen')
            expected=dict(bytes=source.stat().st_size,mtime_ns=source.stat().st_mtime_ns,sha256=digest(source))
            source.write_bytes(b'different weights');cache=root/'cache';cache.mkdir()
            with patch.object(runtime_bundle,'BASE',root),self.assertRaisesRegex(ValueError,'Frozen checkpoint changed'):
                prepare_checkpoint(expected,cache)


if __name__=='__main__':unittest.main()
