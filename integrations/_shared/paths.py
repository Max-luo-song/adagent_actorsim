"""Portable paths for the published source tree and external experiment assets.

The legacy phase layout is retained under integrations/chain-repair. Path imports
in migrated modules use this compatibility boundary instead of changing numerical
code. No global pathlib or filesystem monkey patching is performed.
"""
import os
import re
import json
from functools import lru_cache
from pathlib import Path as NativePath

ROOT = NativePath(os.environ.get('HUGSIM_ROOT', NativePath(__file__).resolve().parents[2])).expanduser().resolve()
OUTPUT_ROOT = NativePath(os.environ.get('HUGSIM_OUTPUT_ROOT', ROOT/'outputs')).expanduser().resolve()
EXTERNAL_ROOT = NativePath(os.environ.get('HUGSIM_EXTERNAL_ROOT', ROOT/'assets/external')).expanduser().resolve()
CHAIN_RELATIVE = 'gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair'
CHAIN_DATA = OUTPUT_ROOT/CHAIN_RELATIVE
CHAIN_CODE = NativePath(os.environ.get('HUGSIM_CHAIN_CODE_ROOT', ROOT/'integrations/chain-repair')).expanduser().resolve()
os.environ.setdefault('HUGSIM_ROOT', str(ROOT))
os.environ.setdefault('HUGSIM_OUTPUT_ROOT', str(OUTPUT_ROOT))
os.environ.setdefault('HUGSIM_EXTERNAL_ROOT', str(EXTERNAL_ROOT))


def expand_string(value):
    """Expand declared roots, including paths found in archived input manifests."""
    if not isinstance(value, str):
        return value
    if value.startswith(('outputs/', 'code/', 'integrations/', 'waymo23-native-import/')):
        value = str(ROOT/value)
    for name, root in [('HUGSIM_ROOT', ROOT), ('HUGSIM_OUTPUT_ROOT', OUTPUT_ROOT),
                       ('HUGSIM_EXTERNAL_ROOT', EXTERNAL_ROOT), ('HUGSIM_CHAIN_CODE_ROOT', CHAIN_CODE)]:
        value = value.replace('${'+name+'}', str(root))
    # Historical manifests are read-only inputs; do not rewrite them in place.
    value = re.sub(r'/inspire/[^\s\"\']*?/hugsim-reproduction(?=/|$)', str(ROOT), value)
    configured_roots = (str(ROOT), str(OUTPUT_ROOT), str(EXTERNAL_ROOT), str(CHAIN_CODE))
    configured = any(value == p or value.startswith(p+'/') for p in configured_roots)
    if value.startswith('/inspire/') and not configured:
        suffix = re.sub(r'^/inspire/(?:hdd3/project/intelligent-driving-agent|hdd/global_user)/[^/]+/?', '', value)
        if suffix == value:
            suffix = value.removeprefix('/inspire/')
        value = str(EXTERNAL_ROOT/suffix)
    output_prefix = str(ROOT/'outputs')
    if value == output_prefix or value.startswith(output_prefix+'/'):
        value = str(OUTPUT_ROOT)+value[len(output_prefix):]
    return value


def _logical(value):
    value = expand_string(os.fspath(value))
    # Keep original phase-relative ancestry so existing __file__.parents layouts
    # remain meaningful even after copying an entire implementation snapshot.
    prefix = str(CHAIN_CODE)
    if value == prefix or value.startswith(prefix+'/'):
        return str(CHAIN_DATA)+value[len(prefix):]
    return value


def physical_path(value):
    """Prefer published implementations; missing source-adjacent data stays external."""
    value = _logical(value)
    prefix = str(CHAIN_DATA)
    if value == prefix or value.startswith(prefix+'/'):
        published = str(CHAIN_CODE)+value[len(prefix):]
        if os.path.lexists(published):
            return published
        relative = value[len(prefix):].lstrip('/')
        removed = _archived_sources()
        if relative in removed or (relative and any(p.startswith(relative+'/') for p in removed)):
            # A consciously pruned implementation must never silently import the
            # old outputs copy. Data files not listed in the archive still fall back.
            return published
    return value


@lru_cache(maxsize=1)
def _archived_sources():
    manifest = ROOT/'integrations/_shared/archived_sources.json'
    if not manifest.is_file():return frozenset()
    record = json.loads(manifest.read_text())
    if record.get('schema') != 'archived_source_paths_v1':
        raise ValueError('Invalid published source exclusion manifest')
    return frozenset(record['paths'])


class Path(type(NativePath())):
    """Explicit legacy-layout adapter; ordinary project paths behave like pathlib.

    Internal components preserve legacy ancestry, while str/os.fspath select the
    published implementation if present. glob merges code and external data.
    Writes should use explicit OUTPUT_ROOT destinations, never source directories.
    """
    def __new__(cls, *args, **kwargs):
        if args:
            args = tuple(_logical(NativePath.__str__(a) if isinstance(a, Path) else a)
                         if isinstance(a, (str, os.PathLike)) else a for a in args)
        return super().__new__(cls, *args, **kwargs)

    def __str__(self):
        return physical_path(super().__str__())

    def resolve(self, strict=False):
        raw = NativePath.__str__(self)
        resolved = NativePath(raw).resolve(strict=False)
        if strict and not os.path.exists(physical_path(str(resolved))):
            raise FileNotFoundError(str(self))
        return type(self)(str(resolved))

    def glob(self, pattern):
        raw = NativePath.__str__(self)
        seen = set()
        for location in dict.fromkeys([physical_path(raw), raw]):
            for item in NativePath(location).glob(pattern):
                result = type(self)(str(item))
                logical = NativePath.__str__(result)
                prefix = str(CHAIN_DATA)+'/'
                if logical.startswith(prefix) and logical[len(prefix):] in _archived_sources():
                    continue
                if logical not in seen:
                    seen.add(logical)
                    yield result

    def rglob(self, pattern):
        return self.glob('**/'+str(pattern))


def portable_path(value):
    """String form for subprocess arguments, configuration and plain open calls."""
    return str(Path(value)) if isinstance(value, (str, os.PathLike)) else value


def expand_paths(value):
    """Expand path tokens in configuration values and manifest path keys."""
    if isinstance(value, str):
        expanded = expand_string(value)
        if expanded.startswith(str(ROOT)) or expanded.startswith(str(OUTPUT_ROOT)) or expanded.startswith(str(EXTERNAL_ROOT)):
            return physical_path(expanded)
        return expanded
    if isinstance(value, list):
        return [expand_paths(item) for item in value]
    if isinstance(value, dict):
        return {expand_paths(k):expand_paths(v) for k,v in value.items()}
    return value


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', nargs='?')
    args = parser.parse_args()
    print(portable_path(args.path) if args.path else ROOT)


if __name__ == '__main__':
    main()
