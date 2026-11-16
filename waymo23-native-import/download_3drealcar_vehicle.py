"""Extract one vehicle through HTTP ranges, never download entire Drive ZIPs."""
import argparse
import hashlib
from html.parser import HTMLParser
import http.cookiejar
import io
import json
from pathlib import Path, PurePosixPath
import time
from urllib.parse import urlencode, urlsplit
from urllib.error import HTTPError, URLError
from urllib.request import build_opener, HTTPCookieProcessor, Request
import zipfile

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE/'data/downloads/3drealcar_raw'


class Confirmation(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action, self.fields = None, {}
        self.active = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'form' and a.get('id') == 'download-form':
            self.active = True
            self.action = a.get('action')
        if self.active and tag == 'input' and a.get('name'):
            self.fields[a['name']] = a.get('value', '')

    def handle_endtag(self, tag):
        if tag == 'form':
            self.active = False


class RemoteZip(io.RawIOBase):
    def __init__(self, archive, budget):
        self.archive, self.size, self.position = archive, int(archive['bytes']), 0
        self.opener = build_opener(HTTPCookieProcessor(http.cookiejar.CookieJar()))
        self.url = 'https://drive.usercontent.google.com/download?'+urlencode(
            dict(export='download', id=archive['id'], confirm='t'))
        self.budget = budget
        self.network_errors = []

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset+(0 if whence == 0 else self.position if whence == 1 else self.size)
        if self.position < 0:
            raise ValueError('Negative remote seek')
        return self.position

    def read(self, count=-1):
        count = self.size-self.position if count < 0 else min(count, self.size-self.position)
        if count <= 0:
            return b''
        if count > 256*1024*1024:
            raise ValueError('Remote read larger than 256MiB; refusing whole archive download')
        if count > 4*1024*1024:
            # zipfile reads the entire central directory in one call. A ZIP
            # containing hundreds of photo sets can have a large directory.
            # Split that logical read into bounded HTTP ranges.
            chunks = []
            remaining = count
            while remaining:
                chunk = self.read(min(remaining, 4*1024*1024))
                if not chunk:
                    raise RuntimeError('Unexpected end of ZIP directory')
                chunks.append(chunk)
                remaining -= len(chunk)
            return b''.join(chunks)
        start = self.position
        for attempt in range(3):
            request = Request(self.url, headers={'User-Agent': 'Mozilla/5.0',
                              'Range': f'bytes={start}-{start+count-1}'})
            try:
                response = self.opener.open(request, timeout=45)
            except (OSError, URLError) as exc:
                # zipfile catches OSError and replaces it with BadZipFile. Keep
                # transport failures visible instead of blaming the archive.
                detail = dict(host=urlsplit(self.url).hostname, error=str(exc),
                              http_status=getattr(exc, 'code', None))
                self.network_errors.append(detail)
                if attempt == 0:
                    self.url = 'https://drive.google.com/uc?'+urlencode(
                        dict(export='download', id=self.archive['id'], confirm='t'))
                    continue
                if attempt == 1 and not isinstance(exc, HTTPError):
                    time.sleep(2.)
                    continue
                raise RuntimeError('Public Drive download request failed: '+json.dumps(self.network_errors)) from exc
            with response:
                if response.status == 206:
                    cr = response.headers.get('Content-Range', '')
                    if cr != f'bytes {start}-{start+count-1}/{self.size}':
                        raise ValueError('Range or archive size mismatch: '+cr)
                    if self.budget['received']+count > self.budget['maximum']:
                        raise ValueError('Fixed download budget exhausted')
                    data = response.read(count+1)
                    if len(data) != count:
                        raise ValueError('Incomplete range response')
                    self.budget['received'] += count
                    self.position += count
                    return data
                kind = response.headers.get('Content-Type', '')
                if 'text/html' not in kind:
                    raise RuntimeError('Server ignores HTTP ranges; whole archive download refused')
                page = response.read(1024*1024).decode('utf-8', errors='replace')
            diagnostics = ROOT/'drive_diagnostics'
            diagnostics.mkdir(exist_ok=True)
            (diagnostics/(self.archive['id']+'_'+str(start)+'.html')).write_text(page)
            if 'Quota exceeded' in page or 'Too many users have viewed or downloaded' in page:
                raise RuntimeError('Google Drive public-file download quota exceeded; response saved in drive_diagnostics')
            form = Confirmation()
            form.feed(page)
            if not form.action or urlsplit(form.action).hostname not in ('drive.google.com', 'drive.usercontent.google.com'):
                raise RuntimeError('Drive needs login, exceeded quota, or supplied no public confirmation form')
            self.url = form.action+'?'+urlencode(form.fields)
            time.sleep(.2)
        raise RuntimeError('Drive confirmation did not produce an HTTP range response')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vehicle', default='2024_06_04_13_44_39')
    parser.add_argument('--max-archives', type=int, default=12)
    args = parser.parse_args()
    if not args.vehicle.replace('_', '').isdigit():
        parser.error('Expected timestamp vehicle identifier')
    if not 1 <= args.max_archives <= 12:
        parser.error('Archive budget must be between 1 and 12')
    inventory = json.loads((ROOT/'archive_inventory.json').read_text())
    inventory.sort(key=lambda a: (not a['name'].startswith('HQ'), a['name']))
    inventory = inventory[:args.max_archives]
    budget = dict(received=0, maximum=1024*1024*1024)
    report = dict(vehicle=args.vehicle, status='searching', archives=[], files=[], download_complete=False,
                  downloader_version=3)
    destination = ROOT/args.vehicle
    destination.mkdir(exist_ok=True)
    status_file = ROOT/'vehicle_download_status.json'
    if status_file.exists():
        previous = ROOT/('vehicle_download_status_'+str(time.time_ns())+'.json')
        previous.write_bytes(status_file.read_bytes())
    def save():
        report['received_bytes'] = budget['received']
        status_file.write_text(json.dumps(report, indent=2))
    for archive in inventory:
        record = dict(name=archive['name'], id=archive['id'], bytes=archive['bytes'])
        report['archives'].append(record)
        save()
        print('Inspecting ZIP index:', archive['name'], flush=True)
        try:
            with zipfile.ZipFile(RemoteZip(archive, budget)) as zipped:
                matches = [v for v in zipped.infolist()
                           if args.vehicle in PurePosixPath(v.filename).parts and not v.is_dir()]
                record['target_members'] = len(matches)
                if not matches:
                    record['status'] = 'target_not_in_archive'
                    continue
                if sum(v.file_size for v in matches) > 2*1024**3:
                    raise ValueError('Vehicle expands beyond fixed 2GiB budget')
                for member in matches:
                    parts = PurePosixPath(member.filename).parts
                    relative = PurePosixPath(*parts[parts.index(args.vehicle)+1:])
                    if '..' in relative.parts or relative.is_absolute() or not relative.parts:
                        raise ValueError('Unsafe archive path')
                    target = destination.joinpath(*relative.parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    partial = target.with_name(target.name+'.part')
                    if target.exists() or partial.exists():
                        raise ValueError('Existing output requires review before retry: '+str(target))
                    sha = hashlib.sha256()
                    with zipped.open(member) as source, partial.open('xb') as stream:
                        while True:
                            block = source.read(1024*1024)
                            if not block:
                                break
                            stream.write(block)
                            sha.update(block)
                    partial.replace(target)
                    report['files'].append(dict(path=str(target), bytes=target.stat().st_size,
                                                sha256=sha.hexdigest(), archive_member=member.filename))
                    save()
                record['status'] = 'target_extracted'
                report.update(status='downloaded_requires_evidence_audit', download_complete=True,
                              n1_reference_validated=False)
                save()
                print('Downloaded vehicle:', destination, flush=True)
                return 0
        except Exception as exc:
            record.update(status='failed', error=str(exc))
            print('Archive unavailable:', str(exc), flush=True)
            if record.get('target_members'):
                report['status'] = 'target_download_failed'
                save()
                return 1
        finally:
            save()
    report['status'] = 'no_download_check_archive_errors'
    save()
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
