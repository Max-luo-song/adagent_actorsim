"""Save the official public Drive listing for bounded download planning."""
import json
from pathlib import Path
import re
import http.cookiejar
from urllib.parse import urlencode
from urllib.request import Request, urlopen, build_opener, HTTPCookieProcessor

BASE = Path(__file__).resolve().parents[1]
URL = 'https://drive.google.com/drive/folders/1csYZdO4sJmxV-tMQRPPVSiSp53_3y6fG'


def probe_baidu(output):
    """Use the publisher's public share code; never use account credentials."""
    opener = build_opener(HTTPCookieProcessor(http.cookiejar.CookieJar()))
    share = 'https://pan.baidu.com/s/10us5DLkTq124hVcyhsWyIg'
    headers = {'User-Agent': 'Mozilla/5.0', 'Referer': share}
    def fetch(url, data=None):
        with opener.open(Request(url, data=data, headers=headers), timeout=30) as response:
            body = response.read(1024*1024+1)
            if len(body) > 1024*1024:
                raise ValueError('Baidu metadata exceeds 1MiB budget')
            return body.decode('utf-8', errors='replace')
    page = fetch(share)
    csrf = re.search(r'"csrf"\s*:\s*"([^"]+)"', page)
    query = dict(surl='0us5DLkTq124hVcyhsWyIg', web='1', clienttype='0')
    if csrf:
        query['csrf'] = csrf[1]
    reply = json.loads(fetch('https://pan.baidu.com/share/verify?'+urlencode(query),
                             urlencode(dict(pwd='pgmb', vcode='', vcode_str='')).encode()))
    result = dict(verification_errno=reply.get('errno'), anonymous_access=True)
    if reply.get('errno') != 0:
        result.update(status='public_code_verification_failed', captcha_or_login_not_automated=True)
        (output/'baidu_access_status.json').write_text(json.dumps(result, indent=2))
        return result
    page = fetch(share)
    (output/'baidu_verified_listing.html').write_text(page)
    owner = re.search(r'share_uk["\s]*[:=]\s*["\']?(\d+)', page)
    shareid = re.search(r'shareid["\s]*[:=]\s*["\']?(\d+)', page)
    if not owner or not shareid:
        raise ValueError('Verified page has no public share identifiers')
    listing = json.loads(fetch('https://pan.baidu.com/share/list?'+urlencode(
        dict(shareid=shareid[1], uk=owner[1], root='1', page='1', num='100', order='name', desc='0', web='1'))))
    (output/'baidu_file_listing.json').write_text(json.dumps(listing, indent=2, ensure_ascii=False))
    result.update(status='listing_received', listing_errno=listing.get('errno'), files=len(listing.get('list', [])))
    (output/'baidu_access_status.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    output = BASE/'data/downloads/3drealcar_raw'
    output.mkdir(parents=True, exist_ok=True)
    status = dict(source=URL, requested_vehicle='2024_06_04_13_44_39',
                  assets_downloaded=False, purpose='Inspect public listing before selecting a download')
    try:
        with urlopen(Request(URL, headers={'User-Agent': 'Mozilla/5.0'}), timeout=30) as response:
            data = response.read(10*1024*1024+1)
            if len(data) > 10*1024*1024:
                raise ValueError('Listing exceeds bounded response size')
            html = data.decode('utf-8', errors='replace')
        (output/'google_drive_listing.html').write_text(html)
        names = sorted(set(re.findall(r'2024[_0-9]+|[^<>"\\]{1,100}\.(?:zip|tar|7z|rar)', html)))
        status.update(status='listing_saved', candidate_names=names[:200],
                      requested_vehicle_mentioned=status['requested_vehicle'] in html)
    except Exception as exc:
        status.update(status='network_or_listing_failed', error=str(exc))
    alternatives = {
        'baidu': 'https://pan.baidu.com/s/10us5DLkTq124hVcyhsWyIg?pwd=pgmb',
        'official_page': 'https://xiaobiaodu.github.io/3drealcar/download.html',
    }
    status['alternative_sources'] = {}
    for name, url in alternatives.items():
        try:
            with urlopen(Request(url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=30) as response:
                page = response.read(1024*1024+1)
                if len(page) > 1024*1024:
                    raise ValueError('Alternative page exceeds 1MiB budget')
                (output/(name+'_listing.html')).write_bytes(page)
                status['alternative_sources'][name] = dict(status='page_saved', url=response.url, bytes=len(page))
        except Exception as exc:
            status['alternative_sources'][name] = dict(status='failed', error=str(exc))
    try:
        status['baidu_public_access'] = probe_baidu(output)
    except Exception as exc:
        status['baidu_public_access'] = dict(status='failed', error=str(exc))
    (output/'download_status.json').write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))
    return 0 if status['status'] == 'listing_saved' else 1


if __name__ == '__main__':
    raise SystemExit(main())
