"""Loopback-only authored test page and bounded JSON result collector."""
import argparse
import http.server
import json
import pathlib
import re
import threading

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=9087)
parser.add_argument('--output', type=pathlib.Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
page = pathlib.Path(__file__).resolve().parent
lock = threading.Lock()


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *values, **keywords):
        super().__init__(*values, directory=str(page), **keywords)

    def log_message(self, *_):
        pass

    def do_POST(self):
        match = re.fullmatch(r'/results/([A-Za-z0-9_-]{1,80})', self.path)
        length = self.headers.get('Content-Length', '')
        if not match or not length.isdigit() or not 0 < int(length) <= 524288:
            self.send_error(400)
            return
        try:
            value = json.loads(self.rfile.read(int(length)))
            if not isinstance(value, dict) or value.get('run') != match[1]:
                raise ValueError('Invalid report identity')
            target = args.output / (match[1] + '.json')
            with lock:
                temporary = target.with_suffix('.new')
                temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
                temporary.replace(target)
        except (ValueError, UnicodeError):
            self.send_error(400)
            return
        self.send_response(204)
        self.end_headers()


http.server.ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
