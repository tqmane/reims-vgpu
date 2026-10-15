"""Collector trust boundaries, without a browser or GPU."""
import http.client
import json
import pathlib
import socket
import subprocess
import sys
import tempfile
import time
import unittest


class Collector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = pathlib.Path(__file__).resolve().parents[2]
        scratch = cls.root / '.work' / 'diagnostics'
        scratch.mkdir(parents=True, exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(prefix='webgl-collector-', dir=scratch)
        cls.output = pathlib.Path(cls.temp.name)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            cls.port = sock.getsockname()[1]
        cls.server = subprocess.Popen([
            sys.executable, str(pathlib.Path(__file__).with_name('server.py')),
            '--output', str(cls.output), '--port', str(cls.port)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):
            if cls.server.poll() is not None:
                raise RuntimeError('Collector exited during startup')
            try:
                with socket.create_connection(('127.0.0.1', cls.port), timeout=.1):
                    return
            except OSError:
                time.sleep(.02)
        cls.tearDownClass()
        raise RuntimeError('Collector startup timed out')

    @classmethod
    def tearDownClass(cls):
        cls.server.terminate()
        cls.server.wait(timeout=5)
        assert cls.output.resolve().is_relative_to(cls.root.resolve())
        cls.temp.cleanup()

    def post(self, path, body, length=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=2)
        try:
            headers = {'Content-Length': str(len(body) if length is None else length)}
            connection.request('POST', path, body, headers)
            response = connection.getresponse()
            response.read()
            return response.status
        finally:
            connection.close()

    def test_matching_report_is_saved(self):
        report = {'run': 'valid', 'passed': False}
        self.assertEqual(self.post('/results/valid', json.dumps(report).encode()), 204)
        self.assertEqual(json.loads((self.output / 'valid.json').read_text()), report)

    def test_invalid_inputs_write_no_report(self):
        for path, body, length in [
            ('/results/../escape', b'{}', None),
            ('/results/bad', b'{', None),
            ('/results/bad', b'\xff', None),
            ('/results/bad', b'[]', None),
            ('/results/bad', b'{"run":"different"}', None),
            ('/results/bad', b'', 524289),
            ('/results/bad', b'', 0),
        ]:
            with self.subTest(path=path, body=body, length=length):
                self.assertEqual(self.post(path, body, length), 400)
        self.assertFalse((self.output / 'bad.json').exists())
        self.assertFalse((self.output.parent / 'escape.json').exists())


if __name__ == '__main__':
    unittest.main()
