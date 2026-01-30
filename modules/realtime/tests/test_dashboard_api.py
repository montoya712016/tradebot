import unittest
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.realtime.dashboard_server import create_app


def _utc_iso():
    return datetime.now(timezone.utc).isoformat()


class DashboardApiTests(unittest.TestCase):
    def setUp(self):
        app, self.store = create_app(demo=False, refresh_sec=0.1)
        app.testing = True
        self.client = app.test_client()

    def test_health(self):
        resp = self.client.get('/api/health')
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.get_json().get('ok'))

    def test_config(self):
        resp = self.client.get('/api/config')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn('demo', data)
        self.assertIn('refresh_sec', data)
        self.assertEqual(data.get('server'), 'realtime_dashboard')

    def test_state_and_update(self):
        ts = _utc_iso()
        payload = {
            "summary": {
                "equity_usd": 1000.0,
                "cash_usd": 800.0,
                "exposure_usd": 200.0,
                "realized_pnl_usd": 10.0,
                "unrealized_pnl_usd": -2.0,
                "updated_at_utc": ts,
            },
            "positions": [],
            "recent_trades": [],
            "allocation": {},
            "meta": {
                "mode": "paper",
                "feed": {"last_ts_ms": 0, "ws_msgs": 0, "symbols": 0, "delay_sec": 0.0},
                "runtime": {
                    "stage": "idle",
                    "detail": "",
                    "stage_ts_utc": ts,
                    "last_cycle_sec": 0.5,
                    "last_cycle_proc_sec": 0.4,
                    "last_cycle_ts_utc": ts,
                    "last_cycle_symbols": 10,
                },
            },
        }
        resp = self.client.post('/api/update', json=payload)
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.get_json().get('ok'))

        state = self.client.get('/api/state').get_json()
        self.assertEqual(state['summary']['equity_usd'], 1000.0)
        self.assertEqual(state['meta']['mode'], 'paper')
        self.assertIn('runtime', state['meta'])
        self.assertEqual(state['meta']['runtime']['stage'], 'idle')
        self.assertIn('last_cycle_proc_sec', state['meta']['runtime'])


if __name__ == '__main__':
    unittest.main()
