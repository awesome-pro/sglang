"""Unit tests for the experimental HeteroSpec iteration-level trace.

CPU-only: the tracer deliberately imports nothing beyond the standard library, so
these run without a GPU. Registered with `register_cpu_ci` to match the other
speculative unit tests.
"""

import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sglang.srt.speculative import heterospec_trace
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_ENV_KEYS = (
    "SGLANG_HETEROSPEC_TRACE",
    "SGLANG_HETEROSPEC_TRACE_MAX_RECORDS",
    "SGLANG_HETEROSPEC_TRACE_FLUSH_EVERY",
)


def _reload(**env):
    """Re-import the tracer so its module-level env read is re-evaluated."""
    cleaned = {k: v for k, v in os.environ.items() if k not in _ENV_KEYS}
    cleaned.update({k: str(v) for k, v in env.items()})
    with mock.patch.dict(os.environ, cleaned, clear=True):
        return importlib.reload(heterospec_trace)


def _read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


class TestHeterospecTrace(unittest.TestCase):
    def setUp(self):
        # Guarantee a clean, disabled baseline for every test.
        self._module = _reload()

    def tearDown(self):
        self._module.close()
        _reload()

    def test_disabled_by_default(self):
        self.assertFalse(self._module.enabled())
        # Must be a silent no-op.
        self._module.record_iteration(active_k=3, rids=["a"], accepted=[1])
        self._module.flush()

    def test_records_expected_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            mod = _reload(SGLANG_HETEROSPEC_TRACE=path)
            self.assertTrue(mod.enabled())

            mod.record_iteration(active_k=5, rids=["r1", "r2"], accepted=[5, 1])
            mod.close()

            rows = _read(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["active_k"], 5)
            self.assertEqual(rows[0]["batch_size"], 2)
            self.assertEqual(rows[0]["iteration"], 0)
            self.assertEqual(
                rows[0]["requests"],
                [{"rid": "r1", "accepted": 5}, {"rid": "r2", "accepted": 1}],
            )

    def test_iteration_counter_increments(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            mod = _reload(SGLANG_HETEROSPEC_TRACE=path)
            for _ in range(3):
                mod.record_iteration(active_k=1, rids=["a"], accepted=[1])
            mod.close()
            self.assertEqual([r["iteration"] for r in _read(path)], [0, 1, 2])

    def test_record_cap_is_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            mod = _reload(
                SGLANG_HETEROSPEC_TRACE=path,
                SGLANG_HETEROSPEC_TRACE_MAX_RECORDS=2,
            )
            for _ in range(5):
                mod.record_iteration(active_k=1, rids=["a"], accepted=[1])
            mod.close()
            self.assertEqual(len(_read(path)), 2)

    def test_invalid_cap_falls_back_to_default(self):
        mod = _reload(
            SGLANG_HETEROSPEC_TRACE="/tmp/heterospec-should-not-exist.jsonl",
            SGLANG_HETEROSPEC_TRACE_MAX_RECORDS="nonsense",
        )
        self.assertEqual(mod._MAX_RECORDS, 200_000)
        mod.close()
        os.unlink("/tmp/heterospec-should-not-exist.jsonl")

    def test_mismatched_lengths_are_truncated_to_the_shorter(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            mod = _reload(SGLANG_HETEROSPEC_TRACE=path)
            mod.record_iteration(active_k=2, rids=["a", "b", "c"], accepted=[1])
            mod.close()
            rows = _read(path)
            self.assertEqual(rows[0]["batch_size"], 1)
            self.assertEqual([r["rid"] for r in rows[0]["requests"]], ["a"])

    def test_active_k_can_be_none(self):
        """A worker without `speculative_num_steps` must not break the trace."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            mod = _reload(SGLANG_HETEROSPEC_TRACE=path)
            mod.record_iteration(active_k=None, rids=["a"], accepted=[0])
            mod.close()
            self.assertIsNone(_read(path)[0]["active_k"])

    def test_file_is_line_delimited_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            mod = _reload(SGLANG_HETEROSPEC_TRACE=path)
            for i in range(4):
                mod.record_iteration(active_k=3, rids=[f"r{i}"], accepted=[i])
            mod.flush()
            for line in path.read_text().splitlines():
                self.assertIsInstance(json.loads(line), dict)
            mod.close()


if __name__ == "__main__":
    unittest.main()
