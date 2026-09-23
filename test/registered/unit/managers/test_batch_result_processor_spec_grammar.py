"""Unit tests for Spec V2 grammar truncation in _resolve_spec_v2_tokens.

The grammar-constrained spec path stops accepting at the grammar-terminating
token, so the over-drafted suffix is never committed to KV nor emitted.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.disaggregation.decode import DecodeRequest, DecodeTransferQueue
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.sampling.sampling_params import (
    REQUEST_REASONING_END_TOKEN_IDS_KEY,
    SamplingParams,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _FakeGrammar:
    """Grammar stub that terminates after `terminate_after` accepted tokens."""

    def __init__(self, terminate_after: int):
        self.accepted = []
        self.finished = False
        self._terminate_after = terminate_after

    def accept_token(self, token_id: int):
        self.accepted.append(token_id)

    def is_terminated(self) -> bool:
        return len(self.accepted) >= self._terminate_after


class _FakeSpecAlgorithm:
    def is_none(self) -> bool:
        return False

    def is_dflash(self) -> bool:
        return False


class _FakeForwardMode:
    def is_decode(self) -> bool:
        return True

    def is_extend(self) -> bool:
        return False


class _FakeBatch:
    def __init__(self, reqs):
        self.reqs = reqs
        self.has_grammar = any(req.grammar is not None for req in reqs)
        self.forward_mode = _FakeForwardMode()
        self.spec_algorithm = _FakeSpecAlgorithm()


def _make_processor(model_worker=None) -> SchedulerBatchResultProcessor:
    return SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=None,
        enable_overlap=False,
        enable_overlap_mlx=False,
        model_config=SimpleNamespace(think_end_ids=None),
        token_to_kv_pool_allocator=None,
        tree_cache=None,
        hisparse_coordinator=None,
        req_to_token_pool=None,
        decode_offload_manager=None,
        metrics_collector=None,
        metrics_reporter=SimpleNamespace(),
        draft_worker=None,
        model_worker=model_worker
        or SimpleNamespace(on_verify_complete_cpu=lambda *a, **k: None),
        logprob_result_processor=None,
        output_streamer=SimpleNamespace(),
        beam_coordinator=SimpleNamespace(),
        abort_request=lambda *a, **k: None,
    )


def _make_req(terminate_after: int, rid: str = "r0") -> Req:
    sp = SamplingParams(max_new_tokens=256, temperature=0)
    sp.normalize(None)
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=[1, 2, 3],
        sampling_params=sp,
    )
    if terminate_after is not None:
        req.grammar = _FakeGrammar(terminate_after=terminate_after)
    req.kv.kv_committed_len = 0
    return req


def _make_result(num_draft_tokens, accept_lens, flat_tokens):
    return GenerationBatchResult(
        next_token_ids=torch.tensor(flat_tokens, dtype=torch.long),
        accept_lens=torch.tensor(accept_lens, dtype=torch.long),
        speculative_num_draft_tokens=num_draft_tokens,
    )


def _commit_disagg_handoff(
    req: Req,
    processor: SchedulerBatchResultProcessor,
    token_id: int,
    *,
    replayed_boundary: bool = False,
) -> None:
    queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
    queue.scheduler = SimpleNamespace(
        batch_result_processor=processor, kv_checksum_computer=None
    )
    queue.spec_algorithm = SimpleNamespace(is_none=lambda: True)
    queue.metadata_buffers = SimpleNamespace(
        get_buf=lambda _: (
            torch.tensor([token_id], dtype=torch.long),
            torch.zeros(7, dtype=torch.long),
            torch.zeros(1),
            torch.zeros(1, dtype=torch.long),
            torch.zeros(1),
            torch.zeros(1, dtype=torch.long),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            torch.tensor([1], dtype=torch.long),
        )
    )
    req.bootstrap_host = "127.0.0.1"
    req.bootstrap_room = 1
    if replayed_boundary:
        req.pd_rebootstrap_forced_output_id = token_id
    decode_req = DecodeRequest(
        req=req,
        kv_receiver=SimpleNamespace(clear=lambda: None),
        metadata_buffer_index=0,
        is_rebootstrap=replayed_boundary,
    )

    queue._commit_transfer_to_req(decode_req)


class TestSpecV2GrammarTruncation(CustomTestCase):
    def test_resolve_truncates_after_grammar_completion(self):
        req = _make_req(terminate_after=2)
        proc = _make_processor()
        # stride=4, accept_len=3 -> proposed [101, 102, 103]; grammar finishes at 102.
        result = _make_result(4, [3], [101, 102, 103, 0])

        predict_tokens = proc._resolve_spec_v2_tokens(result, _FakeBatch([req]))

        self.assertEqual(predict_tokens, [[101, 102]])
        # No pre-claim: commit the full retained run (no -1 refund).
        self.assertEqual(req.kv.kv_committed_len, 2)

    def test_resolve_keeps_all_when_grammar_not_terminated(self):
        req = _make_req(terminate_after=99)
        proc = _make_processor()
        result = _make_result(4, [3], [201, 202, 203, 0])

        predict_tokens = proc._resolve_spec_v2_tokens(result, _FakeBatch([req]))

        self.assertEqual(predict_tokens, [[201, 202, 203]])
        self.assertEqual(req.kv.kv_committed_len, 3)


class TestReasoningTokenAccounting(CustomTestCase):
    def test_multi_token_end_can_span_decode_steps(self):
        req = _make_req(terminate_after=99)
        req.require_reasoning = True
        processor = _make_processor()
        processor.model_config.think_end_ids = [7, 8]

        processor._maybe_update_reasoning_tokens(req, [10, 7])
        processor._maybe_update_reasoning_tokens(req, [8, 11])

        self.assertEqual(req.reasoning_tokens, 3)
        self.assertTrue(req._is_reasoning_over)

    def test_request_selected_end_ignores_other_closer(self):
        req = _make_req(terminate_after=99)
        req.require_reasoning = True
        req.sampling_params.custom_params = {
            REQUEST_REASONING_END_TOKEN_IDS_KEY: [17, 18]
        }
        processor = _make_processor()
        processor.model_config.think_end_ids = [7, 8]
        processor.model_config.request_selectable_think_end_id_sequences = [
            [7, 8],
            [17, 18],
        ]

        # The global/default closer must not end a medium request.
        processor._maybe_update_reasoning_tokens(req, [10, 7])
        processor._maybe_update_reasoning_tokens(req, [8, 11])
        self.assertFalse(req._is_reasoning_over)

        processor._maybe_update_reasoning_tokens(req, [10, 17])
        processor._maybe_update_reasoning_tokens(req, [18, 11])

        self.assertEqual(req.reasoning_tokens, 7)
        self.assertTrue(req._is_reasoning_over)

    def test_disagg_handoff_can_start_multi_token_selected_end(self):
        req = _make_req(terminate_after=99)
        req.require_reasoning = True
        req.sampling_params.custom_params = {
            REQUEST_REASONING_END_TOKEN_IDS_KEY: [17, 18]
        }
        processor = _make_processor()
        processor.model_config.request_selectable_think_end_id_sequences = [
            [7, 8],
            [17, 18],
        ]

        _commit_disagg_handoff(req, processor, 17)
        self.assertEqual(req.reasoning_tokens, 1)
        self.assertFalse(req._is_reasoning_over)

        processor._maybe_update_reasoning_tokens(req, 18)

        self.assertEqual(req.reasoning_tokens, 2)
        self.assertTrue(req._is_reasoning_over)

    def test_disagg_rebootstrap_does_not_recount_boundary(self):
        req = _make_req(terminate_after=99)
        req.require_reasoning = True
        req.sampling_params.custom_params = {REQUEST_REASONING_END_TOKEN_IDS_KEY: [17]}
        processor = _make_processor()
        processor.model_config.request_selectable_think_end_id_sequences = [
            [7],
            [17],
        ]

        _commit_disagg_handoff(req, processor, 17, replayed_boundary=True)

        self.assertEqual(req.reasoning_tokens, 0)
        self.assertFalse(req._is_reasoning_over)


if __name__ == "__main__":
    unittest.main()


class _CapturingWorker:
    """Records what the result processor hands to the spec worker."""

    def __init__(self):
        self.calls = []

    def on_verify_complete_cpu(self, num_correct_drafts_per_req, **kwargs):
        self.calls.append((list(num_correct_drafts_per_req), dict(kwargs)))


class TestRequestIdentityFeedback(CustomTestCase):
    """The verify-feedback call must carry request identity, correctly aligned.

    A policy that keeps request-local state indexes the id list by the same
    position as the count list. If the two were misaligned, acceptance would be
    attributed to the wrong request -- silently, and permanently, since there is
    nothing in the payload to reveal it.
    """

    def test_request_ids_match_batch_reqs_order(self):
        worker = _CapturingWorker()
        proc = _make_processor(model_worker=worker)
        reqs = [
            _make_req(None, rid="alpha"),
            _make_req(None, rid="beta"),
            _make_req(None, rid="gamma"),
        ]
        # stride 4, accept_lens [3, 2, 1] -> 3 * 4 flat tokens
        result = _make_result(4, [3, 2, 1], [101, 102, 103, 0,
                                             201, 202, 0, 0,
                                             301, 0, 0, 0])

        proc._resolve_spec_v2_tokens(result, _FakeBatch(reqs))

        self.assertEqual(len(worker.calls), 1)
        counts, kwargs = worker.calls[0]
        self.assertEqual(kwargs["request_ids"], ["alpha", "beta", "gamma"])
        self.assertEqual(kwargs["batch_size"], 3)
        self.assertEqual(len(kwargs["request_ids"]), len(counts))

    def test_corrected_draft_counts_exclude_the_bonus_token(self):
        """Pins the semantics a request-local estimator is built on."""
        worker = _CapturingWorker()
        proc = _make_processor(model_worker=worker)
        req = _make_req(None, rid="solo")
        # accept_len 3 with num_non_draft 1 -> 2 accepted drafts
        result = _make_result(4, [3], [101, 102, 103, 0])

        proc._resolve_spec_v2_tokens(result, _FakeBatch([req]))

        counts, kwargs = worker.calls[0]
        self.assertEqual(counts, [2])
        self.assertEqual(kwargs["request_ids"], ["solo"])

    def test_alignment_holds_with_grammar_truncation(self):
        """Grammar truncation must not desynchronise ids from counts."""
        worker = _CapturingWorker()
        proc = _make_processor(model_worker=worker)
        reqs = [_make_req(terminate_after=2, rid="g1"),
                _make_req(terminate_after=99, rid="g2")]
        result = _make_result(4, [3, 3], [101, 102, 103, 0,
                                          201, 202, 203, 0])

        proc._resolve_spec_v2_tokens(result, _FakeBatch(reqs))

        counts, kwargs = worker.calls[0]
        self.assertEqual(kwargs["request_ids"], ["g1", "g2"])
        self.assertEqual(len(kwargs["request_ids"]), len(counts))

    def test_single_request_batch_still_passes_ids(self):
        worker = _CapturingWorker()
        proc = _make_processor(model_worker=worker)
        proc._resolve_spec_v2_tokens(
            _make_result(4, [1], [101, 0, 0, 0]),
            _FakeBatch([_make_req(None, rid="only")]),
        )
        _, kwargs = worker.calls[0]
        self.assertEqual(kwargs["request_ids"], ["only"])
