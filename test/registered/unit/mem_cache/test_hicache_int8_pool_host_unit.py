"""Unit tests for the INT8 HiCache host pool (``MHATokenToKVPoolHostINT8``).

Requires CUDA/ROCm: the pool moves bytes through SGLang's JIT HiCache kernels and
pins its arena with the CUDA driver. Run on the pod:

    python -m pytest test/registered/unit/mem_cache/test_hicache_int8_pool_host_unit.py -v

The parts that need no GPU (record codec, staging geometry, growth policy) are
covered by ``test_hicache_int8_codec.py`` and run anywhere.
"""

import os
import unittest
from unittest import mock

import torch

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.pool_host.mha import (
    MHATokenToKVPoolHost,
    get_mha_host_pool_cls,
)
from sglang.srt.mem_cache.pool_host.mha_int8 import MHATokenToKVPoolHostINT8
from sglang.srt.mem_cache.pool_host import int8_codec as codec
from sglang.srt.utils import is_cuda, is_hip, is_npu, is_xpu
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=20, stage="jit-kernel-unit", runner_config="amd")

import pytest

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or is_npu() or is_xpu() or not (is_cuda() or is_hip()),
    reason="INT8 HiCache host pool requires CUDA/ROCm.",
)

DEVICE = "cuda"
LAYER_NUM = 2
HEAD_NUM = 8  # must satisfy codec.check_layout: head_num * head_dim == 1024
HEAD_DIM = 128
POOL_SIZE = 64
PAGE_SIZE = 1

#: The exact bound the format guarantees (see int8_codec docstring).
def error_bound(restored, scales):
    s = scales.float().unsqueeze(-1)
    return (0.5 + 2**-8) * s + 2**-8 * restored.float().abs()


def _make_device_pool(layer_num=LAYER_NUM, size=POOL_SIZE):
    return MHATokenToKVPool(
        size=size,
        page_size=PAGE_SIZE,
        head_num=HEAD_NUM,
        head_dim=HEAD_DIM,
        dtype=torch.bfloat16,
        layer_num=layer_num,
        device=DEVICE,
        enable_memory_saver=False,
    )


def _make_host_pool(device_pool, *, host_size=0, ratio=2.0, layout="layer_first", **kw):
    return MHATokenToKVPoolHostINT8(
        device_pool,
        host_to_device_ratio=ratio,
        host_size=host_size,
        page_size=PAGE_SIZE,
        layout=layout,
        pin_memory=True,
        device="cpu",
        allocator_type="default",
        **kw,
    )


def _fill(device_pool, *, seed=0, num_layers=None):
    """Fill device K/V with distinguishable, non-degenerate values."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    layers = num_layers if num_layers is not None else device_pool.layer_num
    for layer in range(layers):
        for buf, tag in ((device_pool.k_buffer[layer], 0), (device_pool.v_buffer[layer], 1)):
            shape = buf.shape
            values = torch.randn(shape, generator=g) * (1.0 + layer) + tag
            buf.copy_(values.to(torch.bfloat16))


class TestSizing(unittest.TestCase):
    def test_size_per_token_is_the_encoded_size(self):
        device_pool = _make_device_pool()
        host_pool = _make_host_pool(device_pool)
        self.assertEqual(host_pool.size_per_token, codec.bytes_per_token(LAYER_NUM))
        self.assertEqual(host_pool.size_per_token, 2 * LAYER_NUM * 1152)
        # Baseline for this geometry: 2 (K,V) * layers * heads * dim * 2 bytes.
        baseline = 2 * LAYER_NUM * HEAD_NUM * HEAD_DIM * 2
        self.assertEqual(baseline, 4096 * LAYER_NUM)
        self.assertLess(host_pool.size_per_token, baseline)
        self.assertAlmostEqual(baseline / host_pool.size_per_token, 1.7777778, places=6)
        host_pool.destroy()

    def test_fixed_host_size_yields_more_tokens_than_bf16(self):
        """The headline claim: same --hicache-size, ~1.78x the token capacity."""
        device_pool = _make_device_pool()
        int8_pool = _make_host_pool(device_pool, host_size=1)
        bf16_pool = MHATokenToKVPoolHost(
            device_pool,
            host_to_device_ratio=2.0,
            host_size=1,
            page_size=PAGE_SIZE,
            layout="layer_first",
            pin_memory=True,
            device="cpu",
        )
        self.assertEqual(bf16_pool.size_per_token, 4096 * LAYER_NUM)
        self.assertEqual(int8_pool.size_per_token, 2304 * LAYER_NUM)
        # Ratio holds to within the +1 page slack.
        ratio = int8_pool.size / bf16_pool.size
        self.assertAlmostEqual(ratio, 4096 / 2304, delta=0.01)
        int8_pool.destroy()
        bf16_pool.destroy()

    def test_arena_shape_is_layer_first_encoded(self):
        device_pool = _make_device_pool()
        host_pool = _make_host_pool(device_pool)
        self.assertEqual(
            host_pool.kv_buffer.shape,
            (2, LAYER_NUM, host_pool.size, codec.ROW_BYTES),
        )
        self.assertEqual(host_pool.kv_buffer.dtype, torch.uint8)
        # k_buffer/v_buffer are the two halves.
        self.assertEqual(host_pool.k_buffer.shape, (LAYER_NUM, host_pool.size, codec.ROW_BYTES))
        host_pool.destroy()

    def test_layer_refs_are_contiguous_encoded_rows(self):
        device_pool = _make_device_pool()
        host_pool = _make_host_pool(device_pool)
        for layer in range(LAYER_NUM):
            ref = host_pool.k_data_refs[layer]
            self.assertTrue(ref.is_contiguous())
            self.assertEqual(ref.shape[-1], codec.ROW_BYTES)
            self.assertEqual(ref.data_ptr() % codec.ALIGNMENT_BYTES, 0)
        host_pool.destroy()


class TestTransferRoundtrip(unittest.TestCase):
    def test_d2h_then_h2d_reconstructs_within_bound(self):
        device_pool = _make_device_pool()
        host_pool = _make_host_pool(device_pool)
        _fill(device_pool)

        origin = {
            layer: (
                device_pool.k_buffer[layer].clone(),
                device_pool.v_buffer[layer].clone(),
            )
            for layer in range(LAYER_NUM)
        }

        src = torch.tensor([1, 5, 9, 20, 33], device=DEVICE, dtype=torch.int64)
        dst = torch.tensor([0, 1, 2, 3, 4], device=DEVICE, dtype=torch.int64)

        host_pool.backup_from_device_all_layer(device_pool, dst, src, "kernel")
        torch.cuda.synchronize()

        # Wipe the device rows so a stale read cannot pass the test.
        for layer in range(LAYER_NUM):
            device_pool.k_buffer[layer].zero_()
            device_pool.v_buffer[layer].zero_()

        for layer in range(LAYER_NUM):
            host_pool.load_to_device_per_layer(
                device_pool, dst, src, layer, "kernel"
            )
        torch.cuda.synchronize()

        for layer in range(LAYER_NUM):
            for buf, orig, label in (
                (device_pool.k_buffer[layer], origin[layer][0], "K"),
                (device_pool.v_buffer[layer], origin[layer][1], "V"),
            ):
                got = buf[src].float()
                want = orig[src].float()
                scales = codec.compute_scales(
                    orig[src].reshape(len(src), HEAD_NUM, HEAD_DIM)
                )
                bound = error_bound(got.reshape(len(src), HEAD_NUM, HEAD_DIM), scales)
                err = (got.reshape(len(src), HEAD_NUM, HEAD_DIM) - want.reshape(len(src), HEAD_NUM, HEAD_DIM)).abs()
                violations = int((err > bound).sum())
                self.assertEqual(
                    violations, 0, f"layer {layer} {label}: {violations} exceed bound"
                )

    def test_encoded_bytes_actually_shrink(self):
        """Prove the host arena holds compressed data, not raw BF16."""
        device_pool = _make_device_pool()
        host_pool = _make_host_pool(device_pool)
        _fill(device_pool)
        src = torch.tensor([2, 4, 6], device=DEVICE, dtype=torch.int64)
        dst = torch.tensor([10, 11, 12], device=DEVICE, dtype=torch.int64)
        host_pool.backup_from_device_all_layer(device_pool, dst, src, "kernel")
        torch.cuda.synchronize()

        raw = device_pool.k_buffer[0][src]
        stored = host_pool.k_data_refs[0][dst]
        self.assertEqual(stored.dtype, torch.uint8)
        # 1152 encoded bytes per row against 2048 raw bytes.
        self.assertEqual(stored.numel(), len(src) * codec.ROW_BYTES)
        self.assertLess(stored.numel(), raw.numel())
        # Decoding the arena must reproduce the device row within bound, which
        # proves it really is an encoding rather than padding.
        decoded = codec.decode_records(
            stored, head_num=HEAD_NUM, head_dim=HEAD_DIM, dtype=torch.bfloat16
        )
        scales = codec.compute_scales(raw)
        bound = error_bound(decoded, scales)
        err = (decoded.float() - raw.float()).abs()
        self.assertEqual(int((err > bound).sum()), 0)

    def test_padding_bytes_are_zero_in_the_arena(self):
        device_pool = _make_device_pool()
        host_pool = _make_host_pool(device_pool)
        _fill(device_pool)
        src = torch.tensor([1, 2], device=DEVICE, dtype=torch.int64)
        dst = torch.tensor([0, 1], device=DEVICE, dtype=torch.int64)
        host_pool.backup_from_device_all_layer(device_pool, dst, src, "kernel")
        torch.cuda.synchronize()
        padding = host_pool.k_data_refs[0][dst][:, codec.PAYLOAD_BYTES + codec.SCALE_BYTES :]
        self.assertTrue(bool((padding == 0).all()))

    def test_all_zero_kv_round_trips_exactly(self):
        device_pool = _make_device_pool()
        host_pool = _make_host_pool(device_pool)
        for layer in range(LAYER_NUM):
            device_pool.k_buffer[layer].zero_()
            device_pool.v_buffer[layer].zero_()
        src = torch.tensor([3, 7], device=DEVICE, dtype=torch.int64)
        dst = torch.tensor([0, 1], device=DEVICE, dtype=torch.int64)
        host_pool.backup_from_device_all_layer(device_pool, dst, src, "kernel")
        torch.cuda.synchronize()
        for layer in range(LAYER_NUM):
            device_pool.k_buffer[layer].fill_(float("nan"))
            host_pool.load_to_device_per_layer(device_pool, dst, src, layer, "kernel")
        torch.cuda.synchronize()
        for layer in range(LAYER_NUM):
            got = device_pool.k_buffer[layer][src]
            self.assertTrue(bool((got == 0).all()), f"layer {layer} not exactly zero")

    def test_multi_layer_pool_round_trips(self):
        """The all-layer mover must handle more than the 2-layer unit case."""
        device_pool = _make_device_pool(layer_num=36, size=128)
        host_pool = _make_host_pool(device_pool)
        _fill(device_pool, num_layers=36, seed=3)
        src = torch.tensor([1, 2, 3, 4], device=DEVICE, dtype=torch.int64)
        dst = torch.tensor([0, 1, 2, 3], device=DEVICE, dtype=torch.int64)
        host_pool.backup_from_device_all_layer(device_pool, dst, src, "kernel")
        torch.cuda.synchronize()
        for layer in range(36):
            device_pool.k_buffer[layer].zero_()
        for layer in range(36):
            host_pool.load_to_device_per_layer(device_pool, dst, src, layer, "kernel")
        torch.cuda.synchronize()
        for layer in range(36):
            row = device_pool.k_buffer[layer][src].float()
            self.assertTrue(bool(torch.isfinite(row).all()), f"layer {layer}")
            self.assertGreater(float(row.abs().max()), 0.0, f"layer {layer} is empty")

    def test_d2h_then_h2d_with_disjoint_index_ranges(self):
        device_pool = _make_device_pool()
        host_pool = _make_host_pool(device_pool)
        _fill(device_pool)
        src = torch.tensor([0, 1, 2], device=DEVICE, dtype=torch.int64)
        dst = torch.tensor([40, 41, 42], device=DEVICE, dtype=torch.int64)
        host_pool.backup_from_device_all_layer(device_pool, dst, src, "kernel")
        torch.cuda.synchronize()
        for layer in range(LAYER_NUM):
            device_pool.k_buffer[layer].zero_()
        for layer in range(LAYER_NUM):
            host_pool.load_to_device_per_layer(device_pool, dst, src, layer, "kernel")
        torch.cuda.synchronize()
        self.assertGreater(float(device_pool.k_buffer[0][src].float().abs().max()), 0.0)


class TestStagingGrowth(unittest.TestCase):
    def test_large_transfer_grows_staging_and_still_round_trips(self):
        device_pool = _make_device_pool(size=4096)
        host_pool = _make_host_pool(device_pool)
        initial_capacity = host_pool._d2h.capacity
        _fill(device_pool)
        count = initial_capacity + 100  # force a growth
        src = torch.arange(count, device=DEVICE, dtype=torch.int64)
        dst = torch.arange(count, device=DEVICE, dtype=torch.int64)
        host_pool.backup_from_device_all_layer(device_pool, dst, src, "kernel")
        torch.cuda.synchronize()
        self.assertGreater(host_pool._d2h.capacity, initial_capacity)
        self.assertEqual(host_pool._d2h.capacity, host_pool._h2d.capacity)
        for layer in range(LAYER_NUM):
            device_pool.k_buffer[layer].zero_()
        for layer in range(LAYER_NUM):
            host_pool.load_to_device_per_layer(device_pool, dst, src, layer, "kernel")
        torch.cuda.synchronize()
        self.assertGreater(float(device_pool.k_buffer[0][src].float().abs().max()), 0.0)

    def test_pointer_tables_follow_growth(self):
        device_pool = _make_device_pool(size=4096)
        host_pool = _make_host_pool(device_pool)
        count = host_pool._d2h.capacity + 1
        src = torch.arange(count, device=DEVICE, dtype=torch.int64)
        dst = torch.arange(count, device=DEVICE, dtype=torch.int64)
        host_pool.backup_from_device_all_layer(device_pool, dst, src, "kernel")
        torch.cuda.synchronize()
        expected = [t.data_ptr() for t in host_pool._d2h.k_layer_views(count)]
        self.assertEqual([int(v) for v in host_pool._d2h_k_src_ptrs], expected)


class TestAllocFree(unittest.TestCase):
    def test_alloc_free_reuse(self):
        device_pool = _make_device_pool()
        host_pool = _make_host_pool(device_pool)
        first = host_pool.alloc(PAGE_SIZE * 4)
        self.assertIsNotNone(first)
        self.assertEqual(host_pool.free(first), 4)
        second = host_pool.alloc(PAGE_SIZE * 4)
        self.assertEqual(sorted(second.tolist()), sorted(first.tolist()))
        host_pool.destroy()

    def test_double_free_is_detected(self):
        device_pool = _make_device_pool()
        host_pool = _make_host_pool(device_pool)
        slots = host_pool.alloc(PAGE_SIZE * 2)
        host_pool.free(slots)
        with self.assertRaises(AssertionError):
            host_pool.free(slots)
        host_pool.destroy()

    def test_alloc_beyond_capacity_returns_none(self):
        device_pool = _make_device_pool()
        host_pool = _make_host_pool(device_pool)
        self.assertIsNone(host_pool.alloc(host_pool.logical_size + PAGE_SIZE))
        host_pool.destroy()


class TestStoragePages(unittest.TestCase):
    def test_data_page_round_trip(self):
        device_pool = _make_device_pool()
        host_pool = _make_host_pool(device_pool)
        _fill(device_pool)
        src = torch.tensor([5], device=DEVICE, dtype=torch.int64)
        dst = torch.tensor([0], device=DEVICE, dtype=torch.int64)
        host_pool.backup_from_device_all_layer(device_pool, dst, src, "kernel")
        torch.cuda.synchronize()

        page = host_pool.get_data_page(0, flat=True)
        self.assertEqual(page.numel(), 2 * LAYER_NUM * codec.ROW_BYTES)
        self.assertEqual(page.dtype, torch.uint8)

        host_pool.set_from_flat_data_page(1, page)
        self.assertTrue(
            torch.equal(
                host_pool.get_data_page(1, flat=True), host_pool.get_data_page(0, flat=True)
            )
        )
        host_pool.destroy()

    def test_dummy_page_has_encoded_size(self):
        device_pool = _make_device_pool()
        host_pool = _make_host_pool(device_pool)
        dummy = host_pool.get_dummy_flat_data_page()
        self.assertEqual(dummy.numel(), 2 * LAYER_NUM * codec.ROW_BYTES)
        self.assertTrue(bool((dummy == 0).all()))
        host_pool.destroy()

    def test_l3_meta_is_rejected(self):
        device_pool = _make_device_pool()
        host_pool = _make_host_pool(device_pool)
        for call in (
            lambda: host_pool.get_page_buffer_meta(torch.tensor([0])),
            lambda: host_pool.get_split_heads_page_buffer_meta(torch.tensor([0]), 2),
        ):
            with self.assertRaises(NotImplementedError):
                call()
        host_pool.destroy()


class TestFailFast(unittest.TestCase):
    """Every unsupported configuration must raise at construction, not corrupt
    generations later."""

    def test_rejects_non_layer_first_layout(self):
        device_pool = _make_device_pool()
        with self.assertRaises(NotImplementedError) as ctx:
            _make_host_pool(device_pool, layout="page_first")
        self.assertIn("layer_first", str(ctx.exception))

    def test_rejects_page_size_above_one(self):
        device_pool = _make_device_pool()
        with self.assertRaises(NotImplementedError) as ctx:
            MHATokenToKVPoolHostINT8(
                device_pool,
                host_to_device_ratio=2.0,
                host_size=0,
                page_size=16,
                layout="layer_first",
                pin_memory=True,
                device="cpu",
            )
        self.assertIn("page-size", str(ctx.exception))

    def test_rejects_quantized_device_pool(self):
        device_pool = _make_device_pool()
        with mock.patch.object(
            type(device_pool), "is_quantized_kv_cache", property(lambda self: True)
        ):
            with self.assertRaises(NotImplementedError) as ctx:
                _make_host_pool(device_pool)
        self.assertIn("quantized", str(ctx.exception))

    def test_rejects_mtp_draft_pools(self):
        device_pool = _make_device_pool()
        with self.assertRaises(NotImplementedError) as ctx:
            _make_host_pool(device_pool, mtp_draft_device_pools=(device_pool,))
        self.assertIn("MTP", str(ctx.exception))

    def test_rejects_asymmetric_head_dims(self):
        device_pool = _make_device_pool()
        with mock.patch.object(
            type(device_pool), "v_head_dim", property(lambda self: 64)
        ):
            with self.assertRaises(NotImplementedError) as ctx:
                _make_host_pool(device_pool)
        self.assertIn("symmetric", str(ctx.exception))

    def test_rejects_wrong_head_geometry(self):
        """head_num * head_dim must equal the 1024-byte payload exactly."""
        pool = MHATokenToKVPool(
            size=POOL_SIZE,
            page_size=PAGE_SIZE,
            head_num=4,  # 4 * 128 = 512 != 1024
            head_dim=HEAD_DIM,
            dtype=torch.bfloat16,
            layer_num=LAYER_NUM,
            device=DEVICE,
            enable_memory_saver=False,
        )
        with self.assertRaises(ValueError):
            _make_host_pool(pool)

    def test_rejects_non_two_byte_dtype(self):
        pool = MHATokenToKVPool(
            size=POOL_SIZE,
            page_size=PAGE_SIZE,
            head_num=HEAD_NUM,
            head_dim=HEAD_DIM,
            dtype=torch.float32,
            layer_num=LAYER_NUM,
            device=DEVICE,
            enable_memory_saver=False,
        )
        with self.assertRaises((NotImplementedError, ValueError)):
            _make_host_pool(pool)

    def test_rejects_unknown_io_backend_at_transfer_time(self):
        device_pool = _make_device_pool()
        host_pool = _make_host_pool(device_pool)
        src = torch.tensor([1], device=DEVICE, dtype=torch.int64)
        dst = torch.tensor([0], device=DEVICE, dtype=torch.int64)
        with self.assertRaises(NotImplementedError):
            host_pool.backup_from_device_all_layer(device_pool, dst, src, "direct")
        with self.assertRaises(NotImplementedError):
            host_pool.load_to_device_per_layer(device_pool, dst, src, 0, "direct")
        host_pool.destroy()


class TestDispatch(unittest.TestCase):
    def test_env_flag_selects_int8_pool(self):
        device_pool = _make_device_pool()
        with mock.patch.dict(
            "os.environ", {"SGLANG_EXPERIMENTAL_HICACHE_INT8": "1"}
        ):
            self.assertIs(get_mha_host_pool_cls(device_pool), MHATokenToKVPoolHostINT8)
        with mock.patch.dict("os.environ", {}, clear=False):
            os.environ.pop("SGLANG_EXPERIMENTAL_HICACHE_INT8", None)
            self.assertIs(get_mha_host_pool_cls(device_pool), MHATokenToKVPoolHost)

    def test_flag_defaults_to_off(self):
        device_pool = _make_device_pool()
        saved = os.environ.pop("SGLANG_EXPERIMENTAL_HICACHE_INT8", None)
        try:
            self.assertIs(get_mha_host_pool_cls(device_pool), MHATokenToKVPoolHost)
        finally:
            if saved is not None:
                os.environ["SGLANG_EXPERIMENTAL_HICACHE_INT8"] = saved


if __name__ == "__main__":
    unittest.main()
