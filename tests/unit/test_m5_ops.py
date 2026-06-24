"""Tests for M5 ops/integration: invariant sampling, Prometheus render, embedder benchmark."""

from __future__ import annotations

from parsimony.compress import FillerCompressor, LosslessCompressor
from parsimony.compress.sampling import VerifySampler
from parsimony.eval import BUILTIN_RETRIEVAL_SAMPLES, evaluate_retrieval
from parsimony.memory import HashingEmbedder
from parsimony.model import CanonicalRequest, Dialect, Message, Role, Span
from parsimony.obs import render_prometheus


def _req(text: str) -> CanonicalRequest:
    return CanonicalRequest(
        dialect=Dialect.ANTHROPIC,
        model="m",
        messages=(Message(role=Role.USER, spans=(Span(text),)),),
    )


class TestVerifySampler:
    def test_rate_one_always_verifies(self) -> None:
        s = VerifySampler(1.0)
        assert [s.should_verify() for _ in range(4)] == [True, True, True, True]

    def test_rate_zero_never_verifies(self) -> None:
        s = VerifySampler(0.0)
        assert [s.should_verify() for _ in range(4)] == [False, False, False, False]

    def test_rate_half_alternates(self) -> None:
        s = VerifySampler(0.5)
        assert [s.should_verify() for _ in range(4)] == [True, False, True, False]

    def test_clamps_out_of_range(self) -> None:
        assert VerifySampler(5.0).should_verify() is True
        assert VerifySampler(-1.0).should_verify() is False


class TestCompressorSampling:
    def test_lossless_skips_self_check_when_rate_zero(self) -> None:
        _, stats = LosslessCompressor(verify_sample=0.0).compress(_req("hi   \n\n\n\nthere"))
        assert stats[0].ok is None  # self-check skipped

    def test_filler_skips_self_check_when_rate_zero(self) -> None:
        _, stats = FillerCompressor(verify_sample=0.0).compress(_req("please fix this"))
        assert stats[0].ok is None


class TestRenderPrometheus:
    def test_emits_expected_families(self) -> None:
        snapshot = {
            "requests": 3,
            "cache_hits": 1,
            "cache_hit_rate": 0.3333,
            "tokens_before": 100,
            "tokens_after": 70,
            "tokens_saved": 30,
            "overall_ratio": 0.3,
            "stages": {
                "lossless": {
                    "tokens_before": 100,
                    "tokens_after": 80,
                    "tokens_saved": 20,
                    "reduction": 0.2,
                    "accuracy": 1.0,
                    "checked": 3,
                },
                "memory": {
                    "tokens_before": 120,
                    "tokens_after": 100,
                    "tokens_saved": 20,
                    "reduction": 0.1667,
                    "accuracy": None,
                    "checked": 0,
                },
            },
            "evals": {"retrieval": {"score": 0.95, "passed": True}},
        }
        text = render_prometheus(snapshot)
        assert "parsimony_requests_total 3" in text
        assert 'parsimony_stage_reduction_ratio{stage="lossless"} 0.2' in text
        assert 'parsimony_stage_accuracy_ratio{stage="lossless"} 1.0' in text
        # memory has no accuracy -> no accuracy series for it
        assert 'parsimony_stage_accuracy_ratio{stage="memory"}' not in text
        assert 'parsimony_eval_passed{kind="retrieval"} 1' in text
        assert "# TYPE parsimony_requests_total counter" in text

    def test_empty_snapshot_minimal(self) -> None:
        text = render_prometheus(
            {
                "requests": 0,
                "cache_hit_rate": 0.0,
                "tokens_saved": 0,
                "overall_ratio": 0.0,
                "stages": {},
                "evals": {},
            }
        )
        assert "parsimony_requests_total 0" in text


class TestEmbedderBenchmark:
    def test_retrieval_gate_runs_with_hashing_embedder(self) -> None:
        # The recall gate can be run with a semantic embedder (hashing here) vs lexical default.
        report = evaluate_retrieval(BUILTIN_RETRIEVAL_SAMPLES, embedder=HashingEmbedder())
        assert len(report.cases) == len(BUILTIN_RETRIEVAL_SAMPLES)
        assert 0.0 <= report.mean_score <= 1.0

    def test_lexical_default_still_passes(self) -> None:
        assert evaluate_retrieval(BUILTIN_RETRIEVAL_SAMPLES).passed
