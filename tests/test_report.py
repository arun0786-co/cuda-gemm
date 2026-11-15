"""The CUDA kernels are tested on the GPU by src/test.cu (every kernel against cuBLAS). These
tests cover the Python side: the peak numbers the report divides by, and the report itself."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
import report  # noqa: E402


@pytest.mark.parametrize("gpu,precision,mhz,whitepaper_tflops", [
    # NVIDIA Turing whitepaper, Table 1, at the reference boost clocks
    ("rtx2080ti", "fp32", 1545, 13.4),
    ("gtx1080ti", "fp32", 1582, 11.3),
    ("rtx2080ti", "fp16", 1545, 53.8),  # tensor cores, FP32 accumulate
])
def test_peak_matches_the_whitepaper(gpu, precision, mhz, whitepaper_tflops):
    assert report.peak_gflops(gpu, precision, mhz) / 1e3 == pytest.approx(whitepaper_tflops, abs=0.05)


def test_table_keys_rows_by_kernel_precision_and_size():
    rows = [{"kernel": "cublas", "precision": "fp32", "n": 1024, "gflops": 1.0},
            {"kernel": "wmma", "precision": "fp16", "n": 1024, "gflops": 2.0}]
    t = report.table(rows)
    assert t[("wmma", "fp16", 1024)]["gflops"] == 2.0 and len(t) == 2


@pytest.mark.skipif(not (ROOT / "results" / "rtx2080ti" / "gemm.csv").exists(), reason="no results")
def test_report_runs_on_the_saved_results(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "FIGURES", tmp_path / "figures")
    monkeypatch.setattr(report, "SUMMARY", tmp_path / "summary.md")
    report.main(quiet=True)
    text = (tmp_path / "summary.md").read_text()
    assert "### RTX 2080 Ti" in text and "double_buffer" in text
    assert (tmp_path / "figures" / "kernels.png").exists()
