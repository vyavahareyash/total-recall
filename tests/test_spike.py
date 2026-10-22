from __future__ import annotations

from notebooks.run_spike import run_spike_verification


def test_spike_verification_pipeline():
    """Verify all 6 spike acceptance criteria end-to-end."""
    assert run_spike_verification() is True
