"""Minimal reproducible numerical checks. Run: python test_math.py"""
import numpy as np
from math_core import reconstruct, reconstruction_metrics

rng = np.random.default_rng(2026)
x = rng.random((256, 256))
k, mask, observed, restored, inv = reconstruct(x, 0, 0)
assert mask.all(), "Unfiltered case must retain every coefficient"
assert np.allclose(inv.real, x, atol=1e-12), "Full inverse should recover the image"
assert np.allclose(restored, x, atol=1e-12), "Magnitude inverse should recover positive intensities"
assert np.allclose(k, observed), "Unfiltered k-space should remain unchanged"
assert reconstruction_metrics(x, restored, mask)["rmse"] < 1e-12

_, mask_hi, *_ = reconstruct(x, 60, 0)
_, mask_lo, *_ = reconstruct(x, 0, 20)
assert 0 < mask_hi.mean() < 1
assert 0 < mask_lo.mean() < 1
print("PASS: forward/inverse FFT, full reconstruction and both frequency masks")
