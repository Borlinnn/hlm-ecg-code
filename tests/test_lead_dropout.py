import numpy as np

from hlm_ecg.data.lead_dropout import RandomLeadDropout, mask_for_missing_count


def test_sampled_mask_shape_and_missing_counts():
    sampler = RandomLeadDropout(seed=123)
    allowed = {0, 1, 3, 6}
    for _ in range(50):
        mask = sampler.sample()
        assert mask.shape == (12,)
        missing = int((mask == 0).sum())
        assert missing in allowed
        assert float(mask.sum()) >= 1.0


def test_k_zero_mask_is_all_available():
    mask = mask_for_missing_count(0, rng=np.random.default_rng(1))
    assert mask.tolist() == [1.0] * 12


def test_fixed_seed_reproducible_sequence():
    a = RandomLeadDropout(seed=42)
    b = RandomLeadDropout(seed=42)
    seq_a = [a.sample().tolist() for _ in range(10)]
    seq_b = [b.sample().tolist() for _ in range(10)]
    assert seq_a == seq_b
