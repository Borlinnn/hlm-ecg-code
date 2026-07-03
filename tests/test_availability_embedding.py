import pytest
import torch

from hlm_ecg.models.availability_embedding import AvailabilityEmbedding


def test_availability_mlp_output_shape():
    module = AvailabilityEmbedding(num_leads=12, hidden_dim=32, embedding_dim=32)
    mask = torch.ones(4, 12)
    out = module(mask)
    assert tuple(out.shape) == (4, 32)


def test_availability_mlp_rejects_wrong_shape():
    module = AvailabilityEmbedding(num_leads=12, hidden_dim=32, embedding_dim=32)
    with pytest.raises(ValueError, match="availability_mask shape"):
        module(torch.ones(12))
