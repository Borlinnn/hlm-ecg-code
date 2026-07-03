"""Model definitions for HLM-ECG."""
from hlm_ecg.models.availability_embedding import AvailabilityEmbedding
from hlm_ecg.models.backbones import InceptionTime1D, XResNet1DLike
from hlm_ecg.models.mask_token import LearnableLeadMaskToken
from hlm_ecg.models.multitask_heads import SubclassAuxiliaryHead
from hlm_ecg.models.resnet1d import ResNet1D
from hlm_ecg.models.resnet1d_availability import ResNet1DAvailability

__all__ = [
    "AvailabilityEmbedding",
    "InceptionTime1D",
    "LearnableLeadMaskToken",
    "ResNet1D",
    "ResNet1DAvailability",
    "SubclassAuxiliaryHead",
    "XResNet1DLike",
]
