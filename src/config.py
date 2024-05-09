from dataclasses import dataclass, field
from datetime import date
from enum import Enum, auto
from functools import cached_property
from pathlib import Path

import torch

__all__ = ["Options", "InferenceConfig", "TrainingConfig"]


class Language(Enum):
    de = ("de", "de_core_news_sm")
    en = ("en", "en_core_web_sm")

    def __init__(self, code: str, spacy_model: str):
        self.code = code
        self.spacy_model = spacy_model


class Backend(Enum):
    cpu = auto()
    gpu = auto()


@dataclass
class ModeConfig:
    """Base class for inference and training configurations."""


@dataclass
class InferenceConfig(ModeConfig):
    model_path: Path
    """Path to the model to run inference on"""


@dataclass
class TrainingConfig(ModeConfig):
    epochs: int = 30
    lr: float = 1e-4


@dataclass
class Options:
    mode: ModeConfig
    backend: Backend = Backend.cpu
    src: Language = Language.de
    """Source language (translating FROM this language)"""
    tgt: Language = Language.en
    """Target language (translating TO this language)"""
    batch: int = 128
    """Batch size"""
    attn_heads: int = 8
    """Number of attention heads"""
    enc_layers: int = 5
    dec_layers: int = 5
    embed_size: int = 512
    dim_feedforward: int = 512
    dropout: float = 0.1
    logging_dir: Path = field(default_factory=lambda: Path(str(date.today())))
    """Where the output of this program should be placed"""
    dry_run: bool = False

    @cached_property
    def device(self) -> torch.device:
        return torch.device(
            "cuda" if self.backend is Backend.gpu and torch.cuda.is_available() else "cpu"
        )
