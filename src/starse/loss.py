"""Loss used for StaRSE sign-projected post-training."""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
import torch.nn.functional as F


def _unwrap_model(model: Any) -> Any:
    unwrapped = model
    seen: set[int] = set()
    while hasattr(unwrapped, "module") and id(unwrapped) not in seen:
        seen.add(id(unwrapped))
        unwrapped = unwrapped.module
    return unwrapped


def static_embedding_weight(model: Any) -> torch.Tensor:
    """Return the trainable token table from a StaticEmbedding model."""

    model = _unwrap_model(model)
    first_module = model._first_module() if callable(getattr(model, "_first_module", None)) else model[0]
    embedding = getattr(first_module, "embedding", None)
    weight = getattr(embedding, "weight", None)
    if weight is None or getattr(first_module, "tokenizer", None) is None:
        raise TypeError("Expected a SentenceTransformer model whose first module is StaticEmbedding")
    return weight


class SignProjectedContrastiveLoss(torch.nn.Module):
    """Symmetric InfoNCE on a sign-projected token table with teacher KL."""

    def __init__(
        self,
        *,
        model: Any,
        teacher: Any,
        scale: float,
        pair_weight: float,
        teacher_kl_weight: float,
        cube_weight: float = 0.0,
        balance_weight: float = 0.0,
        independence_weight: float = 0.0,
        unit_prior_weight: float = 0.0,
        regularizer_sample_size: int = 8192,
    ) -> None:
        super().__init__()
        self.model = model
        self.teacher = teacher
        self.scale = float(scale)
        self.pair_weight = float(pair_weight)
        self.teacher_kl_weight = float(teacher_kl_weight)
        self.cube_weight = float(cube_weight)
        self.balance_weight = float(balance_weight)
        self.independence_weight = float(independence_weight)
        self.unit_prior_weight = float(unit_prior_weight)
        self.regularizer_sample_size = int(regularizer_sample_size)

        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.teacher.eval()

        initial = static_embedding_weight(model).detach().float()
        initial_norm = torch.linalg.vector_norm(initial, dim=1).clamp_min(1e-12)
        self.register_buffer("initial_unit", initial / initial_norm.unsqueeze(1))

    @staticmethod
    def _features_to_device(features: dict[str, Any], device: torch.device) -> dict[str, Any]:
        return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in features.items()}

    @staticmethod
    def _scores(left: torch.Tensor, right: torch.Tensor, scale: float) -> torch.Tensor:
        return F.normalize(left, p=2, dim=1) @ F.normalize(right, p=2, dim=1).t() * scale

    @staticmethod
    def _symmetric_nce(scores: torch.Tensor) -> torch.Tensor:
        targets = torch.arange(scores.shape[0], device=scores.device)
        return 0.5 * (F.cross_entropy(scores, targets) + F.cross_entropy(scores.t(), targets))

    @staticmethod
    def _symmetric_kl(teacher_scores: torch.Tensor, student_scores: torch.Tensor) -> torch.Tensor:
        forward = F.kl_div(
            F.log_softmax(student_scores, dim=1),
            F.softmax(teacher_scores.detach(), dim=1),
            reduction="batchmean",
        )
        backward = F.kl_div(
            F.log_softmax(student_scores.t(), dim=1),
            F.softmax(teacher_scores.detach().t(), dim=1),
            reduction="batchmean",
        )
        return 0.5 * (forward + backward)

    def _sign_projected_weight(self) -> torch.Tensor:
        weight = static_embedding_weight(self.model)
        norms = torch.linalg.vector_norm(weight.float(), dim=1).to(weight.dtype).clamp_min(1e-12)
        unit = weight / norms.unsqueeze(1)
        hard_unit = torch.where(unit >= 0, torch.ones_like(unit), -torch.ones_like(unit)) / math.sqrt(unit.shape[1])
        straight_through_unit = unit + (hard_unit - unit).detach()
        return norms.unsqueeze(1) * straight_through_unit

    @staticmethod
    def _mean_pool(features: dict[str, Any], weight: torch.Tensor) -> torch.Tensor:
        if "input_ids" not in features or "offsets" not in features:
            raise ValueError("Sign-projected training requires StaticEmbedding input_ids and offsets")
        input_ids = features["input_ids"].to(device=weight.device, dtype=torch.long).view(-1)
        offsets = features["offsets"].to(device=weight.device, dtype=torch.long).view(-1)
        ends = torch.cat([offsets[1:], input_ids.new_tensor([input_ids.numel()])])
        lengths = ends - offsets
        segment_ids = torch.repeat_interleave(torch.arange(offsets.numel(), device=weight.device), lengths)
        pooled = weight.new_zeros((offsets.numel(), weight.shape[1]))
        pooled.index_add_(0, segment_ids, weight.index_select(0, input_ids))
        return pooled / lengths.clamp_min(1).to(pooled.dtype).unsqueeze(1)

    def _regularizer(self) -> torch.Tensor:
        weight = static_embedding_weight(self.model)
        if not any(
            value > 0
            for value in (self.cube_weight, self.balance_weight, self.independence_weight, self.unit_prior_weight)
        ):
            return weight.sum() * 0.0

        norms = torch.linalg.vector_norm(weight.float(), dim=1).to(weight.dtype).clamp_min(1e-12)
        unit = weight / norms.unsqueeze(1)
        initial_unit = self.initial_unit.to(device=unit.device, dtype=unit.dtype)
        if 0 < self.regularizer_sample_size < unit.shape[0]:
            rows = torch.randint(0, unit.shape[0], (self.regularizer_sample_size,), device=unit.device)
            unit = unit.index_select(0, rows)
            initial_unit = initial_unit.index_select(0, rows)

        total = unit.sum() * 0.0
        if self.cube_weight > 0:
            total = total + self.cube_weight * (unit.abs().sum(dim=1) - math.sqrt(unit.shape[1])).square().mean()
        if self.balance_weight > 0 or self.independence_weight > 0:
            hard_bits = torch.where(unit >= 0, torch.ones_like(unit), -torch.ones_like(unit))
            bits = unit + (hard_bits - unit).detach()
            if self.balance_weight > 0:
                total = total + self.balance_weight * bits.mean(dim=0).square().mean()
            if self.independence_weight > 0:
                correlation = bits.t() @ bits / float(bits.shape[0])
                identity = torch.eye(correlation.shape[0], device=unit.device, dtype=unit.dtype)
                total = total + self.independence_weight * (correlation - identity).square().mean()
        if self.unit_prior_weight > 0:
            total = total + self.unit_prior_weight * (1.0 - (unit * initial_unit).sum(dim=1)).mean()
        return total

    def _teacher_embeddings(
        self,
        sentence_features: Sequence[dict[str, Any]],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.teacher.to(device)
        self.teacher.eval()
        with torch.no_grad():
            embeddings = [
                self.teacher(self._features_to_device(features, device))["sentence_embedding"]
                for features in sentence_features
            ]
        return embeddings[0], embeddings[1]

    def forward(self, sentence_features: Sequence[dict[str, Any]], labels: Any = None) -> torch.Tensor:
        del labels
        if len(sentence_features) != 2:
            raise ValueError("SignProjectedContrastiveLoss expects anchor-positive pairs")

        device = static_embedding_weight(self.model).device
        features = [self._features_to_device(item, device) for item in sentence_features]
        projected_weight = self._sign_projected_weight()
        anchor = self._mean_pool(features[0], projected_weight)
        positive = self._mean_pool(features[1], projected_weight)
        scores = self._scores(anchor, positive, self.scale)
        total = self.pair_weight * self._symmetric_nce(scores)

        if self.teacher_kl_weight > 0:
            teacher_anchor, teacher_positive = self._teacher_embeddings(features, device)
            teacher_scores = self._scores(teacher_anchor, teacher_positive, self.scale)
            total = total + self.teacher_kl_weight * self._symmetric_kl(teacher_scores, scores)

        return total + self._regularizer()
