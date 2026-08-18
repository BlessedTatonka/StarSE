import copy

import torch

from starse.loss import SignProjectedContrastiveLoss


class TinyStaticModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.static = torch.nn.Module()
        self.static.embedding = torch.nn.Embedding(5, 4)
        self.static.tokenizer = object()

    def _first_module(self) -> torch.nn.Module:
        return self.static

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pooled = SignProjectedContrastiveLoss._mean_pool(features, self.static.embedding.weight)
        return {"sentence_embedding": pooled}


def test_sign_projected_loss_backpropagates_through_token_table() -> None:
    model = TinyStaticModel()
    teacher = copy.deepcopy(model)
    loss = SignProjectedContrastiveLoss(
        model=model,
        teacher=teacher,
        scale=20.0,
        pair_weight=0.17,
        teacher_kl_weight=1.14,
        cube_weight=1e-4,
        balance_weight=1e-5,
        independence_weight=1e-6,
        unit_prior_weight=1e-3,
        regularizer_sample_size=3,
    )
    anchor = {"input_ids": torch.tensor([1, 2, 2, 3]), "offsets": torch.tensor([0, 2])}
    positive = {"input_ids": torch.tensor([1, 3, 2, 4]), "offsets": torch.tensor([0, 2])}

    value = loss([anchor, positive])
    value.backward()

    assert torch.isfinite(value)
    assert model.static.embedding.weight.grad is not None
    assert model.static.embedding.weight.grad.abs().sum() > 0
