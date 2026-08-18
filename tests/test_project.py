import math

import torch

from starse.project import sign_project


def test_sign_projection_preserves_row_norms() -> None:
    weight = torch.tensor([[3.0, -4.0], [-1.0, 0.0]])
    projected = sign_project(weight)

    torch.testing.assert_close(
        torch.linalg.vector_norm(projected, dim=1),
        torch.linalg.vector_norm(weight, dim=1),
    )
    expected_signs = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]) / math.sqrt(2)
    torch.testing.assert_close(
        projected / torch.linalg.vector_norm(projected, dim=1, keepdim=True),
        expected_signs,
    )
