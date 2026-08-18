import numpy as np

from starse.init_model import build_initial_matrix


def test_initial_matrix_shape_and_dtype() -> None:
    rng = np.random.default_rng(42)
    source = rng.normal(size=(128, 32)).astype(np.float32)
    matrix, metadata = build_initial_matrix(source, target_dim=16)
    assert matrix.shape == (128, 16)
    assert matrix.dtype == np.float32
    assert metadata["leading_components_removed"] == 1
