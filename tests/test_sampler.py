from types import SimpleNamespace

from starse.train import CyclingRoundRobinBatchSampler


class BatchSampler:
    batch_size = 2
    drop_last = True

    def __init__(self, batches: list[list[int]]) -> None:
        self.batches = batches

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)


def test_smaller_source_is_cycled() -> None:
    dataset = SimpleNamespace(datasets=[range(6), range(2)])
    sampler = CyclingRoundRobinBatchSampler(
        dataset=dataset,
        batch_samplers=[BatchSampler([[0, 1], [2, 3], [4, 5]]), BatchSampler([[0, 1]])],
        dataset_names=["primary", "small"],
        primary_dataset="primary",
    )
    assert list(sampler) == [
        [0, 1],
        [6, 7],
        [2, 3],
        [6, 7],
        [4, 5],
        [6, 7],
    ]
