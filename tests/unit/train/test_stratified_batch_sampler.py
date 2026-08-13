import numpy as np

from speculators.train.distributed_batch_sampler import (
    MultipackDistributedBatchSamplerV2,
    StratifiedMultipackDistributedBatchSampler,
)


def test_stratified_sampler_enforces_exact_step_mix_on_every_rank():
    lengths = np.array([8] * 1200 + [64] * 160 + [8] * 400)
    categories = np.array([0] * 1200 + [1] * 160 + [2] * 400)
    schedules = []
    for rank in range(2):
        sampler = StratifiedMultipackDistributedBatchSampler(
            batch_max_length=64,
            lengths=lengths,
            categories=categories,
            category_fractions=[0.7, 0.2, 0.1],
            steps_per_epoch=100,
            num_replicas=2,
            rank=rank,
            seed=42,
        )
        batches = list(sampler)
        schedule = [int(categories[batch[0]]) for batch in batches]
        assert all(
            np.all(categories[batch] == schedule[i])
            for i, batch in enumerate(batches)
        )
        assert np.bincount(schedule, minlength=3).tolist() == [70, 20, 10]
        schedules.append(schedule)
    assert schedules[0] == schedules[1]


def test_stratified_sampler_uses_largest_remainders():
    sampler = StratifiedMultipackDistributedBatchSampler(
        batch_max_length=32,
        lengths=[8] * 300,
        categories=[0] * 100 + [1] * 100 + [2] * 100,
        category_fractions=[0.7, 0.2, 0.1],
        steps_per_epoch=13,
        num_replicas=1,
        rank=0,
    )
    schedule = [int(sampler.categories[batch[0]]) for batch in sampler]
    assert np.bincount(schedule, minlength=3).tolist() == [9, 3, 1]


def test_stratified_sampler_can_cap_samples_per_step():
    sampler = StratifiedMultipackDistributedBatchSampler(
        batch_max_length=64,
        lengths=[8] * 200,
        categories=[0] * 200,
        category_fractions=[1.0],
        steps_per_epoch=10,
        num_replicas=2,
        rank=0,
        max_samples_per_step=1,
    )
    assert all(len(batch) == 1 for batch in sampler)


def test_multipack_sampler_can_cap_samples_per_batch():
    sampler = MultipackDistributedBatchSamplerV2(
        batch_max_length=64,
        lengths=[8] * 200,
        num_replicas=2,
        rank=0,
        max_samples_per_batch=1,
    )
    assert all(len(batch) == 1 for batch in sampler)
