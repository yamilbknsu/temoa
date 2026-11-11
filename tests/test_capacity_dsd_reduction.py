"""
Tests for the binding-hour reduction of capacity rows for annual techs with DSD outputs
(temoa.components.capacity.annual_dsd_binding_hours).

The reference is the brute-force criterion: with F >= 0, the row rho_i . F <= C is implied by
rows j if rho_i <= sum_j lam_j rho_j component-wise for some lam >= 0 with sum lam <= 1.
"""

import logging
from itertools import product
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from scipy.optimize import linprog

from temoa.components.capacity import _upper_right_hull_2d, annual_dsd_binding_hours


def needed_hours(rho: np.ndarray) -> set[int]:
    """Brute-force reference: drop single-hour dominated points, then LP-covered ones."""
    tol = 1e-9 * rho.max()
    cand = []
    for i in np.flatnonzero(rho.any(axis=1)):
        ge = (rho >= rho[i] - tol).all(axis=1)
        better = (rho > rho[i] + tol).any(axis=1) | (np.arange(len(rho)) < i)
        if not (ge & better).any():
            cand.append(int(i))
    keep = list(cand)
    for i in cand:
        others = [j for j in keep if j != i]
        if not others:
            continue
        res = linprog(np.ones(len(others)), A_ub=-rho[others].T, b_ub=-rho[i], bounds=(0, None))
        if res.status == 0 and res.fun <= 1 + 1e-9:
            keep.remove(i)
    return set(keep)


def implied(rho: np.ndarray, kept: list[int]) -> bool:
    """Every hour's row is implied by the kept rows (exactness check)."""
    sub = rho[kept]
    for i in range(len(rho)):
        res = linprog(np.ones(len(kept)), A_ub=-sub.T, b_ub=-rho[i], bounds=(0, None))
        if res.status != 0 or res.fun > 1 + 1e-9:
            return False
    return True


# ---------------------------------------------------------------- 2D hull


@pytest.mark.parametrize(
    'points, expected',
    [
        # opposite seasons: winter heat peak, summer cooling peak
        ([(3.0, 0.1), (0.2, 2.5), (1.0, 1.0), (0.5, 0.5)], {0, 1}),
        # shoulder corner sticks out above the segment between the peaks
        ([(3.0, 0.1), (0.2, 2.5), (2.2, 2.0), (1.0, 1.0)], {0, 1, 2}),
        # two shoulder corners (4 hours kept)
        ([(3.0, 0.0), (0.0, 3.0), (2.6, 1.8), (1.8, 2.6), (1.0, 1.0)], {0, 1, 2, 3}),
        # point exactly on the segment is redundant, but kept: collinear is an unclear case
        ([(2.0, 0.0), (0.0, 2.0), (1.0, 1.0)], {0, 1, 2}),
        # point just inside the segment is dropped
        ([(2.0, 0.0), (0.0, 2.0), (0.99, 0.99)], {0, 1}),
        # one hour dominates everything
        ([(3.0, 3.0), (1.0, 2.0), (2.0, 1.0)], {0}),
    ],
)
def test_hull_2d_cases(points: list[tuple[float, float]], expected: set[int]) -> None:
    assert set(_upper_right_hull_2d(points)) == expected


def test_hull_2d_ties_keep_one() -> None:
    points = [(1.0, 1.0), (3.0, 0.5), (0.5, 3.0), (3.0, 0.5), (0.5, 3.0), (1.0, 1.0)]
    kept = _upper_right_hull_2d(points)
    assert len(kept) == 2
    assert {points[i] for i in kept} == {(3.0, 0.5), (0.5, 3.0)}


@pytest.mark.parametrize('seed', range(20))
def test_hull_2d_matches_brute_force(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 60
    # seasonal-looking profiles plus some exact duplicates
    x = rng.gamma(2.0, 1.0, n)
    y = rng.gamma(2.0, 1.0, n)
    x[5], y[5] = x[7], y[7]
    rho = np.column_stack([x, y])
    kept = _upper_right_hull_2d([tuple(r) for r in rho])
    assert implied(rho, kept)
    assert set(kept) == needed_hours(rho)


# ---------------------------------------------------------------- full routine


def fake_model(
    dsd_profiles: dict[str, list[float]],
    process_outputs: set[str],
    n_seasons: int = 4,
    n_tod: int = 3,
    tech_annual: bool = True,
) -> Any:
    seasons = [f's{i}' for i in range(n_seasons)]
    tods = [f'd{i}' for i in range(n_tod)]
    hours = list(product(seasons, tods))
    seg = {h: 1.0 / len(hours) for h in hours}
    dsd: dict[tuple[Any, ...], float] = {}
    for dem, vals in dsd_profiles.items():
        total = sum(vals)
        for h, val in zip(hours, vals, strict=True):
            dsd['R', 2030, h[0], h[1], dem] = val / total
    demands = set(dsd_profiles)
    return SimpleNamespace(
        annual_dsd_binding_hours=None,
        time_season=seasons,
        time_of_day=tods,
        segment_fraction=seg,
        demand_specific_distribution=dsd,
        capacity_factor_tech=SimpleNamespace(sparse_keys=lambda: []),
        capacity_factor_process=SimpleNamespace(sparse_keys=lambda: []),
        active_capacity_rptv={('R', 2030, 'T', 2030)},
        tech_annual={'T'} if tech_annual else set(),
        tech_curtailment=set(),
        tech_storage=set(),
        process_outputs={('R', 2030, 'T', 2030): process_outputs},
        commodity_dsd={d for d, v in dsd_profiles.items() if len(set(v)) > 1},
        commodity_demand=demands,
    ), hours


def test_1d_peak() -> None:
    vals = [1, 2, 3, 9, 4, 4, 1, 1, 2, 2, 3, 1]
    model, hours = fake_model({'heat': vals}, {'heat'})
    assert annual_dsd_binding_hours(model) == {('R', 2030, 'T', 2030): [hours[3]]}


def test_1d_tied_peak() -> None:
    vals = [1, 9, 3, 9, 4, 4, 1, 9, 2, 2, 3, 1]
    model, hours = fake_model({'heat': vals}, {'heat'})
    result = annual_dsd_binding_hours(model)
    assert len(result['R', 2030, 'T', 2030]) == 1
    assert result['R', 2030, 'T', 2030][0] in {hours[1], hours[3], hours[7]}


def test_2d_with_flat_and_non_demand_outputs() -> None:
    heat = [9, 8, 7, 3, 2, 1, 1, 1, 1, 3, 4, 5]
    cool = [1, 1, 1, 3, 4, 5, 9, 8, 7, 3, 2, 1]
    flat = [1] * 12
    model, hours = fake_model(
        {'heat': heat, 'cool': cool, 'flat': flat}, {'heat', 'cool', 'flat', 'elec'}
    )
    kept = annual_dsd_binding_hours(model)['R', 2030, 'T', 2030]
    assert set(kept) == {hours[0], hours[6]}


def test_identical_profiles_merge() -> None:
    vals = [1, 2, 3, 9, 4, 4, 1, 1, 2, 2, 3, 1]
    model, hours = fake_model({'a': vals, 'b': vals, 'c': vals}, {'a', 'b', 'c'})
    assert annual_dsd_binding_hours(model) == {('R', 2030, 'T', 2030): [hours[3]]}


def test_three_profiles_fall_back(caplog: pytest.LogCaptureFixture) -> None:
    a = [9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    b = [1, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    c = [1, 1, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    model, _ = fake_model({'a': a, 'b': b, 'c': c}, {'a', 'b', 'c'})
    with caplog.at_level(logging.WARNING):
        assert annual_dsd_binding_hours(model) == {}
    assert any('3 or more distinct' in rec.message for rec in caplog.records)


def test_not_annual_untouched() -> None:
    vals = [1, 2, 3, 9, 4, 4, 1, 1, 2, 2, 3, 1]
    model, _ = fake_model({'heat': vals}, {'heat'}, tech_annual=False)
    assert annual_dsd_binding_hours(model) == {}
