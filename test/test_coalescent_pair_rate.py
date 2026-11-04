"""Regression tests for the specified-pair Kingman merger density."""

from types import SimpleNamespace

import numpy as np
import pytest

from treetime.merger_models import Coalescent


def model_with_constant_lineage_count(k, Tc=2.0):
    """Isolate the event factors from the fixed-k survival approximation."""
    model = Coalescent.__new__(Coalescent)
    model.nbranches = lambda t: np.zeros_like(np.asarray(t, dtype=float)) + k
    model.Tc = lambda t: np.zeros_like(np.asarray(t, dtype=float)) + Tc
    model.integral_merger_rate = lambda t: (k - 1) / (2 * Tc) * np.asarray(t, dtype=float)
    return model


@pytest.mark.parametrize("k", [2, 3, 4, 8])
def test_merger_rates_have_distinct_meanings(k):
    model = model_with_constant_lineage_count(k)
    assert model.pair_merger_rate(0.5) == pytest.approx(0.5)
    assert model.total_merger_rate(0.5) == pytest.approx(k * (k - 1) / 4)
    assert model.branch_merger_rate(0.5) == pytest.approx((k - 1) / 4)


@pytest.mark.parametrize("k", [2, 3, 4, 8])
def test_branch_cost_uses_pair_event_rate_but_total_survival_hazard(k):
    Tc = 2.0
    model = model_with_constant_lineage_count(k, Tc)
    duration = 0.4
    # Per-branch survival integral plus half an observed binary merger.
    expected = duration * (k - 1) / (2 * Tc) + 0.5 * np.log(Tc)
    assert model.cost(0.2, duration) == pytest.approx(expected)


@pytest.mark.parametrize("children", [2, 3])
def test_node_contribution_uses_pair_rate(children):
    Tc = 2.0
    model = model_with_constant_lineage_count(4, Tc)
    node = SimpleNamespace(clades=[object()] * children)
    times = np.array([0.1, 0.2, 0.4, 0.8])
    contribution = model.node_contribution(node, times)
    expected = (children - 1) * (3 * times / (2 * Tc) + np.log(Tc))
    np.testing.assert_allclose(contribution.y, expected)


def test_pair_rate_supports_time_varying_Tc():
    model = model_with_constant_lineage_count(4)
    model.Tc = lambda t: 2.0 + np.asarray(t)
    times = np.array([0.0, 1.0, 2.0])
    np.testing.assert_allclose(model.pair_merger_rate(times), 1.0 / (2.0 + times))


def test_complete_three_tip_tree_matches_labelled_kingman():
    """Two specified mergers and survival at k=3 then k=2 for one time unit each."""
    from Bio.Phylo.Newick import Clade, Tree

    a, b, c = (Clade(name=name) for name in ("a", "b", "c"))
    ab = Clade(name="ab", clades=[a, b])
    root = Clade(name="root", clades=[ab, c])
    for child, parent, time in (
        (a, ab, 0.0), (b, ab, 0.0), (c, root, 0.0),
        (ab, root, 1.0),
    ):
        child.up = parent
        child.time_before_present = time
        child.branch_length = (1.0 if parent is ab else 2.0) - time
    root.up = None
    root.time_before_present = 2.0
    root.branch_length = 0.0
    for node in (a, b, c, ab, root):
        node.bad_branch = False

    model = Coalescent(Tree(root=root), Tc=2.0)
    # Survival = -(choose(3,2) + choose(2,2)) / Tc = -2.
    # Two observed pair mergers each contribute log(1/Tc).
    assert model.total_LH() == pytest.approx(-2.0 - 2.0 * np.log(2.0), abs=1e-5)
