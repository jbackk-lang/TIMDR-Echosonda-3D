import numpy as np
import pytest
from timdr_midwater_targets import TIMDRMidWaterDetector


def make_bottom(n=100, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 100, n)
    y = rng.uniform(0, 100, n)
    z = 50.0 + rng.normal(0, 0.1, n)  # plaskie dno na glebokosci 50m
    return np.column_stack([x, y, z])


@pytest.fixture
def det():
    return TIMDRMidWaterDetector(make_bottom())


def test_walidacja_bottom_points():
    with pytest.raises(ValueError):
        TIMDRMidWaterDetector([[0, 0], [1, 1]])


def test_walidacja_returns_shape(det):
    with pytest.raises(ValueError):
        det.detect_targets([[0, 0, 10]])


def test_odrzuca_zwroty_z_samego_dna(det):
    """Zwroty leżące NA dnie (głębokość ~ lokalne dno) nie są celami."""
    returns = [[50, 50, 50.05, 0.9], [10, 10, 49.98, 0.8]]
    targets = det.detect_targets(returns, min_clearance=1.0)
    assert len(targets) == 0


def test_wykrywa_zwrot_w_toni_wodnej(det):
    """Zwrot wyraźnie płycej niż dno (np. 20m przy dnie na 50m) to cel."""
    returns = [[50, 50, 20.0, 0.8], [10, 10, 49.98, 0.8]]  # drugi to dno, nie cel
    targets = det.detect_targets(returns, min_clearance=1.0)
    assert len(targets) == 1
    assert targets[0][2] == 20.0


def test_odrzuca_slabe_echo_ponizej_min_amplitude(det):
    returns = [[50, 50, 20.0, 0.02]]
    targets = det.detect_targets(returns, min_clearance=1.0, min_amplitude=0.1)
    assert len(targets) == 0


def test_pelny_pipeline_rozroznia_4_typy_celow(det):
    """
    Kluczowy test: pojedyncza ryba, ławica, wieloryb, obiekt sztuczny -
    wygenerowane jako oddzielne, gęste skupiska o rosnącym rozmiarze i
    amplitudzie. Pipeline (detect -> cluster -> classify) powinien
    poprawnie rozdzielić je na 4 klastry z sensowną klasyfikacją.
    """
    rng = np.random.default_rng(0)

    def make_cluster(center, n_points, extent, amp_mean, amp_std):
        pts = []
        while len(pts) < n_points:
            cand = rng.uniform(-1, 1, 3) * extent
            if np.sum((cand / extent) ** 2) <= 1.0:
                pts.append(cand)
        pts = np.array(pts) + center
        amp = np.clip(rng.normal(amp_mean, amp_std, n_points), 0, None)
        return pts, amp

    fish, fish_amp = make_cluster(np.array([10, 10, 20]), 1, np.array([0.15, 0.15, 0.1]), 0.3, 0.05)
    school_pts, school_amp = [], []
    for i in range(10):
        p, a = make_cluster(np.array([30 + i * 0.5, 30, 25]), 2, np.array([0.2, 0.2, 0.15]), 0.35, 0.05)
        school_pts.append(p); school_amp.append(a)
    school_pts = np.vstack(school_pts); school_amp = np.concatenate(school_amp)
    whale, whale_amp = make_cluster(np.array([60, 60, 15]), 15, np.array([2.0, 0.7, 0.6]), 0.85, 0.1)
    sub, sub_amp = make_cluster(np.array([90, 20, 30]), 40, np.array([4.0, 1.0, 1.0]), 0.98, 0.05)

    all_pts = np.vstack([fish, school_pts, whale, sub])
    all_amp = np.concatenate([fish_amp, school_amp, whale_amp, sub_amp])
    returns = np.column_stack([all_pts, all_amp])

    targets = det.detect_targets(returns, min_clearance=1.0)
    assert len(targets) == len(returns)  # wszystkie sa wyraznie plycej niz dno (50m)

    labels, unique = det.cluster_targets(targets, eps=1.2)
    n_clusters = len(unique[unique != -1])
    assert n_clusters == 4, f"oczekiwano 4 klastrow, dostano {n_clusters}"

    classified = det.classify_targets(targets, labels)
    klasy = {c["klasa"] for c in classified}
    assert "ryba" in klasy
    assert "lawica_ryb" in klasy
    assert "wielorb_ssak" in klasy
    assert "obiekt_sztuczny" in klasy


def test_cluster_targets_pusta_lista(det):
    labels, unique = det.cluster_targets(np.empty((0, 4)))
    assert len(labels) == 0
