import numpy as np
import pytest
from timdr_echosonda_3d import TIMDREchosonda3D


@pytest.fixture
def ech():
    return TIMDREchosonda3D(k_neighbors=10)


def test_walidacja_ksztaltu(ech):
    with pytest.raises(ValueError):
        ech.flow_3d([[0, 0], [1, 1]])


def test_walidacja_min_punktow(ech):
    with pytest.raises(ValueError):
        ech.flow_3d([[0, 0, 1], [1, 1, 2]])


def test_dane_ze_zgloszenia_bez_nan_inf(ech):
    """
    Regresja kluczowego bugu: przykład z opisu ma y=0 dla wszystkich
    punktów -> oryginalny kod dzielił przez zero w np.gradient(z, y),
    dając inf/nan wszędzie i pusty twist_points na przykładzie z
    jawnie zakodowanym uskokiem.
    """
    ech_small = TIMDREchosonda3D(k_neighbors=4)
    points = [[0, 0, 10], [1, 0, 12], [2, 0, 15], [3, 0, 40], [4, 0, 42], [5, 0, 43]]
    flow = ech_small.flow_3d(points)
    assert np.all(np.isfinite(flow)), f"flow zawiera NaN/inf: {flow}"
    twist_pts, twist_str = ech_small.twist_3d(points)
    assert np.all(np.isfinite(twist_str))


def test_gradient_plaszczyzny_poprawny_na_rozrzuconej_chmurze(ech):
    """
    Regresja bugu fundamentalnego: prawdziwy gradient płaszczyzny
    pochyłej (dzdx=0.5, dzdy=0.0) na losowo rozrzuconych punktach 2D.
    """
    rng = np.random.default_rng(0)
    n = 50
    x = rng.uniform(0, 10, n)
    y = rng.uniform(0, 10, n)
    z = 20 + 0.5 * x
    points = np.column_stack([x, y, z]).tolist()
    flow = ech.flow_3d(points)
    assert np.allclose(flow[:, 0], 0.5, atol=1e-6)
    assert np.allclose(flow[:, 1], 0.0, atol=1e-6)


def test_gradient_niezmienniczy_na_kolejnosc_punktow(ech):
    rng = np.random.default_rng(0)
    n = 30
    x = rng.uniform(0, 10, n)
    y = rng.uniform(0, 10, n)
    z = 20 + 0.3 * x + 0.1 * y
    points = np.column_stack([x, y, z])
    flow1 = ech.flow_3d(points.tolist())
    perm = rng.permutation(n)
    flow2 = ech.flow_3d(points[perm].tolist())
    inv = np.argsort(perm)
    assert np.allclose(flow1, flow2[inv], atol=1e-8)


def test_curvature_niezmiennicza_na_kolejnosc_punktow(ech):
    """
    Regresja bugu: curvature() w oryginale zależała wyłącznie od
    kolejności punktów w tablicy, nie od ich rzeczywistej geometrii.
    """
    rng = np.random.default_rng(2)
    n = 20
    x = rng.uniform(0, 10, n)
    y = rng.uniform(0, 10, n)
    z = 20 + 0.1 * x**2 + 0.1 * y**2
    points = np.column_stack([x, y, z])
    ech6 = TIMDREchosonda3D(k_neighbors=10)
    curv1 = ech6.curvature(points.tolist())
    perm = rng.permutation(n)
    curv2 = ech6.curvature(points[perm].tolist())
    inv = np.argsort(perm)
    assert np.allclose(curv1, curv2[inv], atol=1e-6)


def test_curvature_plaszczyzna_ma_zerowa_krzywizne(ech):
    rng = np.random.default_rng(3)
    n = 30
    x = rng.uniform(0, 10, n)
    y = rng.uniform(0, 10, n)
    z = 5 + 0.3 * x - 0.2 * y  # idealna plaszczyzna, krzywizna = 0
    points = np.column_stack([x, y, z]).tolist()
    curv = ech.curvature(points)
    assert np.allclose(curv, 0.0, atol=1e-6)


def test_segment_shapes_brak_falszywych_cech_na_plaskim_szumie(ech):
    """
    Regresja bugu braku tolerancji szumu: płaskie dno + szum ~2cm
    dawało w oryginale 64% falszywych 'jam'/'garbow'.
    """
    rng = np.random.default_rng(1)
    n = 40
    x = rng.uniform(0, 20, n)
    y = rng.uniform(0, 20, n)
    z = 20.0 + rng.normal(0, 0.02, n)
    points = np.column_stack([x, y, z]).tolist()
    segs = ech.segment_shapes(points, list(range(n)))
    n_false = len(segs["jamy"]) + len(segs["garby"])
    assert n_false == 0, f"falszywe cechy na plaskim dnie: {n_false}"


def test_segment_shapes_wykrywa_prawdziwy_uskok(ech):
    """
    Prawdziwy uskok przestrzenny: cala polowa obszaru (x>10) podniesiona
    o 30m - spojny geograficznie stopien terenu, nie losowe pojedyncze
    punkty (te klasyfikowalyby sie poprawnie jako 'garby', nie 'uskoki').
    """
    rng = np.random.default_rng(1)
    n = 60
    x = rng.uniform(0, 20, n)
    y = rng.uniform(0, 20, n)
    z = 20.0 + rng.normal(0, 0.02, n) + np.where(x > 10, 30.0, 0.0)
    points = np.column_stack([x, y, z]).tolist()
    segs = ech.segment_shapes(points, list(range(n)))
    near_fault = [i for i in range(n) if 8 < x[i] < 12]
    assert len(segs["uskoki"]) > 0
    assert any(i in near_fault for i in segs["uskoki"])


def test_trm_surface_wygladza(ech):
    rng = np.random.default_rng(4)
    n = 30
    x = rng.uniform(0, 10, n)
    y = rng.uniform(0, 10, n)
    z = 20.0 + rng.normal(0, 1.0, n)
    points = np.column_stack([x, y, z]).tolist()
    smooth = ech.trm_surface(points)
    assert np.std(smooth[:, 2]) < np.std(z)
