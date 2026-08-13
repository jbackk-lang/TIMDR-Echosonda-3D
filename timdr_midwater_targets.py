"""
TIMDR Mid-Water Target Detector — timdr_midwater_targets.py
==============================================================
Rozszerzenie TIMDR-Echosonda-3D o wykrywanie celów w toni wodnej (nie na
dnie): ryby, ławice, ssaki morskie (wieloryby), obiekty sztuczne (łodzie
podwodne, wraki w zawieszeniu). To osobne zadanie od analizy dna
(TIMDREchosonda3D) - wymaga surowych ech sonaru (wszystkie odbicia, nie
tylko dno), bo cel wodny to z definicji echo, które NIE pochodzi od dna.

Konwencja głębokości: tak jak w TIMDREchosonda3D - większa wartość =
głębiej (głębokość dodatnia w dół od powierzchni). Cel wodny ma MNIEJSZĄ
głębokość niż lokalne dno pod nim (jest bliżej powierzchni).

⚠️ Uczciwe zastrzeżenie: to jest ZGRUBNA HEURYSTYKA oparta na rozmiarze
skupiska punktów i sile echa (amplituda), skalibrowana tylko na danych
syntetycznych (patrz testy). NIE jest to zamiennik prawdziwej klasyfikacji
celów sonarowych, która w praktyce używa widma czasowo-częstotliwościowego
echa, śledzenia wielo-pingowego (ruch/tor celu) i uczenia maszynowego na
kształcie fali powrotnej. Progi rozmiaru/amplitudy poniżej są punktem
startowym do kalibracji na prawdziwych danych, nie zwalidowaną normą.
"""

import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN


class TIMDRMidWaterDetector:
    def __init__(self, bottom_points, k_neighbors=8):
        """
        bottom_points: Nx3 [x, y, depth] - chmura punktów DNA (np. wynik
        TIMDREchosonda3D.trm_surface na wcześniej wykrytych zwrotach dna).
        """
        pts = np.asarray(bottom_points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"bottom_points musi mieć kształt (N, 3), dostano {pts.shape}")
        if len(pts) < 3:
            raise ValueError("bottom_points musi mieć co najmniej 3 punkty")
        self.bottom_pts = pts
        self.bottom_tree = KDTree(pts[:, :2])
        self.k = min(k_neighbors, len(pts))

    def local_bottom_depth(self, x, y):
        """Mediana głębokości dna z k najbliższych punktów dna wokół (x, y)."""
        _, idx = self.bottom_tree.query([x, y], k=self.k)
        idx = np.atleast_1d(idx)
        return float(np.median(self.bottom_pts[idx, 2]))

    def detect_targets(self, returns, min_clearance=1.0, min_amplitude=0.0):
        """
        returns: Nx4 [x, y, depth, amplitude] - WSZYSTKIE echa (nie tylko dno)
        min_clearance: cel musi być co najmniej tyle płycej niż lokalne dno
        (w tych samych jednostkach co depth), żeby nie mylić go z
        pomiarowym szumem samego dna.
        min_amplitude: odrzuca bardzo słabe echa (szum aparatury) przed
        klasyfikacją.

        Zwraca podzbiór `returns` (te same 4 kolumny) będący kandydatami
        na cele w toni wodnej.
        """
        r = np.asarray(returns, dtype=np.float64)
        if r.ndim != 2 or r.shape[1] != 4:
            raise ValueError(f"returns musi mieć kształt (N, 4) [x,y,depth,amplitude], dostano {r.shape}")

        keep = []
        for row in r:
            x, y, depth, amp = row
            if amp < min_amplitude:
                continue
            bottom_depth = self.local_bottom_depth(x, y)
            if bottom_depth - depth >= min_clearance:  # cel plyciej niz dno o min_clearance
                keep.append(row)
        return np.array(keep) if keep else np.empty((0, 4))

    def cluster_targets(self, targets, eps=1.5, min_samples=1):
        """
        Grupuje pojedyncze zwroty w cele (DBSCAN po x,y,depth).
        eps: maksymalna odległość (w jednostkach danych) między punktami
        tego samego celu - dostosuj do gęstości Twoich danych ec sonaru.
        Zwraca (cluster_labels, unikalne_etykiety).
        """
        targets = np.asarray(targets, dtype=np.float64)
        if len(targets) == 0:
            return np.array([]), np.array([])
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(targets[:, :3])
        return db.labels_, np.unique(db.labels_)

    def classify_targets(self, targets, labels):
        """
        Bardzo zgruba heurystyka na podstawie rozmiaru skupiska (proxy
        na rozmiar fizyczny) i średniej amplitudy (proxy na siłę echa /
        target strength). Patrz zastrzeżenie w docstringu modułu.

        Zwraca listę słowników: {label, n_points, size_m, mean_amplitude, klasa}
        klasa ∈ {"ryba", "lawica_ryb", "wielorb_ssak", "obiekt_sztuczny", "nieokreslony"}
        """
        results = []
        for lbl in np.unique(labels):
            if lbl == -1:
                continue  # szum DBSCAN (brak sąsiadów w promieniu eps)
            mask = labels == lbl
            pts = targets[mask, :3]
            amp = targets[mask, 3]
            n = int(mask.sum())
            extent = np.max(pts, axis=0) - np.min(pts, axis=0) if n > 1 else np.zeros(3)
            size_m = float(np.linalg.norm(extent))
            mean_amp = float(np.mean(amp))

            if size_m < 1.0:
                klasa = "ryba" if mean_amp < 0.5 else "nieokreslony"
            elif mean_amp < 0.5:
                klasa = "lawica_ryb"
            elif size_m < 5.0:
                klasa = "wielorb_ssak"
            elif mean_amp >= 0.7:
                klasa = "obiekt_sztuczny"
            else:
                klasa = "nieokreslony"

            results.append({
                "label": int(lbl),
                "n_points": n,
                "size_m": size_m,
                "mean_amplitude": mean_amp,
                "klasa": klasa,
            })
        return results
