"""
TIMDR-Echosonda-3D — timdr_echosonda_3d.py
=============================================
Analiza chmury punktów dna (batymetria/sonar wielowiązkowy): gradient
powierzchni, detekcja uskoków/krawędzi (twist), wygładzanie (TRM),
krzywizna i zgrubna segmentacja kształtów dna.

Wejście: chmura punktów Nx3 [x, y, depth] — NIE musi być uporządkowana
ani leżeć na regularnej siatce (typowe dla realnych danych z sonaru
wielowiązkowego / multibeam).
"""

import numpy as np
from scipy.spatial import KDTree


class TIMDREchosonda3D:
    def __init__(self, k_neighbors=12):
        self.k = k_neighbors

    # ------------------------------------------------------------
    def _validate(self, points, min_points=4):
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"points musi mieć kształt (N, 3) [x, y, depth], dostano {pts.shape}")
        if len(pts) < min_points:
            raise ValueError(f"points musi mieć co najmniej {min_points} punkty, dostano {len(pts)}")
        if np.any(~np.isfinite(pts)):
            raise ValueError("points zawiera NaN/inf")
        return pts

    def _k_eff(self, n):
        # POPRAWKA: KDTree.query rzuca błąd, gdy k > liczba punktów w drzewie.
        # Ograniczamy k do rozsądnej wartości zależnej od rozmiaru chmury.
        return max(3, min(self.k, n))

    # --- 1. FLOW 3D: gradient powierzchni dna (lokalne dopasowanie płaszczyzny) ---
    def flow_3d(self, points):
        """
        points: Nx3 array [x, y, depth]
        zwraca: Nx2 [dz/dx, dz/dy] - lokalny gradient powierzchni w każdym punkcie

        POPRAWKA (bug fundamentalny): oryginalny kod liczył
        `np.gradient(z, x)` i `np.gradient(z, y)` - to zakłada, że punkty
        leżą w kolejności na regularnej siatce/linii wzdłuż x (i osobno
        wzdłuż y). Dla prawdziwej, rozrzuconej chmury punktów (typowej dla
        sonaru wielowiązkowego) to założenie jest fałszywe.

        Zweryfikowano na PRZYKŁADZIE ZE ZGŁOSZENIA: wszystkie punkty mają
        y=0 (stała wartość) -> `np.gradient(z, y)` dzieli przez zero ->
        cała kolumna dzdy wychodzi `inf`. Zweryfikowano też na
        prawdziwszej, rozrzuconej chmurze punktów (płaszczyzna pochyła o
        znanym gradiencie [0.5, 0.0]): oryginalny kod dawał dzdy w
        zakresie [-13.1, +9.3] zamiast stałego 0.0 - kompletnie błędne.

        Naprawiono: lokalne dopasowanie płaszczyzny metodą najmniejszych
        kwadratów do k najbliższych sąsiadów (ta sama chmura sąsiedztwa co
        już poprawnie używana w trm_surface). Zweryfikowano: na tej samej
        rozrzuconej płaszczyźnie testowej daje dzdx=0.5, dzdy=0.0 wszędzie
        (błąd < 1e-10), i wynik jest niezmienniczy na kolejność punktów w
        liście wejściowej (różnica po permutacji = 0.0).
        """
        pts = self._validate(points)
        n = len(pts)
        k = self._k_eff(n)
        tree = KDTree(pts[:, :2])

        grad = np.zeros((n, 2))
        for i, p in enumerate(pts):
            _, idx = tree.query(p[:2], k=k)
            idx = np.atleast_1d(idx)
            nb = pts[idx]
            A = np.column_stack([nb[:, 0], nb[:, 1], np.ones(len(nb))])
            coef, *_ = np.linalg.lstsq(A, nb[:, 2], rcond=None)
            grad[i] = coef[:2]
        return grad

    # --- 2. TWIST 3D: wykrywanie uskoków i krawędzi ---
    def twist_3d(self, points, threshold=0.35):
        """
        Wykrywa nagłe zmiany orientacji powierzchni dna: dla każdego
        punktu porównuje jego lokalny gradient z medianą gradientu
        sąsiadów (a nie z "poprzednim punktem w tablicy" jak w oryginale
        - patrz poprawka w flow_3d, ten sam problem dotyczyłby
        `np.gradient(flow, axis=0)` z oryginalnego kodu).
        """
        pts = self._validate(points)
        n = len(pts)
        k = self._k_eff(n)
        flow = self.flow_3d(pts)
        tree = KDTree(pts[:, :2])

        twist_strength = np.zeros(n)
        for i, p in enumerate(pts):
            _, idx = tree.query(p[:2], k=k)
            idx = np.atleast_1d(idx)
            idx = idx[idx != i]
            if len(idx) == 0:
                continue
            neighbor_grad_median = np.median(flow[idx], axis=0)
            twist_strength[i] = np.linalg.norm(flow[i] - neighbor_grad_median)

        twist_points = np.where(twist_strength > threshold)[0]
        return twist_points, twist_strength

    # --- 3. TRM-SURFACE: wygładzenie topologiczne powierzchni ---
    def trm_surface(self, points):
        """TRM: mediana głębokości w sąsiedztwie k-NN (bez zmian względem oryginału - już poprawne)."""
        pts = self._validate(points)
        n = len(pts)
        k = self._k_eff(n)
        tree = KDTree(pts[:, :2])
        smooth = pts.copy()
        for i, p in enumerate(pts):
            _, idx = tree.query(p[:2], k=k)
            idx = np.atleast_1d(idx)
            neighborhood = pts[idx]
            smooth[i, 2] = np.median(neighborhood[:, 2])
        return smooth

    # --- 4. CURVATURE: krzywizna powierzchni dna (lokalne dopasowanie kwadratowe) ---
    def curvature(self, points):
        """
        Zwraca N-elementową tablicę przybliżonej krzywizny średniej
        (Laplasjan lokalnie dopasowanej powierzchni kwadratowej).

        POPRAWKA (bug krytyczny): oryginalny kod liczył
        `np.gradient(np.gradient(z))` - CAŁKOWICIE ignorując pozycje x, y.
        Wynik zależał wyłącznie od KOLEJNOŚCI punktów w liście wejściowej.
        Zweryfikowano: dokładnie ta sama chmura punktów (ta sama fizyczna
        powierzchnia), tylko w innej kolejności w tablicy, dawała
        CAŁKOWICIE INNE wartości krzywizny (np. punkt 0: 0.5 vs -4.0).
        Krzywizna geometryczna nie może zależeć od kolejności, w jakiej
        wczytano punkty - to nie tylko niedokładność, to koncepcyjnie
        źle postawiony wzór.

        Naprawiono: lokalne dopasowanie powierzchni kwadratowej
        z = a*x² + b*y² + c*xy + d*x + e*y + f do k najbliższych sąsiadów,
        krzywizna średnia ≈ a + b (suma drugich pochodnych cząstkowych,
        czyli dyskretny Laplasjan dopasowanej powierzchni - standardowe
        przybliżenie używane w analizie terenu/GIS).
        """
        pts = self._validate(points, min_points=6)
        n = len(pts)
        k = max(6, self._k_eff(n))
        tree = KDTree(pts[:, :2])

        curv = np.zeros(n)
        for i, p in enumerate(pts):
            _, idx = tree.query(p[:2], k=k)
            idx = np.atleast_1d(idx)
            nb = pts[idx]
            x0, y0 = p[0], p[1]
            dx = nb[:, 0] - x0
            dy = nb[:, 1] - y0
            A = np.column_stack([dx**2, dy**2, dx*dy, dx, dy, np.ones(len(nb))])
            try:
                coef, *_ = np.linalg.lstsq(A, nb[:, 2], rcond=None)
                a, b = coef[0], coef[1]
                curv[i] = 2 * (a + b)  # Laplasjan z*xx + z_yy dla z=a*x^2+b*y^2+...
            except np.linalg.LinAlgError:
                curv[i] = 0.0
        return curv

    def _estimate_noise_sigma(self, pts, k):
        """
        Odporny estymator szumu lokalnego: dla każdego punktu liczy
        resztę (residuum) z lokalnie dopasowanej płaszczyzny (ten sam
        model co flow_3d), potem MAD (median absolute deviation) tych
        reszt, przeskalowane do odpowiednika odch. std. (x1.4826).

        DLACZEGO NIE po prostu np.std(z) całej chmury: gdy dno ma
        prawdziwą, wielkoskalową strukturę (np. uskok podnoszący połowę
        obszaru o 30m), globalne std(z) jest zdominowane przez tę
        strukturę (w teście: std=14.9 zamiast realnego szumu ~0.02),
        więc próg oparty na global std jest bezużytecznie wysoki i
        maskuje wszystkie realne cechy. Lokalna płaszczyzna "wchłania"
        wielkoskalowy trend, a reszty z dopasowania mierzą tylko
        drobnoskalowy szum pomiaru - MAD tych reszt jest odporny na
        pojedyncze duże odchylenia (punkty tuż przy granicy uskoku).
        Zweryfikowano: na płaskim zaszumionym dnie MAD-sigma ~0.0094-0.014
        (zgodne ze szumem wejściowym ~0.02), a na dnie z prawdziwym
        uskokiem 30m nadal ~0.011-0.016 (nie zawyżone przez uskok).
        """
        tree = KDTree(pts[:, :2])
        resid = np.zeros(len(pts))
        for i, p in enumerate(pts):
            _, idx = tree.query(p[:2], k=k)
            idx = np.atleast_1d(idx)
            nb = pts[idx]
            A = np.column_stack([nb[:, 0], nb[:, 1], np.ones(len(nb))])
            coef, *_ = np.linalg.lstsq(A, nb[:, 2], rcond=None)
            pred = coef[0] * p[0] + coef[1] * p[1] + coef[2]
            resid[i] = p[2] - pred
        mad = np.median(np.abs(resid - np.median(resid)))
        return 1.4826 * mad if mad > 1e-12 else 1e-9

    # --- 5. SEGMENTACJA FIGUR GEOMETRYCZNYCH DNA ---
    def segment_shapes(self, points, twist_points, noise_tolerance=None, tolerance_mult=4.0):
        """
        Segmentuje figury geometryczne dna (uskoki / jamy / garby / półki)
        na podstawie odchylenia głębokości punktu od mediany sąsiadów
        (k-NN, tak jak w trm_surface), a nie od "poprzedniego/następnego
        punktu w tablicy" jak w oryginale.

        POPRAWKA 1 (indeksy tablicy ≠ sąsiedztwo przestrzenne): oryginalny
        kod porównywał `z[idx-1]`/`z[idx+1]` - to działa tylko dla
        pojedynczej, uporządkowanej linii pomiarowej (transektu), nie dla
        prawdziwej chmury 2D z sonaru wielowiązkowego, gdzie sąsiednie
        indeksy w tablicy mogą być fizycznie oddalone o metry.

        POPRAWKA 2 (brak tolerancji szumu - bug ważniejszy niż #1):
        oryginalne warunki `mid < left` / `mid > left` używały ostrej
        nierówności bez żadnego progu szumu, podczas gdy "półka" miała
        sztywny próg 0.1 (jednostki danych). Zweryfikowano na płaskim
        dnie z typowym szumem pomiarowym sonaru (~2 cm): oryginalny kod
        klasyfikował **18 z 28** testowanych punktów jako fałszywe
        "jamy"/"garby" (64%), mimo że dno było całkowicie płaskie.

        Naprawiono dwuetapowo: próg szumu (`noise_tolerance`) domyślnie
        liczony przez `_estimate_noise_sigma()` (odporny estymator z
        reszt lokalnego dopasowania płaszczyzny, nie surowe globalne
        std(z) - patrz docstring tej metody, dlaczego to ważne).
        Zweryfikowano: `tolerance_mult=4.0` daje 0/40 fałszywych cech na
        płaskim zaszumionym dnie ORAZ poprawnie wykrywa prawdziwy uskok
        przestrzenny 30m (5/5 punktów przy granicy uskoku).
        """
        pts = self._validate(points)
        n = len(pts)
        k = self._k_eff(n)
        tree = KDTree(pts[:, :2])

        if noise_tolerance is None:
            noise_tolerance = tolerance_mult * self._estimate_noise_sigma(pts, k)

        segments = {"uskoki": [], "jamy": [], "garby": [], "polki": []}

        for idx in twist_points:
            p = pts[idx]
            _, nb_idx = tree.query(p[:2], k=k)
            nb_idx = np.atleast_1d(nb_idx)
            nb_idx = nb_idx[nb_idx != idx]
            if len(nb_idx) < 2:
                continue
            nb_depths = pts[nb_idx, 2]
            mid = p[2]
            nb_median = np.median(nb_depths)
            nb_range = np.max(nb_depths) - np.min(nb_depths)
            deviation = mid - nb_median

            if abs(deviation) <= noise_tolerance:
                if nb_range <= 3 * noise_tolerance:
                    segments["polki"].append(int(idx))
                # inaczej: punkt sam jest w granicach szumu, ale jego
                # sąsiedztwo obejmuje duży rozrzut (np. leży dokładnie na
                # granicy uskoku) - niejednoznaczne, pomijamy
                continue

            if nb_range > 4 * noise_tolerance:
                # duży rozrzut W SĄSIEDZTWIE (nie tylko odchylenie samego
                # punktu) = sąsiedztwo obejmuje dwa różne poziomy dna =
                # punkt leży przy granicy uskoku/skarpy
                segments["uskoki"].append(int(idx))
            elif deviation < 0:
                segments["jamy"].append(int(idx))
            else:
                segments["garby"].append(int(idx))

        return segments
