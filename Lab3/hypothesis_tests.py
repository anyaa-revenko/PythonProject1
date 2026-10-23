from math import sqrt
import numpy as np
from scipy.stats import t as student_t, chi2

from statistics import normal_density, histogram2d


def mean_tests(s, gamma, mx, my):
    alpha = 1 - gamma

    tcrit = float(
        student_t.ppf(1 - alpha / 2, s.n - 1)
    )

    results = []

    parameters = [
        ("X", s.x_mean, s.x_sd, mx),
        ("Y", s.y_mean, s.y_sd, my)
    ]

    for name, estimate, sd, target in parameters:
        t_stat = (estimate - target) / (sd / sqrt(s.n))

        p_value = float(
            2 * student_t.sf(abs(t_stat), s.n - 1)
        )

        results.append({
            "name": name,
            "t": t_stat,
            "critical": tcrit,
            "p": p_value,
            "reject": abs(t_stat) > tcrit
        })

    return results


def _merge_small_cells(observed, expected, threshold=5.0):
    rows, cols = observed.shape

    groups = {
        i: {i} for i in range(rows * cols)
    }

    obs = observed.ravel().astype(float)
    exp = expected.ravel().astype(float)

    neighbors = {}

    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            neighbors[idx] = set()

            for di, dj in [
                (-1, 0), (1, 0),
                (0, -1), (0, 1)
            ]:
                a = i + di
                b = j + dj

                if 0 <= a < rows and 0 <= b < cols:
                    neighbors[idx].add(a * cols + b)

    while len(groups) > 1:
        totals = {
            g: sum(exp[list(cells)])
            for g, cells in groups.items()
        }

        weak = [
            g for g in groups
            if totals[g] < threshold
        ]

        if not weak:
            break

        g = min(weak, key=lambda v: totals[v])

        adjacent = [
            h for h in groups
            if h != g and any(
                neighbors[i] & groups[h]
                for i in groups[g]
            )
        ]

        if not adjacent:
            adjacent = [
                h for h in groups if h != g
            ]

        h = min(
            adjacent,
            key=lambda v: (
                abs(totals[g] + totals[v] - threshold),
                -totals[v]
            )
        )

        groups[h].update(groups.pop(g))

    merged_obs = np.array([
        sum(obs[list(cells)])
        for cells in groups.values()
    ])

    merged_exp = np.array([
        sum(exp[list(cells)])
        for cells in groups.values()
    ])

    return merged_obs, merged_exp


def chi_square_test(df, s, gamma, k):
    h, xe, ye = histogram2d(df, k)

    xc = (xe[:-1] + xe[1:]) / 2
    yc = (ye[:-1] + ye[1:]) / 2

    xx, yy = np.meshgrid(
        xc, yc, indexing="ij"
    )

    weights = (
        normal_density(xx, yy, s)
        * np.diff(xe)[:, None]
        * np.diff(ye)[None, :]
    )

    weights = np.maximum(weights, 1e-15)

    # Нормування ймовірностей на скінченній сітці
    weights /= weights.sum()

    observed, expected = _merge_small_cells(
        h, s.n * weights
    )

    valid = expected > 0

    chi_stat = float(
        sum(
            (observed[valid] - expected[valid]) ** 2
            / expected[valid]
        )
    )

    # Віднімаємо 5 оцінених параметрів
    dfree = len(expected) - 1 - 5

    if dfree <= 0:
        return {
            "chi": chi_stat,
            "df": dfree,
            "critical": None,
            "p": None,
            "reject": None,
            "groups": len(expected),
            "minimum_expected": float(min(expected))
        }

    critical = float(chi2.ppf(gamma, dfree))

    p_value = float(
        chi2.sf(chi_stat, dfree)
    )

    return {
        "chi": chi_stat,
        "df": dfree,
        "critical": critical,
        "p": p_value,
        "reject": chi_stat > critical,
        "groups": len(expected),
        "minimum_expected": float(min(expected))
    }
