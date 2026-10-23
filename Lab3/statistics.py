from dataclasses import dataclass
from math import sqrt
import numpy as np
from scipy.stats import t as student_t, chi2


@dataclass
class Stats:
    n: int
    x_mean: float
    y_mean: float
    x_var: float
    y_var: float
    x_sd: float
    y_sd: float
    covariance: float
    correlation: float


def calculate(df):
    x = df["X"].to_numpy(dtype=float)
    y = df["Y"].to_numpy(dtype=float)

    n = len(x)

    mx = float(sum(x) / n)
    my = float(sum(y) / n)

    dx = x - mx
    dy = y - my

    vx = float(sum(dx * dx) / (n - 1))
    vy = float(sum(dy * dy) / (n - 1))

    if vx <= 0 or vy <= 0:
        raise ValueError("Обидві змінні повинні мати ненульовий розкид.")

    cov = float(sum(dx * dy) / (n - 1))

    sx = sqrt(vx)
    sy = sqrt(vy)

    r = cov / (sx * sy)
    r = max(-1.0, min(1.0, r))

    return Stats(
        n, mx, my, vx, vy,
        sx, sy, cov, r
    )


def confidence_intervals(s, gamma):
    if not 0 < gamma < 1:
        raise ValueError("Довірча ймовірність повинна бути між 0 та 1.")

    alpha = 1 - gamma
    nu = s.n - 1

    tcrit = float(student_t.ppf(1 - alpha / 2, nu))

    chi_lo = float(chi2.ppf(alpha / 2, nu))
    chi_hi = float(chi2.ppf(1 - alpha / 2, nu))

    return {
        "x_mean": (
            s.x_mean - tcrit * s.x_sd / sqrt(s.n),
            s.x_mean + tcrit * s.x_sd / sqrt(s.n)
        ),

        "y_mean": (
            s.y_mean - tcrit * s.y_sd / sqrt(s.n),
            s.y_mean + tcrit * s.y_sd / sqrt(s.n)
        ),

        "x_var": (
            nu * s.x_var / chi_hi,
            nu * s.x_var / chi_lo
        ),

        "y_var": (
            nu * s.y_var / chi_hi,
            nu * s.y_var / chi_lo
        )
    }


def normal_density(x, y, s):
    r = s.correlation

    zx = (x - s.x_mean) / s.x_sd
    zy = (y - s.y_mean) / s.y_sd

    q = zx ** 2 - 2 * r * zx * zy + zy ** 2

    exponent = -q / (2 * (1 - r ** 2))

    denominator = (
        2 * np.pi * s.x_sd * s.y_sd * sqrt(1 - r ** 2)
    )

    return np.exp(exponent) / denominator


def histogram2d(df, k):
    if k < 2 or k > 30:
        raise ValueError("Кількість класів повинна бути від 2 до 30.")

    h, xe, ye = np.histogram2d(
        df["X"],
        df["Y"],
        bins=k
    )

    return h.astype(int), xe, ye
