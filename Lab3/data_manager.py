from pathlib import Path
import numpy as np
import pandas as pd


def validate_parameters(mx, my, sx, sy, r, n, seed):
    numbers = [mx, my, sx, sy, r]

    if not all(np.isfinite(v) for v in numbers):
        raise ValueError("Параметри повинні бути скінченними числами.")

    if sx <= 0 or sy <= 0:
        raise ValueError("Стандартні відхилення повинні бути додатними.")

    if not -1 < r < 1:
        raise ValueError("Кореляція повинна бути між -1 та 1.")

    if n < 10:
        raise ValueError("Обсяг вибірки N повинен бути не меншим за 10.")

    if seed < 0:
        raise ValueError("Seed не може бути від'ємним.")


def generate_sample(mx=67, my=55, sx=8, sy=4,
                    r=0.6, n=250, seed=700):

    validate_parameters(mx, my, sx, sy, r, n, seed)

    covariance_matrix = [
        [sx ** 2, r * sx * sy],
        [r * sx * sy, sy ** 2]
    ]

    rng = np.random.default_rng(seed)

    sample = rng.multivariate_normal(
        [mx, my],
        covariance_matrix,
        size=n
    )

    return pd.DataFrame(sample, columns=["X", "Y"])


def load_sample(filename):
    path = Path(filename)

    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)

    elif path.suffix.lower() in (".csv", ".txt"):
        df = pd.read_csv(
            path,
            sep=None,
            engine="python",
            encoding="utf-8-sig"
        )

        if df.shape[1] < 2:
            df = pd.read_csv(
                path,
                sep=r"\s+",
                engine="python",
                encoding="utf-8-sig"
            )
    else:
        raise ValueError("Підтримуються CSV, TXT, XLSX та XLS.")

    if {"X", "Y"}.issubset(df.columns):
        df = df[["X", "Y"]]

    elif df.shape[1] >= 2:
        df = df.iloc[:, :2].copy()
        df.columns = ["X", "Y"]

    else:
        raise ValueError("Файл повинен містити два стовпці X та Y.")

    try:
        for col in ["X", "Y"]:
            df[col] = pd.to_numeric(df[col], errors="raise")
    except (ValueError, TypeError):
        raise ValueError("Стовпці X та Y повинні містити числа.")

    if len(df) < 10 or not np.isfinite(df.to_numpy(dtype=float)).all():
        raise ValueError("Потрібно щонайменше 10 пар чисел без пропусків.")

    return df.reset_index(drop=True)


def save_sample(df, filename):
    path = Path(filename)

    if path.suffix.lower() == ".xlsx":
        df.to_excel(path, index=False)

    elif path.suffix.lower() == ".txt":
        df.to_csv(
            path,
            index=False,
            sep="\t",
            float_format="%.8f"
        )

    else:
        df.to_csv(
            path,
            index=False,
            float_format="%.8f"
        )
