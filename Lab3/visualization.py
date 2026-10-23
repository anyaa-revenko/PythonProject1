import numpy as np

from matplotlib.figure import Figure
from matplotlib import cm

from Lab3.statistics import histogram2d, normal_density


def _figure(title, projection=None):
    fig = Figure(
        figsize=(8.6, 5.7),
        dpi=100,
        facecolor="white"
    )

    ax = fig.add_subplot(
        111,
        projection=projection
    )

    ax.set_title(title, fontsize=13, pad=14)

    return fig, ax


def correlation_plot(df, s, contours=False):
    title = (
        "Кореляційне поле та лінії рівня"
        if contours
        else "Кореляційне поле (X, Y)"
    )

    fig, ax = _figure(title)

    ax.scatter(
        df["X"],
        df["Y"],
        alpha=0.6,
        s=20,
        color="#2563eb",
        edgecolors="none",
        label="Спостереження"
    )

    if contours:
        x = np.linspace(
            df.X.min() - 3,
            df.X.max() + 3,
            130
        )

        y = np.linspace(
            df.Y.min() - 3,
            df.Y.max() + 3,
            130
        )

        xx, yy = np.meshgrid(x, y)

        density = normal_density(xx, yy, s)

        levels = np.linspace(
            float(density.max()) * 0.12,
            float(density.max()) * 0.85,
            6
        )

        cs = ax.contour(
            xx, yy, density,
            levels=levels,
            colors="#dc2626",
            linewidths=1.4
        )

        ax.clabel(
            cs,
            inline=True,
            fontsize=7,
            fmt="%.4f"
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(alpha=0.2)

    ax.text(
        0.02, 0.98,
        f"r = {s.correlation:.4f} | N = {s.n}",
        transform=ax.transAxes,
        va="top",
        bbox=dict(
            facecolor="white",
            alpha=0.8,
            edgecolor="#ddd"
        )
    )

    fig.tight_layout()

    return fig


def heatmap(df, k, relative=False):
    h, xe, ye = histogram2d(df, k)

    values = h / len(df) if relative else h

    title = (
        "2D-гістограма: відносні частоти"
        if relative
        else "2D-гістограма: абсолютні частоти"
    )

    fig, ax = _figure(title)

    im = ax.pcolormesh(
        xe,
        ye,
        values.T,
        shading="flat",
        cmap="YlGnBu"
    )

    fig.colorbar(
        im,
        ax=ax,
        label=(
            "Відносна частота"
            if relative
            else "Кількість"
        )
    )

    for i in range(k):
        for j in range(k):
            val = values[i, j]

            label = (
                f"{val:.2f}"
                if relative
                else str(int(val))
            )

            ax.text(
                (xe[i] + xe[i + 1]) / 2,
                (ye[j] + ye[j + 1]) / 2,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="black"
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    fig.tight_layout()

    return fig


def bars3d(df, k):
    h, xe, ye = histogram2d(df, k)

    fig, ax = _figure(
        "3D-гістограма спільних частот",
        projection="3d"
    )

    x0, y0 = np.meshgrid(
        xe[:-1],
        ye[:-1],
        indexing="ij"
    )

    dx, dy = np.meshgrid(
        np.diff(xe),
        np.diff(ye),
        indexing="ij"
    )

    heights = h.ravel()

    colors = cm.viridis(
        heights / max(1, heights.max())
    )

    ax.bar3d(
        x0.ravel(),
        y0.ravel(),
        np.zeros(k * k),
        dx.ravel() * 0.85,
        dy.ravel() * 0.85,
        heights,
        color=colors,
        shade=True,
        alpha=0.93
    )

    ax.set_xlabel("X", labelpad=8)
    ax.set_ylabel("Y", labelpad=8)
    ax.set_zlabel("Частота", labelpad=8)

    ax.view_init(elev=26, azim=-55)

    fig.tight_layout()

    return fig


def density_plot(df, s):
    x = np.linspace(
        df.X.min() - 3,
        df.X.max() + 3,
        110
    )

    y = np.linspace(
        df.Y.min() - 3,
        df.Y.max() + 3,
        110
    )

    xx, yy = np.meshgrid(x, y)

    zz = normal_density(xx, yy, s)

    fig, ax = _figure(
        "Відтворена щільність двовимірного нормального розподілу",
        projection="3d"
    )

    surf = ax.plot_surface(
        xx, yy, zz,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        alpha=0.95
    )

    fig.colorbar(
        surf,
        ax=ax,
        shrink=0.65,
        pad=0.12,
        label="f(x, y)"
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("f(x, y)")

    fig.tight_layout()

    return fig
