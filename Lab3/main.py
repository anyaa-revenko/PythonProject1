import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk
)

from data_manager import (
    generate_sample,
    load_sample,
    save_sample,
    validate_parameters
)

from statistics import (
    calculate,
    confidence_intervals
)

from hypothesis_tests import (
    mean_tests,
    chi_square_test
)

from visualization import (
    correlation_plot,
    heatmap,
    bars3d,
    density_plot
)


class Lab3App(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title(
            "Лабораторна №3 — Двовимірні вибірки | Варіант 7"
        )

        self.geometry("1190x760")
        self.minsize(960, 630)

        self.df = None
        self.stats = None
        self.canvases = {}

        self._init_style()
        self._init_variables()
        self._build_ui()

        self.generate()

    def _init_style(self):
        style = ttk.Style(self)

        if "clam" in style.theme_names():
            style.theme_use("clam")

        style.configure(
            "TNotebook.Tab",
            padding=(14, 8),
            font=("Segoe UI", 10)
        )

        style.configure(
            "TButton",
            padding=(9, 6),
            font=("Segoe UI", 10)
        )

        style.configure(
            "TLabel",
            font=("Segoe UI", 10)
        )

        style.configure(
            "Header.TLabel",
            font=("Segoe UI", 15, "bold")
        )

        style.configure(
            "Treeview",
            rowheight=26,
            font=("Segoe UI", 10)
        )

        style.configure(
            "Treeview.Heading",
            font=("Segoe UI", 10, "bold")
        )

    def _init_variables(self):
        defaults = {
            "mx": "67",
            "my": "55",
            "sx": "8",
            "sy": "4",
            "r": "0.60",
            "n": "250",
            "gamma": "0.95",
            "k": "7",
            "seed": "700"
        }

        self.vars = {
            key: tk.StringVar(value=value)
            for key, value in defaults.items()
        }

        self.relative = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Готово до роботи")

    def _build_ui(self):
        top = ttk.Frame(self, padding=(14, 10))
        top.pack(fill="x")

        ttk.Label(
            top,
            text="Аналіз двовимірних випадкових вибірок",
            style="Header.TLabel"
        ).pack(anchor="w")

        ttk.Label(
            top,
            text="Лабораторна робота №3 · Варіант 7"
        ).pack(anchor="w")

        params = ttk.LabelFrame(
            self,
            text="Параметри моделювання та аналізу",
            padding=9
        )

        params.pack(
            fill="x",
            padx=14,
            pady=(0, 8)
        )

        entries = [
            ("mx", "mₓ"),
            ("my", "mᵧ"),
            ("sx", "σₓ"),
            ("sy", "σᵧ"),
            ("r", "rₓᵧ"),
            ("n", "N"),
            ("gamma", "γ"),
            ("k", "k"),
            ("seed", "seed")
        ]

        for i, (key, label) in enumerate(entries):
            ttk.Label(
                params,
                text=label
            ).grid(
                row=0,
                column=i,
                padx=(5, 1),
                pady=4
            )

            ttk.Entry(
                params,
                textvariable=self.vars[key],
                width=9
            ).grid(
                row=1,
                column=i,
                padx=(2, 8),
                pady=2,
                sticky="ew"
            )

            params.columnconfigure(i, weight=1)

        actions = ttk.Frame(
            self,
            padding=(14, 0)
        )

        actions.pack(
            fill="x",
            pady=(0, 8)
        )

        ttk.Button(
            actions,
            text="Згенерувати вибірку",
            command=self.generate
        ).pack(side="left", padx=(0, 8))

        ttk.Button(
            actions,
            text="Завантажити файл",
            command=self.load
        ).pack(side="left", padx=(0, 8))

        ttk.Button(
            actions,
            text="Зберегти вибірку",
            command=self.save
        ).pack(side="left", padx=(0, 8))

        ttk.Button(
            actions,
            text="Перерахувати γ, k",
            command=self.refresh
        ).pack(side="left", padx=(0, 8))

        self.notebook = ttk.Notebook(self)

        self.notebook.pack(
            fill="both",
            expand=True,
            padx=14,
            pady=(0, 7)
        )

        self.tabs = {}

        tab_names = [
            ("data", "Дані"),
            ("stats", "Характеристики"),
            ("corr", "Кореляція"),
            ("heat", "2D-гістограма"),
            ("bars", "3D-гістограма"),
            ("density", "Щільність"),
            ("tests", "Перевірка гіпотез")
        ]

        for key, name in tab_names:
            frame = ttk.Frame(
                self.notebook,
                padding=8
            )

            self.notebook.add(frame, text=name)
            self.tabs[key] = frame

        self.notebook.bind(
            "<<NotebookTabChanged>>",
            self._tab_changed
        )

        self._build_data_tab()
        self._build_stats_tab()
        self._build_tests_tab()
        self._build_heat_tab()

        ttk.Label(
            self,
            textvariable=self.status,
            anchor="w",
            padding=(14, 6)
        ).pack(fill="x")

    def _build_data_tab(self):
        frame = self.tabs["data"]

        ttk.Label(
            frame,
            text="Згенеровані / завантажені пари (X, Y)",
            style="Header.TLabel"
        ).pack(anchor="w", pady=(0, 6))

        table_frame = ttk.Frame(frame)
        table_frame.pack(fill="both", expand=True)

        self.data_tree = ttk.Treeview(
            table_frame,
            columns=("id", "x", "y"),
            show="headings"
        )

        columns = [
            ("id", "№", 65),
            ("x", "X", 175),
            ("y", "Y", 175)
        ]

        for key, label, width in columns:
            self.data_tree.heading(key, text=label)

            self.data_tree.column(
                key,
                width=width,
                anchor="center"
            )

        scrollbar = ttk.Scrollbar(
            table_frame,
            orient="vertical",
            command=self.data_tree.yview
        )

        self.data_tree.configure(
            yscrollcommand=scrollbar.set
        )

        self.data_tree.pack(
            side="left",
            fill="both",
            expand=True
        )

        scrollbar.pack(
            side="right",
            fill="y"
        )

        self.data_caption = ttk.Label(
            frame,
            text=""
        )

        self.data_caption.pack(
            anchor="w",
            pady=(7, 0)
        )

    def _build_stats_tab(self):
        frame = self.tabs["stats"]

        ttk.Label(
            frame,
            text="Оцінки та довірчі інтервали",
            style="Header.TLabel"
        ).pack(anchor="w", pady=(0, 8))

        self.stats_tree = ttk.Treeview(
            frame,
            columns=(
                "metric",
                "estimate",
                "target",
                "interval",
                "contains"
            ),
            show="headings",
            height=11
        )

        columns = [
            ("metric", "Показник", 225),
            ("estimate", "Оцінка", 135),
            ("target", "Заданий параметр", 145),
            ("interval", "Довірчий інтервал", 260),
            ("contains", "Накриває?", 110)
        ]

        for key, label, width in columns:
            self.stats_tree.heading(key, text=label)

            self.stats_tree.column(
                key,
                width=width,
                anchor="w" if key == "metric" else "center"
            )

        self.stats_tree.pack(
            fill="x",
            pady=4
        )

        self.stats_caption = ttk.Label(
            frame,
            text="",
            wraplength=900,
            justify="left"
        )

        self.stats_caption.pack(
            anchor="w",
            pady=12
        )

    def _build_tests_tab(self):
        frame = self.tabs["tests"]

        ttk.Label(
            frame,
            text="Перевірка статистичних гіпотез",
            style="Header.TLabel"
        ).pack(anchor="w", pady=(0, 12))

        ttk.Label(
            frame,
            text="1. T-критерій Стьюдента",
            font=("Segoe UI", 12, "bold")
        ).pack(anchor="w", pady=(0, 6))

        self.t_tree = ttk.Treeview(
            frame,
            columns=(
                "name",
                "t",
                "critical",
                "p",
                "decision"
            ),
            show="headings",
            height=2
        )

        t_columns = [
            ("name", "Випадкова величина", 170),
            ("t", "Обчислене значення t", 190),
            ("critical", "Критичні межі t", 190),
            ("p", "p-value", 130),
            ("decision", "Результат перевірки H₀", 240)
        ]

        for key, title, width in t_columns:
            self.t_tree.heading(key, text=title)

            self.t_tree.column(
                key,
                width=width,
                anchor="center"
            )

        self.t_tree.pack(
            fill="x",
            pady=(0, 20)
        )

        ttk.Label(
            frame,
            text="2. Критерій χ² Пірсона",
            font=("Segoe UI", 12, "bold")
        ).pack(anchor="w", pady=(0, 6))

        self.chi_tree = ttk.Treeview(
            frame,
            columns=("metric", "value"),
            show="headings",
            height=7
        )

        self.chi_tree.heading(
            "metric",
            text="Показник"
        )

        self.chi_tree.heading(
            "value",
            text="Значення"
        )

        self.chi_tree.column(
            "metric",
            width=350,
            anchor="w"
        )

        self.chi_tree.column(
            "value",
            width=280,
            anchor="center"
        )

        self.chi_tree.pack(
            fill="x",
            pady=(0, 15)
        )

        self.tests_caption = ttk.Label(
            frame,
            text="",
            wraplength=950,
            justify="left"
        )

        self.tests_caption.pack(
            anchor="w",
            pady=8
        )

    def _build_heat_tab(self):
        bar = ttk.Frame(
            self.tabs["heat"]
        )

        bar.pack(fill="x")

        ttk.Checkbutton(
            bar,
            text="Показати відносні частоти pᵢⱼ = nᵢⱼ / N",
            variable=self.relative,
            command=lambda: self._draw("heat")
        ).pack(
            side="left",
            pady=(0, 6)
        )

        self.chart_frames = {
            "heat": ttk.Frame(
                self.tabs["heat"]
            )
        }

        self.chart_frames["heat"].pack(
            fill="both",
            expand=True
        )

    def _get_settings(self):
        try:
            p = {}

            for key in (
                "mx", "my", "sx",
                "sy", "r", "gamma"
            ):
                p[key] = float(
                    self.vars[key].get().replace(",", ".")
                )

            for key in ("n", "k", "seed"):
                p[key] = int(
                    self.vars[key].get()
                )

            validate_parameters(
                p["mx"],
                p["my"],
                p["sx"],
                p["sy"],
                p["r"],
                p["n"],
                p["seed"]
            )

            if not 0 < p["gamma"] < 1:
                raise ValueError(
                    "γ має бути в інтервалі (0, 1)."
                )

            if not 2 <= p["k"] <= 30:
                raise ValueError(
                    "k має бути від 2 до 30."
                )

            return p

        except (ValueError, OverflowError) as exc:
            raise ValueError(
                f"Некоректні параметри: {exc}"
            ) from exc

    def generate(self):
        try:
            p = self._get_settings()

            self.df = generate_sample(
                p["mx"],
                p["my"],
                p["sx"],
                p["sy"],
                p["r"],
                p["n"],
                p["seed"]
            )

            self._recalculate(p)

            self.status.set(
                f"Згенеровано {len(self.df)} пар. "
                f"Seed = {p['seed']}."
            )

        except Exception as exc:
            messagebox.showerror(
                "Помилка генерації",
                str(exc)
            )

    def load(self):
        filename = filedialog.askopenfilename(
            filetypes=[
                ("Таблиці", "*.csv *.txt *.xlsx *.xls"),
                ("Усі файли", "*.*")
            ]
        )

        if not filename:
            return

        try:
            p = self._get_settings()

            self.df = load_sample(
                filename
            )

            self.vars["n"].set(
                str(len(self.df))
            )

            self._recalculate(p)

            self.status.set(
                f"Завантажено {len(self.df)} пар "
                f"із файлу {filename}"
            )

        except Exception as exc:
            messagebox.showerror(
                "Помилка завантаження",
                str(exc)
            )

    def save(self):
        if self.df is None:
            messagebox.showwarning(
                "Немає даних",
                "Спочатку згенеруйте або завантажте вибірку."
            )
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile="variant7_sample.csv",
            filetypes=[
                ("CSV", "*.csv"),
                ("TXT", "*.txt"),
                ("Excel", "*.xlsx")
            ]
        )

        if filename:
            try:
                save_sample(
                    self.df,
                    filename
                )

                self.status.set(
                    f"Вибірку збережено: {filename}"
                )

            except Exception as exc:
                messagebox.showerror(
                    "Помилка збереження",
                    str(exc)
                )

    def refresh(self):
        if self.df is None:
            return

        try:
            p = self._get_settings()

            self._recalculate(p)

            self.status.set(
                f"Результати перераховано: "
                f"γ = {p['gamma']}, "
                f"k = {p['k']}."
            )

        except Exception as exc:
            messagebox.showerror(
                "Помилка перерахунку",
                str(exc)
            )

    def _recalculate(self, p):
        self.stats = calculate(
            self.df
        )

        self._fill_data()
        self._fill_stats(p)
        self._fill_tests(p)

        self._draw(
            self._active_tab()
        )

    def _fill_data(self):
        self.data_tree.delete(
            *self.data_tree.get_children()
        )

        pairs = self.df[["X", "Y"]].itertuples(
            index=False,
            name=None
        )

        for i, (x, y) in enumerate(pairs, 1):
            self.data_tree.insert(
                "",
                "end",
                values=(
                    i,
                    f"{x:.6f}",
                    f"{y:.6f}"
                )
            )

        self.data_caption.configure(
            text=(
                f"Кількість спостережень: "
                f"{len(self.df)}. "
                "Варіант 7."
            )
        )

    def _fill_stats(self, p):
        s = self.stats

        ci = confidence_intervals(
            s,
            p["gamma"]
        )

        self.stats_tree.delete(
            *self.stats_tree.get_children()
        )

        rows = [
            (
                "Середнє X",
                s.x_mean,
                p["mx"],
                "x_mean"
            ),

            (
                "Середнє Y",
                s.y_mean,
                p["my"],
                "y_mean"
            ),

            (
                "Дисперсія X",
                s.x_var,
                p["sx"] ** 2,
                "x_var"
            ),

            (
                "Дисперсія Y",
                s.y_var,
                p["sy"] ** 2,
                "y_var"
            ),

            (
                "Стандартне відхилення X",
                s.x_sd,
                p["sx"],
                None
            ),

            (
                "Стандартне відхилення Y",
                s.y_sd,
                p["sy"],
                None
            ),

            (
                "Коваріація X, Y",
                s.covariance,
                p["r"] * p["sx"] * p["sy"],
                None
            ),

            (
                "Кореляція r",
                s.correlation,
                p["r"],
                None
            )
        ]

        for name, estimate, target, key in rows:
            if key:
                lo, hi = ci[key]

                interval = (
                    f"[{lo:.4f}; {hi:.4f}]"
                )

                covered = (
                    "Так"
                    if lo <= target <= hi
                    else "Ні"
                )

            else:
                interval = "—"
                covered = "—"

            self.stats_tree.insert(
                "",
                "end",
                values=(
                    name,
                    f"{estimate:.5f}",
                    f"{target:.5f}",
                    interval,
                    covered
                )
            )

        self.stats_caption.configure(
            text=(
                f"N = {s.n}, "
                f"γ = {p['gamma']:.2f}, "
                f"α = {1 - p['gamma']:.2f}. "
                "Для середніх використано розподіл "
                "Стьюдента, для дисперсій — "
                "розподіл χ²."
            )
        )

    def _fill_tests(self, p):
        s = self.stats

        tests = mean_tests(
            s,
            p["gamma"],
            p["mx"],
            p["my"]
        )

        chi = chi_square_test(
            self.df,
            s,
            p["gamma"],
            p["k"]
        )

        self.t_tree.delete(
            *self.t_tree.get_children()
        )

        self.chi_tree.delete(
            *self.chi_tree.get_children()
        )

        for test in tests:
            decision = (
                "Відхиляємо H₀"
                if test["reject"]
                else "Не відхиляємо H₀"
            )

            self.t_tree.insert(
                "",
                "end",
                values=(
                    test["name"],
                    f"{test['t']:.5f}",
                    f"від {-test['critical']:.5f} до {test['critical']:.5f}",
                    f"{test['p']:.5f}",
                    decision
                )
            )

        chi_rows = [
            (
                "Кількість груп після об'єднання",
                str(chi["groups"])
            ),

            (
                "Мінімальна очікувана частота",
                f"{chi['minimum_expected']:.3f}"
            ),

            (
                "χ² спостережуване",
                f"{chi['chi']:.5f}"
            ),

            (
                "Ступені свободи ν",
                str(chi["df"])
            ),

            (
                "χ² критичне",
                f"{chi['critical']:.5f}"
                if chi["critical"] is not None
                else "—"
            ),

            (
                "p-value",
                f"{chi['p']:.5f}"
                if chi["p"] is not None
                else "—"
            ),

            (
                "Рішення",
                "Недостатньо груп"
                if chi["reject"] is None
                else (
                    "Відхиляємо H₀"
                    if chi["reject"]
                    else "Не відхиляємо H₀"
                )
            )
        ]

        for metric, value in chi_rows:
            self.chi_tree.insert(
                "",
                "end",
                values=(metric, value)
            )

        self.tests_caption.configure(
            text=(
                f"Рівень довіри γ = {p['gamma']:.2f}; "
                f"рівень значущості "
                f"α = {1 - p['gamma']:.2f}. "
                "T-критерій перевіряє відповідність "
                "вибіркових середніх заданим значенням. "
                "Критерій χ² оцінює узгодженість вибірки "
                "з відтвореним двовимірним нормальним "
                "розподілом (наближена перевірка)."
            )
        )

    def _active_tab(self):
        index = self.notebook.index(
            self.notebook.select()
        )

        return list(
            self.tabs.keys()
        )[index]

    def _tab_changed(self, event=None):
        if self.df is not None:
            try:
                self._draw(
                    self._active_tab()
                )

            except Exception as exc:
                self.status.set(
                    f"Помилка побудови графіка: {exc}"
                )

    def _draw(self, key):
        if self.df is None:
            return

        if key not in (
            "corr",
            "heat",
            "bars",
            "density"
        ):
            return

        try:
            k = int(
                self.vars["k"].get()
            )

            if not 2 <= k <= 30:
                raise ValueError(
                    "k має бути від 2 до 30."
                )

            if key == "corr":
                fig = correlation_plot(
                    self.df,
                    self.stats,
                    contours=True
                )

            elif key == "heat":
                fig = heatmap(
                    self.df,
                    k,
                    relative=self.relative.get()
                )

            elif key == "bars":
                fig = bars3d(
                    self.df,
                    k
                )

            else:
                fig = density_plot(
                    self.df,
                    self.stats
                )

            if key not in self.chart_frames:
                self.chart_frames[key] = ttk.Frame(
                    self.tabs[key]
                )

                self.chart_frames[key].pack(
                    fill="both",
                    expand=True
                )

            holder = self.chart_frames[key]

            for child in holder.winfo_children():
                child.destroy()

            canvas = FigureCanvasTkAgg(
                fig,
                master=holder
            )

            canvas.draw()

            canvas.get_tk_widget().pack(
                fill="both",
                expand=True
            )

            toolbar = NavigationToolbar2Tk(
                canvas,
                holder,
                pack_toolbar=False
            )

            toolbar.update()

            toolbar.pack(
                side="bottom",
                fill="x"
            )

            self.canvases[key] = canvas

        except Exception as exc:
            messagebox.showerror(
                "Помилка графіка",
                str(exc)
            )


if __name__ == "__main__":
    app = Lab3App()
    app.mainloop()
