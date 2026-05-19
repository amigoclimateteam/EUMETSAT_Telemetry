#!/usr/bin/env python3
"""Multi-mission PCA utilities for Sentinel-3 OLCI telemetry analysis.

The module is import-safe: no files are read and no figures are generated until
``run_analysis`` or ``main`` is called.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq, rfft, rfftfreq
from scipy.signal import detrend as scipy_detrend
from scipy.signal import find_peaks, windows
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL


@dataclass(frozen=True)
class MultiMissionConfig:
    payload: str = "olci"
    mission1: str = "S3A"
    mission2: str = "S3B"
    mission_name1: str = "Sentinel-3A"
    mission_name2: str = "Sentinel-3B"
    input_path: Path = Path(".")
    out_path: Path = Path("figures/Task3")
    start_year: int = 2019
    end_year: int = 2025
    text_start_year: int = 2018
    text_end_year: int = 2025
    n_components: int = 10
    pc: int = 1
    detrend_data: bool = True
    zoom_year: int = 2024
    outlier_year: int = 2024
    outlier_sigma: float = 4.0
    strict_sigma: float = 8.0
    corr_factor: float = 0.15
    product_column: str = ""
    calcurve_column: str = "CCB0102X_AVG"
    strict_calcurve_index: int = 19
    strict_calcurve_year: int = 2025
    combo_fft: bool = False
    detrend_fft: bool = False
    show_plots: bool = False


@dataclass
class PCAResult:
    explained: np.ndarray
    loadings: pd.DataFrame
    scores: pd.DataFrame
    scaler: StandardScaler
    model: PCA


@dataclass
class MissionAnalysis:
    data: pd.DataFrame
    text_calcurves: pd.DataFrame
    pca: PCAResult
    outliers: pd.Series
    strict_outliers: pd.Series
    correlated_variables: dict[str, float]
    anticorrelated_variables: dict[str, float]
    pca_with_state: pd.DataFrame
    fft_power: np.ndarray
    fft_peaks: np.ndarray
    fft_period_days: np.ndarray


@dataclass
class MultiMissionResult:
    mission1: MissionAnalysis
    mission2: MissionAnalysis
    common_correlated_variables: list[str]
    common_text_calcurves: list[str]


def _as_path(path: str | Path) -> Path:
    return Path(path).expanduser()


def _csv_path(input_path: str | Path, filename: str) -> Path:
    return _as_path(input_path) / filename


def _savefig(path: str | Path, show: bool = False) -> None:
    path = _as_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close()


def _pc_name(pc: int | str) -> str:
    if isinstance(pc, str):
        return pc if pc.upper().startswith("PC") else f"PC{pc}"
    return f"PC{pc}"


def _safe_filename(value: str) -> str:
    return value.replace(" ", "_").replace("/", "_")


def column_renaming(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize CSV columns produced by the telemetry extraction."""
    df = df.reset_index()
    old_cols = list(df.columns)
    real_cols = [col for col in old_cols if not str(col).startswith("level_")]

    if len(real_cols) < 2:
        raise ValueError("Expected at least a time column and a row-count column.")

    time_col = real_cols[0]
    rowcount_col = real_cols[1]
    var_cols = real_cols[2:]
    new_cols: list[str] = []
    var_idx = 0

    for idx, _ in enumerate(old_cols):
        if idx == 0:
            new_cols.append(time_col)
        elif idx == 1:
            new_cols.append(rowcount_col)
        elif idx % 2 == 0 and var_idx < len(var_cols):
            new_cols.append(var_cols[var_idx])
            var_idx += 1
        else:
            new_cols.append(f"Unnamed: {idx}")

    fixed = df.copy()
    fixed.columns = new_cols
    return fixed


def data_fixing(ds: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """Clean one raw telemetry table and select the requested year interval."""
    ds = column_renaming(ds)
    ds = ds.rename(
        columns={
            "Rowcount S3_TM_HKP2": "Rowcount",
            "Rowcount S3_TM_HKP1": "Rowcount",
        }
    )

    if "Rowcount" not in ds.columns:
        raise ValueError("Missing Rowcount column after column normalization.")

    rowcount = ds["Rowcount"]
    lower = rowcount.mean() - rowcount.std()
    upper = rowcount.mean() + rowcount.std()
    ds = ds.loc[rowcount.between(lower, upper)]
    ds = ds.loc[:, ~ds.columns.str.contains("Unnamed:", regex=False)]
    ds = ds.drop(columns=["Rowcount"]).dropna()

    if "Time (ISO8601)" not in ds.columns:
        raise ValueError("Missing 'Time (ISO8601)' column.")

    ds = ds.copy()
    ds["Time (ISO8601)"] = pd.to_datetime(ds["Time (ISO8601)"], errors="coerce")
    ds = ds.dropna(subset=["Time (ISO8601)"])
    ds = ds.rename(columns={"Time (ISO8601)": "time"}).set_index("time")
    return ds.loc[f"{start}-01-01" : f"{end}-12-31"]


def product_data_opening(
    payload: str,
    mission: str,
    start: int,
    end: int,
    input_path: str | Path = ".",
    verbose: bool = True,
) -> pd.DataFrame:
    """Load and merge the temperature telemetry files for one mission."""
    filenames = [
        f"temperature_{payload}_{mission}.csv",
        f"temperature_{payload}_{mission}_1.csv",
    ]
    tables = [
        data_fixing(pd.read_csv(_csv_path(input_path, filename)), start, end)
        for filename in filenames
    ]

    df = pd.concat(tables, axis=1, join="inner")
    df = df.loc[:, df.nunique() > 1]
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df.index = pd.to_datetime(df.index)

    if verbose:
        if len(df.columns) == 40:
            print(f"{mission}: temperature sensor count is 40.")
        else:
            print(f"{mission}: check temperature sensor count ({len(df.columns)} columns).")

    return df


def text_calcurve_preprocessing(
    mission: str,
    input_path: str | Path = ".",
    start: int = 2018,
    end: int = 2025,
    payload: str = "olci",
) -> pd.DataFrame:
    """Load and clean text calibration curves for one mission."""
    filenames = [
        f"textcalcurve_{payload}_{mission}.csv",
        f"textcalcurve_{payload}_{mission}_1.csv",
    ]
    tables = []
    for filename in filenames:
        path = _csv_path(input_path, filename)
        if path.exists():
            tables.append(data_fixing(pd.read_csv(path), start, end))

    if not tables:
        raise FileNotFoundError(
            f"No text calibration curve CSV files found for {mission} in {_as_path(input_path)}."
        )

    text_calcurves = pd.concat(tables, axis=1, join="inner")
    text_calcurves = text_calcurves.loc[:, text_calcurves.nunique() > 1]
    text_calcurves = text_calcurves.loc[:, ~text_calcurves.columns.duplicated()].copy()
    text_calcurves.index = pd.to_datetime(text_calcurves.index)
    return text_calcurves


def detrend_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the linear trend from all numeric columns."""
    df_out = df.copy()
    numeric_cols = df_out.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df_out.loc[:, numeric_cols] = scipy_detrend(
            df_out.loc[:, numeric_cols].to_numpy(),
            axis=0,
        )
    return df_out


def pca_execution(
    df: pd.DataFrame,
    n_components: int,
    mission_name: str,
    verbose: bool = True,
) -> PCAResult:
    """Standardize numerical columns and run PCA."""
    x_df = df.select_dtypes(include=[np.number]).dropna(axis=0, how="any")
    if x_df.empty:
        raise ValueError("No numeric data available for PCA.")

    n_components = min(n_components, x_df.shape[0], x_df.shape[1])
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_df.to_numpy())

    pca = PCA(n_components=n_components, random_state=0)
    x_pca = pca.fit_transform(x_scaled)
    explained = pca.explained_variance_ratio_

    if verbose:
        print(f"OLCI | Mission {mission_name} - variances explained:", explained)
        print(f"OLCI | Mission {mission_name} - cumulative variances:", np.cumsum(explained))

    columns = [f"PC{i + 1}" for i in range(n_components)]
    loadings = pd.DataFrame(pca.components_.T, index=x_df.columns, columns=columns)
    scores = pd.DataFrame(x_pca, index=x_df.index, columns=columns)
    return PCAResult(explained, loadings, scores, scaler, pca)


def plot_pca_loadings_double(
    loadings1: pd.DataFrame,
    loadings2: pd.DataFrame,
    mission_name1: str,
    mission_name2: str,
    out_path: str | Path = "figures/Task3",
    pc: int | str = 1,
    show: bool = False,
) -> Path:
    """Compare the loadings of the same PC for two missions."""
    pc_name = _pc_name(pc)
    series1 = loadings1[pc_name]
    series2 = loadings2[pc_name]
    filename = _as_path(out_path) / f"loadings_{pc_name}_comparison.png"

    fig, axes = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    configs = [
        (axes[0], series1, "steelblue", mission_name1),
        (axes[1], series2, "darkorange", mission_name2),
    ]

    for ax, series, color, title in configs:
        ax.bar(series.index, series.values, color=color, edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=1)
        ax.axhline(0.15, color="maroon", linewidth=1.7, linestyle="--")
        ax.axhline(-0.15, color="maroon", linewidth=1.7, linestyle="--")
        ax.set_ylim(-0.3, 0.3)
        ax.set_ylabel("Loading value", fontsize=20)
        ax.set_title(title, fontsize=25)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="y", labelsize=18)

    plt.xticks(rotation=45, ha="right", size=18)
    axes[1].set_xlabel("Variables", fontsize=20)
    _savefig(filename, show=show)
    return filename


def plot_pc_timeseries_double(
    ts1: pd.Series,
    ts2: pd.Series,
    mission_name1: str,
    mission_name2: str,
    year: int,
    out_path: str | Path = "figures/Task3",
    pc: int | str = 1,
    detrend: bool = False,
    show: bool = False,
) -> Path:
    """Plot full and yearly PCA score time series for two missions."""
    pc_name = _pc_name(pc)

    def preprocess(ts: pd.Series) -> np.ndarray:
        values = ts.to_numpy()
        if not detrend:
            return values
        mask = np.isfinite(values)
        detrended = np.full_like(values, np.nan, dtype=float)
        if mask.sum() > 1:
            detrended[mask] = scipy_detrend(values[mask], type="linear")
        return detrended

    ts1 = ts1.copy()
    ts2 = ts2.copy()
    ts1.index = pd.to_datetime(ts1.index)
    ts2.index = pd.to_datetime(ts2.index)
    ts1_zoom = ts1.loc[ts1.index.year == year]
    ts2_zoom = ts2.loc[ts2.index.year == year]

    suffix = "_detrended" if detrend else ""
    filename = _as_path(out_path) / f"scores_{pc_name}_comparison_full_and_{year}{suffix}.png"

    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=False)
    title_suffix = " (detrended)" if detrend else ""
    plot_data = [
        (axes[0, 0], ts1, preprocess(ts1), "steelblue", f"{mission_name1}{title_suffix}"),
        (axes[0, 1], ts2, preprocess(ts2), "darkorange", f"{mission_name2}{title_suffix}"),
        (axes[1, 0], ts1_zoom, preprocess(ts1_zoom), "steelblue", f"{mission_name1} zoom {year}{title_suffix}"),
        (axes[1, 1], ts2_zoom, preprocess(ts2_zoom), "darkorange", f"{mission_name2} zoom {year}{title_suffix}"),
    ]

    for ax, ts, values, color, title in plot_data:
        ax.plot(ts.index.to_numpy(), values, linewidth=0.8, color=color)
        ax.set_title(title, fontsize=25)
        ax.set_yscale("symlog", linthresh=10)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_ylabel("Score", fontsize=20)
        ax.tick_params(axis="both", labelsize=18)
        ax.grid(alpha=0.3)

    axes[1, 0].set_xlabel("Time", fontsize=20)
    axes[1, 1].set_xlabel("Time", fontsize=20)
    _savefig(filename, show=show)
    return filename


def outliers_sigma(df: pd.DataFrame, col: str, k: float) -> pd.Series:
    """Detect rare events after detrending a selected PCA score."""
    ts_detrended = pd.Series(scipy_detrend(df[col].to_numpy()), index=df.index)
    mean = ts_detrended.mean()
    std = ts_detrended.std()
    return ts_detrended.loc[(ts_detrended < mean - k * std) | (ts_detrended > mean + k * std)]


def get_loading_clusters(
    loadings: pd.DataFrame,
    pc: int | str = 1,
    corr_factor: float = 0.15,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return positively and negatively loaded variables for a PC."""
    pc_name = _pc_name(pc)
    anticorrelated = loadings.loc[loadings[pc_name] < -corr_factor, pc_name].to_dict()
    correlated = loadings.loc[loadings[pc_name] > corr_factor, pc_name].to_dict()
    return correlated, anticorrelated


def outlier_detection_and_plot_double_ts(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    col: str,
    outliers1: pd.Series,
    outliers2: pd.Series,
    mission_name1: str,
    mission_name2: str,
    year: int,
    pc: int | str = 1,
    out_path: str | Path = "figures/Task3",
    show: bool = False,
) -> Path:
    """Plot one common variable for two missions and highlight PCA outlier timestamps."""
    if col not in df1.columns or col not in df2.columns:
        raise KeyError(f"Column must exist in both mission dataframes: {col}")

    ts1 = df1[col].copy()
    ts2 = df2[col].copy()
    ts1.index = pd.to_datetime(ts1.index)
    ts2.index = pd.to_datetime(ts2.index)

    outlier_index1 = pd.to_datetime(outliers1.index).intersection(ts1.index)
    outlier_index2 = pd.to_datetime(outliers2.index).intersection(ts2.index)
    ts1_out = ts1.loc[outlier_index1]
    ts2_out = ts2.loc[outlier_index2]
    ts1_year = ts1.loc[ts1.index.year == year]
    ts2_year = ts2.loc[ts2.index.year == year]
    ts1_out_year = ts1_year.loc[outlier_index1.intersection(ts1_year.index)]
    ts2_out_year = ts2_year.loc[outlier_index2.intersection(ts2_year.index)]

    name1 = _safe_filename(mission_name1)
    name2 = _safe_filename(mission_name2)
    filename = _as_path(out_path) / f"{name1}_{name2}_outliers_{col}_{_pc_name(pc)}_full_and_{year}.png"

    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=False)
    plot_data = [
        (axes[0, 0], ts1, ts1_out, "steelblue", f"{mission_name1} - {col} full time series"),
        (axes[0, 1], ts2, ts2_out, "darkorange", f"{mission_name2} - {col} full time series"),
        (axes[1, 0], ts1_year, ts1_out_year, "steelblue", f"{mission_name1} - {col} zoom {year}"),
        (axes[1, 1], ts2_year, ts2_out_year, "darkorange", f"{mission_name2} - {col} zoom {year}"),
    ]

    for ax, ts, ts_out, color, title in plot_data:
        ax.plot(ts.index.to_numpy(), ts.to_numpy(), linewidth=0.8, color=color, label="Time series")
        if not ts_out.empty:
            ax.scatter(ts_out.index.to_numpy(), ts_out.to_numpy(), color="red", s=30, label="Outliers", zorder=3)
        ax.set_title(title, fontsize=25)
        ax.set_xlabel("Time", fontsize=20)
        ax.set_ylabel("Value", fontsize=20)
        ax.tick_params(axis="both", labelsize=18)
        ax.grid(True, alpha=0.3)
        ax.legend()

    _savefig(filename, show=show)
    return filename


def aligning_text_and_scores(textcalcurves: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Align text calibration curves with PCA score timestamps."""
    common_index = scores.index.intersection(textcalcurves.index)
    scores_aligned = scores.loc[common_index].sort_index()
    state_aligned = textcalcurves.loc[common_index].sort_index()
    return scores_aligned.join(state_aligned, how="inner").dropna(axis=0, how="any")


def retrieve_ts(
    scores: pd.DataFrame,
    explained: np.ndarray,
    pc: int = 1,
    combo: bool = False,
    detrend: bool = False,
) -> tuple[pd.Series, float]:
    """Select one PC score or a weighted PC combination and return its sampling interval."""
    if combo:
        if scores.shape[1] < 3 or len(explained) < 3:
            raise ValueError("The weighted FFT combination requires at least three PCs.")
        ts = (
            scores["PC1"].dropna() * explained[0]
            + scores["PC2"].dropna() * explained[1]
            + scores["PC3"].dropna() * explained[2]
        ).copy()
    else:
        ts = scores[f"PC{pc}"].dropna()

    ts.index = pd.to_datetime(ts.index)
    ts = ts.sort_index()
    if len(ts) < 2:
        raise ValueError("At least two samples are required for frequency analysis.")
    if detrend:
        ts = pd.Series(scipy_detrend(ts.to_numpy()), index=ts.index)
    dt = (ts.index[1] - ts.index[0]).total_seconds()
    return ts, dt


def compute_spectrum(
    ts: pd.Series,
    dt: float,
    remove_annual: bool = False,
    max_peaks_after_annual_removal: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute positive-frequency period spectrum and peak indices."""
    signal = ts.to_numpy()
    n_samples = len(signal)

    if remove_annual:
        signal = (signal - np.mean(signal)) * windows.hann(n_samples)
        yf = rfft(signal)
        xf = rfftfreq(n_samples, d=dt)
    else:
        yf = fft(signal - np.mean(signal))
        xf = fftfreq(n_samples, dt)

    power = np.abs(yf) ** 2
    mask = xf > 0
    freq = xf[mask]
    power = power[mask]
    period_days = 1 / freq / 86400
    order = np.argsort(period_days)
    period_days = period_days[order]
    power = power[order]

    if remove_annual:
        mask = (period_days > 0.1) & (period_days < 100)
        period_days = period_days[mask]
        power = power[mask]

    peaks, _ = find_peaks(power, prominence=np.max(power) * 0.05)
    if remove_annual and len(peaks) > max_peaks_after_annual_removal:
        peaks = peaks[np.argsort(power[peaks])[-max_peaks_after_annual_removal:]]
    return period_days, power, peaks


def plot_power_spectrum_double(
    ts1: pd.Series,
    ts2: pd.Series,
    dt1: float,
    dt2: float,
    mission_name1: str,
    mission_name2: str,
    combo: bool,
    detrend: bool,
    pc: int = 1,
    remove_annual: bool = False,
    out_path: str | Path = "figures/Task3",
    show: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute and compare power spectra for two mission time series."""
    period1, power1, peaks1 = compute_spectrum(ts1, dt1, remove_annual)
    period2, power2, peaks2 = compute_spectrum(ts2, dt2, remove_annual)
    detrend_label = "detrended" if detrend else ""
    annual_label = "removed_annual" if remove_annual else ""

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    configs = [
        (axes[0], period1, power1, peaks1, "steelblue", mission_name1),
        (axes[1], period2, power2, peaks2, "darkorange", mission_name2),
    ]

    for ax, period_days, power, peaks, color, mission in configs:
        ax.plot(period_days, power, linewidth=1, color=color)
        ax.scatter(period_days[peaks], power[peaks], color="red", zorder=3)
        plotted_labels = set()
        for peak in peaks:
            period = period_days[peak]
            label = f"{period * 24:.1f} h" if period < 1 else f"{period:.1f} d"
            if label in plotted_labels:
                continue
            plotted_labels.add(label)
            ax.annotate(label, (period_days[peak], power[peak]), textcoords="offset points", xytext=(5, 5), fontsize=13)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel("Power", fontsize=20)
        ax.set_title(mission, fontsize=20)
        ax.grid(True, which="both", alpha=0.3)

    axes[1].set_xlabel("Period [days]", fontsize=20)
    plt.tick_params(axis="both", labelsize=16)

    if combo:
        filename = _as_path(out_path) / f"fft_{detrend_label}_{annual_label}_power_spectrum_combo_comparison.png"
    else:
        filename = _as_path(out_path) / f"fft_{detrend_label}_{annual_label}_power_spectrum_pca{pc}_comparison.png"
    _savefig(filename, show=show)
    return power1, peaks1, period1, power2, peaks2, period2


def remove_annual_semiannual_cycles(ts: pd.Series) -> pd.Series:
    """Remove annual and semi-annual harmonic components from a time series."""
    ts = ts.copy()
    ts.index = pd.to_datetime(ts.index)
    mask = ts.notna()
    y = ts.loc[mask].to_numpy()
    t = (ts.loc[mask].index - ts.loc[mask].index[0]).total_seconds() / 86400

    annual_period = 365.25
    semiannual_period = annual_period / 2
    design = np.column_stack(
        [
            np.ones(len(t)),
            np.sin(2 * np.pi * t / annual_period),
            np.cos(2 * np.pi * t / annual_period),
            np.sin(2 * np.pi * t / semiannual_period),
            np.cos(2 * np.pi * t / semiannual_period),
        ]
    )
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)

    ts_removed = ts.copy()
    ts_removed.loc[mask] = y - design @ beta
    return ts_removed


def _compute_stl_component(ts: pd.Series, component: str):
    ts = ts.copy()
    ts.index = pd.to_datetime(ts.index)
    q01, q99 = ts.quantile([0.01, 0.99])
    ts_clean = ts.clip(lower=q01, upper=q99)
    result = STL(ts_clean, period=288, robust=True).fit()

    component = component.lower()
    if component in {"trend", "low_frequency"}:
        return result.trend, "Trend / low-frequency component", result, "trend"
    if component in {"seasonality", "seasonal", "daily_seasonality"}:
        return result.seasonal, "Daily seasonality", result, "seasonality"
    if component in {"residuals", "residual", "resid"}:
        return result.resid, "Residuals", result, "residuals"
    raise ValueError("component must be one of: 'trend', 'seasonality', or 'residuals'")


def plot_stl_double(
    ts1: pd.Series,
    ts2: pd.Series,
    combo: bool,
    detrend: bool,
    mission_name1: str = "Sentinel-3A",
    mission_name2: str = "Sentinel-3B",
    pc: int = 1,
    remove_annual: bool = False,
    component: str = "residuals",
    out_path: str | Path = "figures/Task3",
    show: bool = False,
):
    """Run STL decomposition for two missions and plot one selected component."""
    detrend_label = "detrended" if detrend else ""
    annual_label = "removed_annual_semiannual" if remove_annual else ""
    if remove_annual:
        ts1 = remove_annual_semiannual_cycles(ts1)
        ts2 = remove_annual_semiannual_cycles(ts2)

    y1, component_label, result1, component_name = _compute_stl_component(ts1, component)
    y2, _, result2, _ = _compute_stl_component(ts2, component)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"hspace": 0.2})

    axes[0].plot(y1.index.to_numpy(), y1.to_numpy(), linewidth=1, color="steelblue", label=component_label)
    axes[0].set_title(mission_name1, fontsize=22)
    axes[0].set_ylabel("Value", fontsize=20)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(y2.index.to_numpy(), y2.to_numpy(), linewidth=1, color="darkorange", label=component_label)
    axes[1].set_title(mission_name2, fontsize=22)
    axes[1].set_ylabel("Value", fontsize=20)
    axes[1].set_xlabel("Time", fontsize=20)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    name1 = _safe_filename(mission_name1)
    name2 = _safe_filename(mission_name2)
    if combo:
        filename = _as_path(out_path) / f"fft_{detrend_label}_{annual_label}_STL_combo_{component_name}_{name1}_{name2}.png"
    else:
        filename = _as_path(out_path) / f"fft_{detrend_label}_{annual_label}_STL_pca{pc}_{component_name}_{name1}_{name2}.png"

    plt.xticks(size=18)
    plt.yticks(size=18)
    _savefig(filename, show=show)
    return result1, result2


def _plot_common_column_if_available(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    col: str,
    outliers1: pd.Series,
    outliers2: pd.Series,
    config: MultiMissionConfig,
    year: int,
) -> None:
    if col and col in df1.columns and col in df2.columns:
        outlier_detection_and_plot_double_ts(
            df1,
            df2,
            col,
            outliers1,
            outliers2,
            config.mission_name1,
            config.mission_name2,
            year,
            config.pc,
            config.out_path,
            config.show_plots,
        )
    elif col:
        warnings.warn(f"Column not available in both missions, skipping plot: {col}", stacklevel=2)


def run_analysis(config: MultiMissionConfig | None = None, verbose: bool = True) -> MultiMissionResult:
    """Run the complete Task 3 multi-mission comparison pipeline."""
    config = config or MultiMissionConfig()
    out_path = _as_path(config.out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    df1 = product_data_opening(
        config.payload,
        config.mission1,
        config.start_year,
        config.end_year,
        config.input_path,
        verbose,
    )
    df2 = product_data_opening(
        config.payload,
        config.mission2,
        config.start_year,
        config.end_year,
        config.input_path,
        verbose,
    )

    if config.detrend_data:
        df1 = detrend_dataframe(df1)
        df2 = detrend_dataframe(df2)

    text1 = text_calcurve_preprocessing(
        config.mission1,
        config.input_path,
        config.text_start_year,
        config.text_end_year,
        config.payload,
    )
    text2 = text_calcurve_preprocessing(
        config.mission2,
        config.input_path,
        config.text_start_year,
        config.text_end_year,
        config.payload,
    )

    pca1 = pca_execution(df1, config.n_components, config.mission_name1, verbose)
    pca2 = pca_execution(df2, config.n_components, config.mission_name2, verbose)
    pc_name = _pc_name(config.pc)
    if pc_name not in pca1.scores.columns or pc_name not in pca2.scores.columns:
        raise ValueError(f"{pc_name} is not available for both missions.")

    plot_pca_loadings_double(
        pca1.loadings,
        pca2.loadings,
        config.mission_name1,
        config.mission_name2,
        out_path,
        config.pc,
        config.show_plots,
    )
    plot_pc_timeseries_double(
        pca1.scores[pc_name],
        pca2.scores[pc_name],
        config.mission_name1,
        config.mission_name2,
        config.zoom_year,
        out_path,
        config.pc,
        detrend=False,
        show=config.show_plots,
    )

    outliers1 = outliers_sigma(pca1.scores, pc_name, config.outlier_sigma)
    outliers2 = outliers_sigma(pca2.scores, pc_name, config.outlier_sigma)
    strict_outliers1 = outliers_sigma(pca1.scores, pc_name, config.strict_sigma)
    strict_outliers2 = outliers_sigma(pca2.scores, pc_name, config.strict_sigma)

    correlated1, anticorrelated1 = get_loading_clusters(pca1.loadings, config.pc, config.corr_factor)
    correlated2, anticorrelated2 = get_loading_clusters(pca2.loadings, config.pc, config.corr_factor)
    common_correlated = sorted(set(correlated1) & set(correlated2))

    if verbose:
        print(f"{config.mission_name1} has {len(outliers1)} outliers in {config.start_year}-{config.end_year}.")
        print(f"{config.mission_name2} has {len(outliers2)} outliers in {config.start_year}-{config.end_year}.")
        print(f"PCA{config.pc} - {config.mission_name1}, anti-correlated parameters: {list(anticorrelated1)}")
        print(f"PCA{config.pc} - {config.mission_name1}, correlated parameters: {list(correlated1)}")
        print(f"PCA{config.pc} - {config.mission_name2}, anti-correlated parameters: {list(anticorrelated2)}")
        print(f"PCA{config.pc} - {config.mission_name2}, correlated parameters: {list(correlated2)}")

    product_column = config.product_column or (common_correlated[0] if common_correlated else "")
    _plot_common_column_if_available(
        df1,
        df2,
        product_column,
        outliers1,
        outliers2,
        config,
        config.outlier_year,
    )

    pca_with_state1 = aligning_text_and_scores(text1, pca1.scores)
    pca_with_state2 = aligning_text_and_scores(text2, pca2.scores)
    common_text = sorted(
        set(pca_with_state1.iloc[:, len(pca1.scores.columns) :].columns)
        & set(pca_with_state2.iloc[:, len(pca2.scores.columns) :].columns)
    )

    _plot_common_column_if_available(
        pca_with_state1,
        pca_with_state2,
        config.calcurve_column,
        outliers1,
        outliers2,
        config,
        config.outlier_year,
    )

    if common_text:
        strict_col = (
            common_text[config.strict_calcurve_index]
            if config.strict_calcurve_index < len(common_text)
            else common_text[-1]
        )
        _plot_common_column_if_available(
            pca_with_state1,
            pca_with_state2,
            strict_col,
            strict_outliers1,
            strict_outliers2,
            config,
            config.strict_calcurve_year,
        )

    ts1, dt1 = retrieve_ts(pca1.scores, pca1.explained, config.pc, config.combo_fft, config.detrend_fft)
    ts2, dt2 = retrieve_ts(pca2.scores, pca2.explained, config.pc, config.combo_fft, config.detrend_fft)
    power1, peaks1, period1, power2, peaks2, period2 = plot_power_spectrum_double(
        ts1,
        ts2,
        dt1,
        dt2,
        config.mission_name1,
        config.mission_name2,
        config.combo_fft,
        config.detrend_fft,
        config.pc,
        out_path=out_path,
        show=config.show_plots,
    )
    plot_power_spectrum_double(
        ts1,
        ts2,
        dt1,
        dt2,
        config.mission_name1,
        config.mission_name2,
        config.combo_fft,
        config.detrend_fft,
        config.pc,
        remove_annual=True,
        out_path=out_path,
        show=config.show_plots,
    )

    for component in ("residuals", "trend", "seasonality"):
        plot_stl_double(
            ts1,
            ts2,
            config.combo_fft,
            config.detrend_fft,
            config.mission_name1,
            config.mission_name2,
            config.pc,
            component=component,
            out_path=out_path,
            show=config.show_plots,
        )

    mission1 = MissionAnalysis(
        data=df1,
        text_calcurves=text1,
        pca=pca1,
        outliers=outliers1,
        strict_outliers=strict_outliers1,
        correlated_variables=correlated1,
        anticorrelated_variables=anticorrelated1,
        pca_with_state=pca_with_state1,
        fft_power=power1,
        fft_peaks=peaks1,
        fft_period_days=period1,
    )
    mission2 = MissionAnalysis(
        data=df2,
        text_calcurves=text2,
        pca=pca2,
        outliers=outliers2,
        strict_outliers=strict_outliers2,
        correlated_variables=correlated2,
        anticorrelated_variables=anticorrelated2,
        pca_with_state=pca_with_state2,
        fft_power=power2,
        fft_peaks=peaks2,
        fft_period_days=period2,
    )
    return MultiMissionResult(mission1, mission2, common_correlated, common_text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Sentinel-3 OLCI multi-mission PCA comparison.")
    parser.add_argument("--payload", default="olci")
    parser.add_argument("--mission1", default="S3A")
    parser.add_argument("--mission2", default="S3B")
    parser.add_argument("--mission-name1", default="Sentinel-3A")
    parser.add_argument("--mission-name2", default="Sentinel-3B")
    parser.add_argument("--input-path", default=".")
    parser.add_argument("--out-path", default="figures/Task3")
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--text-start-year", type=int, default=2018)
    parser.add_argument("--text-end-year", type=int, default=2025)
    parser.add_argument("--n-components", type=int, default=10)
    parser.add_argument("--pc", type=int, default=1)
    parser.add_argument("--zoom-year", type=int, default=2024)
    parser.add_argument("--outlier-year", type=int, default=2024)
    parser.add_argument("--outlier-sigma", type=float, default=4.0)
    parser.add_argument("--strict-sigma", type=float, default=8.0)
    parser.add_argument("--corr-factor", type=float, default=0.15)
    parser.add_argument("--product-column", default="")
    parser.add_argument("--calcurve-column", default="CCB0102X_AVG")
    parser.add_argument("--strict-calcurve-index", type=int, default=19)
    parser.add_argument("--strict-calcurve-year", type=int, default=2025)
    parser.add_argument("--combo-fft", action="store_true")
    parser.add_argument("--detrend-fft", action="store_true")
    parser.add_argument("--show-plots", action="store_true")
    parser.add_argument("--no-detrend-data", action="store_true")
    return parser


def parse_args(argv: Iterable[str] | None = None) -> MultiMissionConfig:
    args = build_parser().parse_args(argv)
    return MultiMissionConfig(
        payload=args.payload,
        mission1=args.mission1,
        mission2=args.mission2,
        mission_name1=args.mission_name1,
        mission_name2=args.mission_name2,
        input_path=Path(args.input_path),
        out_path=Path(args.out_path),
        start_year=args.start_year,
        end_year=args.end_year,
        text_start_year=args.text_start_year,
        text_end_year=args.text_end_year,
        n_components=args.n_components,
        pc=args.pc,
        detrend_data=not args.no_detrend_data,
        zoom_year=args.zoom_year,
        outlier_year=args.outlier_year,
        outlier_sigma=args.outlier_sigma,
        strict_sigma=args.strict_sigma,
        corr_factor=args.corr_factor,
        product_column=args.product_column,
        calcurve_column=args.calcurve_column,
        strict_calcurve_index=args.strict_calcurve_index,
        strict_calcurve_year=args.strict_calcurve_year,
        combo_fft=args.combo_fft,
        detrend_fft=args.detrend_fft,
        show_plots=args.show_plots,
    )


def main(argv: Iterable[str] | None = None) -> MultiMissionResult:
    config = parse_args(argv)
    return run_analysis(config)


if __name__ == "__main__":
    main()
