#!/usr/bin/env python3
"""PCA utilities for Sentinel-3 OLCI payload telemetry analysis.

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
class AnalysisConfig:
    mission: str = "S3A"
    payload: str = "olci"
    input_path: Path = Path(".")
    out_path: Path = Path("figures/Task2")
    start_year: int = 2017
    end_year: int = 2024
    text_start_year: int = 2017
    text_end_year: int = 2025
    n_components: int = 10
    pc: int = 1
    detrend_data: bool = True
    zoom_year: int = 2024
    outlier_year: int = 2024
    product_column: str = "CCB0241K_AVG"
    calcurve_column: str = "CCB0102X_AVG"
    corr_factor: float = 0.15
    strict_sigma: float = 4.0
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
class AnalysisResult:
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
    """Clean one raw product table and select the requested year interval."""
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
) -> pd.DataFrame:
    """Load and merge all numeric product tables used for Task 2."""
    filenames = [
        f"temperature_{payload}_{mission}.csv",
        f"temperature_{payload}_{mission}_1.csv",
        f"energy_{payload}_{mission}.csv",
        f"voltage_{payload}_{mission}.csv",
    ]
    tables = [
        data_fixing(pd.read_csv(_csv_path(input_path, filename)), start, end)
        for filename in filenames
    ]

    df = pd.concat(tables, axis=1, join="inner")
    df = df.loc[:, df.nunique() > 1]
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df.index = pd.to_datetime(df.index)
    return df


def text_calcurve_preprocessing(
    mission: str,
    input_path: str | Path = ".",
    start: int = 2017,
    end: int = 2025,
    payload: str = "olci",
) -> pd.DataFrame:
    """Load and clean text calibration curves."""
    filename = f"textcalcurve_{payload}_{mission}.csv"
    text_calcurve = pd.read_csv(_csv_path(input_path, filename))
    text_calcurve = data_fixing(text_calcurve, start, end)
    text_calcurve = text_calcurve.loc[:, text_calcurve.nunique() > 1]
    text_calcurve = text_calcurve.loc[:, ~text_calcurve.columns.duplicated()].copy()
    text_calcurve.index = pd.to_datetime(text_calcurve.index)
    return text_calcurve


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


def plot_pca_loadings(
    loadings: pd.DataFrame,
    pc: int | str = 1,
    out_path: str | Path = "figures/Task2",
    show: bool = False,
) -> Path:
    """Plot variable loadings for one selected principal component."""
    pc_name = _pc_name(pc)
    series = loadings[pc_name]
    filename = _as_path(out_path) / f"loadings_{pc_name.lower()}.png"

    plt.figure(figsize=(22, 10))
    plt.bar(series.index, series.values, color="steelblue", edgecolor="black", linewidth=0.5)
    plt.axhline(0, color="black", linewidth=1)
    plt.axhline(0.15, color="maroon", linestyle="--", linewidth=1.7)
    plt.axhline(-0.15, color="maroon", linestyle="--", linewidth=1.7)
    plt.ylim(-0.3, 0.3)
    plt.xlabel("Variables", fontsize=25)
    plt.ylabel("Loading value", fontsize=25)
    plt.xticks(rotation=45, ha="right", size=18)
    plt.yticks(size=20)
    plt.grid(axis="y", alpha=0.3)
    _savefig(filename, show=show)
    return filename


def plot_pc_timeseries(
    ts: pd.Series,
    pc: int = 1,
    out_path: str | Path = "figures/Task2",
    detrend: bool = False,
    zoom_year: int = 2022,
    show: bool = False,
) -> Path:
    """Plot a PCA score over the full period and a selected year."""
    suffix = "_detrended" if detrend else ""
    ts = ts.copy()
    ts.index = pd.to_datetime(ts.index)
    if detrend:
        ts = pd.Series(scipy_detrend(ts.to_numpy()), index=ts.index)

    ts_zoom = ts.loc[ts.index.year == zoom_year]
    filename = _as_path(out_path) / f"scores_pc{pc}{suffix}_zoom_{zoom_year}.png"

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=False)
    axes[0].plot(ts.index.to_numpy(), ts.to_numpy(), linewidth=0.8, color="steelblue")
    axes[0].set_yscale("symlog", linthresh=10)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Time", fontsize=18)
    axes[0].set_ylabel("Score", fontsize=18)
    axes[0].tick_params(axis="both", labelsize=13)
    axes[0].grid(alpha=0.3)

    axes[1].plot(ts_zoom.index.to_numpy(), ts_zoom.to_numpy(), linewidth=0.8, color="steelblue")
    axes[1].set_yscale("symlog", linthresh=10)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Time", fontsize=20)
    axes[1].set_ylabel("Score", fontsize=20)
    axes[1].set_title(f"Zoom: {zoom_year}", fontsize=22)
    axes[1].tick_params(axis="both", labelsize=16)
    axes[1].grid(alpha=0.3)
    _savefig(filename, show=show)
    return filename


def outliers_sigma(df: pd.DataFrame, col: str, k: float) -> pd.Series:
    """Detect rare events after detrending a selected score column."""
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


def outlier_detection_and_plot(
    df: pd.DataFrame,
    col: str,
    outliers: pd.Series,
    year: int,
    pc: int = 1,
    out_path: str | Path = "figures/Task2",
    detrend: bool = False,
    show: bool = False,
) -> Path:
    """Plot one variable and highlight the PCA outlier timestamps."""
    if col not in df.columns:
        raise KeyError(f"Column not found: {col}")

    suffix = "_detrended" if detrend else ""
    ts = df[col].copy()
    ts.index = pd.to_datetime(ts.index)
    outlier_index = pd.to_datetime(outliers.index).intersection(ts.index)
    ts_out = ts.loc[outlier_index]
    ts_year = ts.loc[f"{year}-01-01" : f"{year}-12-31"]
    ts_out_year = ts_year.loc[outlier_index.intersection(ts_year.index)]
    filename = _as_path(out_path) / f"outliers_{col}_pc{pc}_full_and_{year}{suffix}.png"

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)
    axes[0].plot(ts.index.to_numpy(), ts.to_numpy(), linewidth=0.8, color="steelblue", label="Time series")
    if not ts_out.empty:
        axes[0].scatter(ts_out.index.to_numpy(), ts_out.to_numpy(), color="red", s=30, label="Outliers", zorder=3)
    axes[0].set_title(f"{col} - full time series{suffix}", fontsize=25)
    axes[0].set_xlabel("Time", fontsize=20)
    axes[0].set_ylabel("Value", fontsize=20)
    axes[0].tick_params(axis="both", labelsize=16)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ts_year.index.to_numpy(), ts_year.to_numpy(), linewidth=0.8, color="steelblue", label="Time series")
    if not ts_out_year.empty:
        axes[1].scatter(
            ts_out_year.index.to_numpy(),
            ts_out_year.to_numpy(),
            color="red",
            s=30,
            label="Outliers",
            zorder=3,
        )
    axes[1].set_title(f"{col} - {year}{suffix}", fontsize=25)
    axes[1].set_xlabel("Time", fontsize=20)
    axes[1].set_ylabel("Value", fontsize=20)
    axes[1].tick_params(axis="both", labelsize=16)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    _savefig(filename, show=show)
    return filename


def aligning_text_and_scores(textcalcurves: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Align text calibration curves with PCA score timestamps."""
    common_index = scores.index.intersection(textcalcurves.index)
    scores_aligned = scores.loc[common_index].sort_index()
    state_aligned = textcalcurves.loc[common_index].sort_index()
    return scores_aligned.join(state_aligned, how="inner").dropna(axis=0, how="any")


def ts_creation(ts: pd.Series, detrend: bool = False) -> tuple[pd.Series, float]:
    """Prepare a regular time series and return its sampling interval in seconds."""
    ts = ts.copy()
    ts.index = pd.to_datetime(ts.index)
    ts = ts.sort_index()
    if len(ts) < 2:
        raise ValueError("At least two samples are required for frequency analysis.")
    if detrend:
        ts = pd.Series(scipy_detrend(ts.to_numpy()), index=ts.index)
    dt = (ts.index[1] - ts.index[0]).total_seconds()
    return ts, dt


def select_fft_series(
    scores: pd.DataFrame,
    explained: np.ndarray,
    pc: int = 1,
    combo: bool = False,
) -> pd.Series:
    """Select a PC score or weighted PC1-PC3 combination for FFT."""
    if combo:
        if scores.shape[1] < 3 or len(explained) < 3:
            raise ValueError("The weighted FFT combination requires at least three PCs.")
        return (
            scores["PC1"].dropna() * explained[0]
            + scores["PC2"].dropna() * explained[1]
            + scores["PC3"].dropna() * explained[2]
        ).copy()
    return scores[f"PC{pc}"].dropna()


def plot_power_spectrum(
    ts: pd.Series,
    dt: float,
    combo: bool,
    detrend: bool,
    pc: int = 1,
    remove_annual: bool = False,
    out_path: str | Path = "figures/Task2",
    show: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute and plot the power spectrum of a PCA score or score combination."""
    detrend_label = "detrended" if detrend else ""
    annual_label = "removed_annual" if remove_annual else ""
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
    if remove_annual and len(peaks) > 3:
        peaks = peaks[np.argsort(power[peaks])[-3:]]

    plt.figure(figsize=(10, 5))
    plt.plot(period_days, power, linewidth=1, color="steelblue")
    plt.scatter(period_days[peaks], power[peaks], color="red", zorder=3)
    for peak in peaks:
        period = period_days[peak]
        label = f"{period * 24:.1f} h" if period < 1 else f"{period:.1f} d"
        plt.annotate(label, (period_days[peak], power[peak]), textcoords="offset points", xytext=(5, 5), fontsize=12)
    plt.xscale("log")
    plt.yscale("log")
    plt.tick_params(axis="both", labelsize=16)
    plt.xlabel("Period [days]", fontsize=20)
    plt.ylabel("Power", fontsize=20)
    plt.grid(True, which="both", alpha=0.3)

    if combo:
        filename = _as_path(out_path) / f"fft_{detrend_label}_{annual_label}_power_spectrum_combo.png"
    else:
        filename = _as_path(out_path) / f"fft_{detrend_label}_{annual_label}_power_spectrum_pca{pc}.png"
    _savefig(filename, show=show)
    return power, peaks, period_days


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


def plot_stl(
    ts: pd.Series,
    combo: bool,
    detrend: bool,
    pc: int = 1,
    remove_annual: bool = False,
    component: str = "residuals",
    out_path: str | Path = "figures/Task2",
    show: bool = False,
) -> STL:
    """Run STL and plot one selected decomposition component."""
    detrend_label = "detrended" if detrend else ""
    annual_label = "removed_annual_semiannual" if remove_annual else ""
    if remove_annual:
        ts = remove_annual_semiannual_cycles(ts)

    q01, q99 = ts.quantile([0.01, 0.99])
    ts_clean = ts.clip(lower=q01, upper=q99)
    result = STL(ts_clean, period=288, robust=True).fit()

    component = component.lower()
    if component in {"trend", "low_frequency"}:
        y = result.trend
        component_label = "Trend / low-frequency component"
        component_name = "trend"
    elif component in {"seasonality", "seasonal", "daily_seasonality"}:
        y = result.seasonal
        component_label = "Daily seasonality"
        component_name = "seasonality"
    elif component in {"residuals", "residual", "resid"}:
        y = result.resid
        component_label = "Residuals"
        component_name = "residuals"
    else:
        raise ValueError("component must be one of: 'trend', 'seasonality', or 'residuals'")

    plt.figure(figsize=(14, 5))
    plt.plot(y.index.to_numpy(), y.to_numpy(), linewidth=1, color="steelblue")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title(component_label, fontsize=25)
    plt.xlabel("Time", fontsize=20)
    plt.ylabel("Value", fontsize=20)
    plt.xticks(size=18)
    plt.yticks(size=18)
    plt.grid(True, alpha=0.3)

    if combo:
        filename = _as_path(out_path) / f"fft_{detrend_label}_{annual_label}_STL_combo_{component_name}.png"
    else:
        filename = _as_path(out_path) / f"fft_{detrend_label}_{annual_label}_STL_pca{pc}_{component_name}.png"
    _savefig(filename, show=show)
    return result


def _default_sigma(pc: int) -> float:
    return 1.6 if pc == 1 else 2.5


def _plot_column_if_available(
    df: pd.DataFrame,
    col: str,
    outliers: pd.Series,
    year: int,
    pc: int,
    out_path: Path,
    detrend: bool,
    show: bool,
) -> None:
    if col in df.columns:
        outlier_detection_and_plot(df, col, outliers, year, pc, out_path, detrend, show)
    else:
        warnings.warn(f"Column not available, skipping plot: {col}", stacklevel=2)


def run_analysis(config: AnalysisConfig | None = None, verbose: bool = True) -> AnalysisResult:
    """Run the complete Task 2 analysis pipeline."""
    config = config or AnalysisConfig()
    out_path = _as_path(config.out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    df = product_data_opening(
        config.payload,
        config.mission,
        config.start_year,
        config.end_year,
        config.input_path,
    )
    if config.detrend_data:
        df = detrend_dataframe(df)

    text_calcurves = text_calcurve_preprocessing(
        config.mission,
        config.input_path,
        config.text_start_year,
        config.text_end_year,
        config.payload,
    )

    pca_result = pca_execution(df, config.n_components, config.mission, verbose=verbose)
    pc_name = _pc_name(config.pc)
    if pc_name not in pca_result.scores.columns:
        raise ValueError(f"{pc_name} is not available. PCA produced {len(pca_result.scores.columns)} components.")

    plot_pca_loadings(pca_result.loadings, config.pc, out_path, config.show_plots)
    plot_pc_timeseries(
        pca_result.scores[pc_name],
        config.pc,
        out_path,
        detrend=False,
        zoom_year=config.zoom_year,
        show=config.show_plots,
    )

    outliers = outliers_sigma(pca_result.scores, pc_name, _default_sigma(config.pc))
    strict_outliers = outliers_sigma(pca_result.scores, pc_name, config.strict_sigma)
    correlated, anticorrelated = get_loading_clusters(pca_result.loadings, config.pc, config.corr_factor)

    if verbose:
        print(f"PCA{config.pc} - anticorrelated parameters: {list(anticorrelated)}")
        print(f"PCA{config.pc} - correlated parameters: {list(correlated)}")

    _plot_column_if_available(
        df,
        config.product_column,
        outliers,
        config.outlier_year,
        config.pc,
        out_path,
        config.detrend_data,
        config.show_plots,
    )

    pca_with_state = aligning_text_and_scores(text_calcurves, pca_result.scores)
    _plot_column_if_available(
        pca_with_state,
        config.calcurve_column,
        outliers,
        config.outlier_year,
        config.pc,
        out_path,
        False,
        config.show_plots,
    )
    _plot_column_if_available(
        pca_with_state,
        config.calcurve_column,
        strict_outliers,
        config.outlier_year,
        config.pc,
        out_path,
        False,
        config.show_plots,
    )

    ts = select_fft_series(pca_result.scores, pca_result.explained, config.pc, config.combo_fft)
    ts_fixed, dt = ts_creation(ts, config.detrend_fft)
    power, peaks, period_days = plot_power_spectrum(
        ts_fixed,
        dt,
        config.combo_fft,
        config.detrend_fft,
        config.pc,
        out_path=out_path,
        show=config.show_plots,
    )
    plot_power_spectrum(
        ts_fixed,
        dt,
        config.combo_fft,
        config.detrend_fft,
        config.pc,
        remove_annual=True,
        out_path=out_path,
        show=config.show_plots,
    )

    for component in ("residuals", "seasonality", "trend"):
        plot_stl(
            ts,
            combo=config.combo_fft,
            detrend=config.detrend_fft,
            pc=config.pc,
            component=component,
            out_path=out_path,
            show=config.show_plots,
        )

    return AnalysisResult(
        data=df,
        text_calcurves=text_calcurves,
        pca=pca_result,
        outliers=outliers,
        strict_outliers=strict_outliers,
        correlated_variables=correlated,
        anticorrelated_variables=anticorrelated,
        pca_with_state=pca_with_state,
        fft_power=power,
        fft_peaks=peaks,
        fft_period_days=period_days,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Sentinel-3 OLCI payload PCA analysis.")
    parser.add_argument("--mission", default="S3A")
    parser.add_argument("--payload", default="olci")
    parser.add_argument("--input-path", default=".")
    parser.add_argument("--out-path", default="figures/Task2")
    parser.add_argument("--start-year", type=int, default=2017)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--text-start-year", type=int, default=2017)
    parser.add_argument("--text-end-year", type=int, default=2025)
    parser.add_argument("--n-components", type=int, default=10)
    parser.add_argument("--pc", type=int, default=1)
    parser.add_argument("--zoom-year", type=int, default=2024)
    parser.add_argument("--outlier-year", type=int, default=2024)
    parser.add_argument("--product-column", default="CCB0241K_AVG")
    parser.add_argument("--calcurve-column", default="CCB0102X_AVG")
    parser.add_argument("--corr-factor", type=float, default=0.15)
    parser.add_argument("--strict-sigma", type=float, default=4.0)
    parser.add_argument("--combo-fft", action="store_true")
    parser.add_argument("--detrend-fft", action="store_true")
    parser.add_argument("--show-plots", action="store_true")
    parser.add_argument("--no-detrend-data", action="store_true")
    return parser


def parse_args(argv: Iterable[str] | None = None) -> AnalysisConfig:
    args = build_parser().parse_args(argv)
    return AnalysisConfig(
        mission=args.mission,
        payload=args.payload,
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
        product_column=args.product_column,
        calcurve_column=args.calcurve_column,
        corr_factor=args.corr_factor,
        strict_sigma=args.strict_sigma,
        combo_fft=args.combo_fft,
        detrend_fft=args.detrend_fft,
        show_plots=args.show_plots,
    )


def main(argv: Iterable[str] | None = None) -> AnalysisResult:
    config = parse_args(argv)
    return run_analysis(config)


if __name__ == "__main__":
    main()
