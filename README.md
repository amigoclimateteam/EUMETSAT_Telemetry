# EUMETSAT Telemetry Analysis

Python utilities for Sentinel-3 OLCI payload telemetry analysis. The scripts
load telemetry CSV files, clean and align the time series, run PCA, detect
outliers, and write diagnostic plots under an output directory.

## Repository Contents

- `Telemetry_Task2_Single_payload_analysis_12.py`: import-safe library and CLI
  for single-payload OLCI PCA analysis.
- `Telemetry_Task3_Multi_mission_analysis_22.py`: import-safe library and CLI
  for multi-mission OLCI PCA comparison.

## Requirements

Use a Python environment with the scientific stack installed:

```bash
pip install numpy pandas scipy scikit-learn statsmodels matplotlib
```

## Input Files

Task 2 expects these CSV files in the selected input directory:

```text
temperature_<payload>_<mission>.csv
temperature_<payload>_<mission>_1.csv
energy_<payload>_<mission>.csv
voltage_<payload>_<mission>.csv
textcalcurve_<payload>_<mission>.csv
```

Task 3 expects temperature files for both missions:

```text
temperature_<payload>_<mission1>.csv
temperature_<payload>_<mission1>_1.csv
temperature_<payload>_<mission2>.csv
temperature_<payload>_<mission2>_1.csv
```

Task 3 also loads text calibration curve files for both missions. The `_1` file
is used when present:

```text
textcalcurve_<payload>_<mission1>.csv
textcalcurve_<payload>_<mission1>_1.csv
textcalcurve_<payload>_<mission2>.csv
textcalcurve_<payload>_<mission2>_1.csv
```

With the default arguments, `<payload>` is `olci`, `<mission>` is `S3A` for Task
2, and Task 3 compares `S3A` against `S3B`.

## Task 2 Execution

From the repository directory, run:

```bash
python Telemetry_Task2_Single_payload_analysis_12.py \
  --input-path . \
  --out-path figures/Task2
```

This runs the full analysis using the default configuration:

- mission: `S3A`
- payload: `olci`
- analysis years: `2017` to `2024`
- text calibration curve years: `2017` to `2025`
- PCA components: `10`
- selected principal component: `PC1`
- output directory: `figures/Task2`

Figures are saved to the output directory. Plots are not displayed interactively
unless `--show-plots` is passed.

### Task 2 Arguments

Common examples:

```bash
# Analyze a different mission.
python Telemetry_Task2_Single_payload_analysis_12.py --mission S3B

# Use CSV files from a dedicated data directory.
python Telemetry_Task2_Single_payload_analysis_12.py --input-path data

# Change the output directory.
python Telemetry_Task2_Single_payload_analysis_12.py --out-path results/task2

# Analyze PC2 with 6 PCA components.
python Telemetry_Task2_Single_payload_analysis_12.py --pc 2 --n-components 6

# Disable detrending before PCA.
python Telemetry_Task2_Single_payload_analysis_12.py --no-detrend-data

# Use a weighted FFT combination of PC1, PC2, and PC3.
python Telemetry_Task2_Single_payload_analysis_12.py --combo-fft

# Show plots while also saving them.
python Telemetry_Task2_Single_payload_analysis_12.py --show-plots
```

Available CLI options:

```text
--mission              Mission name, default: S3A
--payload              Payload name, default: olci
--input-path           Directory containing input CSV files, default: .
--out-path             Directory for generated figures, default: figures/Task2
--start-year           First product-data year, default: 2017
--end-year             Last product-data year, default: 2024
--text-start-year      First text-calibration year, default: 2017
--text-end-year        Last text-calibration year, default: 2025
--n-components         Number of PCA components, default: 10
--pc                   Principal component to inspect, default: 1
--zoom-year            Year used in score zoom plots, default: 2024
--outlier-year         Year used in outlier zoom plots, default: 2024
--product-column       Product variable plotted with outliers
--calcurve-column      Text calibration curve plotted with outliers
--corr-factor          Loading threshold for variable clusters, default: 0.15
--strict-sigma         Stricter outlier sigma threshold, default: 4.0
--combo-fft            Use weighted PC1-PC3 combination for FFT
--detrend-fft          Detrend the selected FFT time series
--show-plots           Display plots interactively
--no-detrend-data      Do not detrend product variables before PCA
```

## Task 3 Execution

From the repository directory, run:

```bash
python Telemetry_Task3_Multi_mission_analysis_22.py \
  --input-path . \
  --out-path figures/Task3
```

This runs the full multi-mission comparison using the default configuration:

- payload: `olci`
- mission 1: `S3A`
- mission 2: `S3B`
- analysis years: `2019` to `2025`
- text calibration curve years: `2018` to `2025`
- PCA components: `10`
- selected principal component: `PC1`
- output directory: `figures/Task3`

Figures are saved to the output directory. Plots are not displayed interactively
unless `--show-plots` is passed.

### Task 3 Arguments

Common examples:

```bash
# Compare different missions.
python Telemetry_Task3_Multi_mission_analysis_22.py --mission1 S3A --mission2 S3B

# Use CSV files from a dedicated data directory.
python Telemetry_Task3_Multi_mission_analysis_22.py --input-path data

# Change the output directory.
python Telemetry_Task3_Multi_mission_analysis_22.py --out-path results/task3

# Analyze PC2 with 6 PCA components.
python Telemetry_Task3_Multi_mission_analysis_22.py --pc 2 --n-components 6

# Set a specific telemetry variable for the shared outlier plot.
python Telemetry_Task3_Multi_mission_analysis_22.py --product-column CCB0241K_AVG

# Tune the outlier thresholds.
python Telemetry_Task3_Multi_mission_analysis_22.py --outlier-sigma 4 --strict-sigma 8

# Disable detrending before PCA.
python Telemetry_Task3_Multi_mission_analysis_22.py --no-detrend-data

# Use a weighted FFT combination of PC1, PC2, and PC3.
python Telemetry_Task3_Multi_mission_analysis_22.py --combo-fft
```

Available CLI options:

```text
--payload                  Payload name, default: olci
--mission1                 First mission identifier, default: S3A
--mission2                 Second mission identifier, default: S3B
--mission-name1            First mission display name, default: Sentinel-3A
--mission-name2            Second mission display name, default: Sentinel-3B
--input-path               Directory containing input CSV files, default: .
--out-path                 Directory for generated figures, default: figures/Task3
--start-year               First product-data year, default: 2019
--end-year                 Last product-data year, default: 2025
--text-start-year          First text-calibration year, default: 2018
--text-end-year            Last text-calibration year, default: 2025
--n-components             Number of PCA components, default: 10
--pc                       Principal component to inspect, default: 1
--zoom-year                Year used in score zoom plots, default: 2024
--outlier-year             Year used in outlier zoom plots, default: 2024
--outlier-sigma            Main outlier sigma threshold, default: 4.0
--strict-sigma             Stricter outlier sigma threshold, default: 8.0
--corr-factor              Loading threshold for variable clusters, default: 0.15
--product-column           Product variable plotted with outliers
--calcurve-column          Text calibration curve plotted with outliers
--strict-calcurve-index    Common calibration curve index for strict plot, default: 19
--strict-calcurve-year     Year used in strict calibration plot, default: 2025
--combo-fft                Use weighted PC1-PC3 combination for FFT
--detrend-fft              Detrend the selected FFT time series
--show-plots               Display plots interactively
--no-detrend-data          Do not detrend product variables before PCA
```

## Library Usage

The Task 2 script can also be imported without running the analysis:

```python
from pathlib import Path

from Telemetry_Task2_Single_payload_analysis_12 import AnalysisConfig, run_analysis

config = AnalysisConfig(
    mission="S3A",
    payload="olci",
    input_path=Path("data"),
    out_path=Path("figures/Task2"),
    pc=1,
)

result = run_analysis(config)

print(result.pca.explained)
print(result.correlated_variables)
```

Task 3 can be used the same way:

```python
from pathlib import Path

from Telemetry_Task3_Multi_mission_analysis_22 import MultiMissionConfig, run_analysis

config = MultiMissionConfig(
    mission1="S3A",
    mission2="S3B",
    input_path=Path("data"),
    out_path=Path("figures/Task3"),
    pc=1,
)

result = run_analysis(config)

print(result.mission1.pca.explained)
print(result.mission2.pca.explained)
print(result.common_correlated_variables)
```

Importing either module does not read CSV files, create directories, or generate
figures. Those actions happen only when `run_analysis()` or the CLI entrypoint is
called.
