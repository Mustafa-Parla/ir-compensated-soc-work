# IR-compensated neural SoC estimation at an unseen low temperature

Code and results for the article *Learned temperature-dependent IR compensation for neural state-of-charge estimation at an unseen low temperature* (under review).

A single learnable, temperature-dependent IR-compensation channel, u = V − R0(T)·Ī with R0(T) = exp[a + b(T_n − 1)],
is trained end-to-end from SoC labels only and feeds a GRU→Transformer backbone. On the Panasonic 18650PF data it
keeps 1.8 % RMSE at an unseen −20 °C chamber temperature, where data-driven baselines reach 4.6–12.7 %.

## Data
The raw data are **not** included. Download the *Panasonic 18650PF Li-ion Battery Data* (P. Kollmeyer, Mendeley Data,
doi: 10.17632/wykht8y7tg.1 — please verify the DOI before publication) and place the temperature folders under `data/`
(folder names are listed in `data_utils/files.py`).

## Environment
```
conda env create -f environment.yml
```

## Run order
Open the notebooks from the `notebooks/` folder (ROOT is set to the repository root).
| Notebook | Purpose |
|---|---|
| `00_environment_and_data_check.ipynb` | environment, data loading, file table, window counts |
| `01_hyperparameter_tuning.ipynb` | equal-budget tuning of all networks (validation only) |
| `02_train_baselines.ipynb` | baselines × 5 seeds (+ former RC-hybrid candidate) |
| `03_ablation_without_current_input.ipynb` | variants without current input to the backbone |
| `03b_train_main_model_factorial.ipynb` | **trains the main model** and the factorial current-input control |
| `03c_main_model_ablation_sensitivity.ipynb` | temperature-independent R, h = 64, sensitivity |
| `04_ecm_ekf_baseline.ipynb` | 2RC-ECM identification + EKF |
| `04b_step_resistance_from_data.ipynb` | model-independent step resistance ΔV/ΔI from the raw data |
| `05_robustness_and_leakage_baselines.ipynb` | sensor-fault robustness and leakage audit (baselines) |
| `05c_main_model_evaluation.ipynb` | registers the main model, time series, robustness, efficiency |
| `06_summary_tables_and_figures.ipynb` | all tables (summary_tables.xlsx) and figures of the article |
| `07_latency_measurement.ipynb` | latency of all models re-measured in one session |

The main model is trained in `03b` under the internal name *"RC: yalnız ohmik + akım girişi"* (ohmic-only compensation
with current input to the backbone), selected by the lowest validation error in the factorial experiment, and
registered under the name *"Önerilen (IR-telafili hibrit)"* (Proposed) in `05c`. Its hyperparameters were inherited
from the tuned RC-hybrid and were not tuned separately. The test set was never used for model selection.

## Results
`results/summary_tables.xlsx` contains all tables of the article (T1–T15). Sheet and column names in `results/` are
in English. The code itself writes files with internal (Turkish) names; the mapping is given below and in
`file_name_mapping.csv`. Internal model names are translated in the released tables as follows: see `EXACT` in `release.py`.

## File-name mapping (internal → released)
| Internal | Released |
|---|---|
| `notebooks/00_ortam_kontrol.ipynb` | `notebooks/00_environment_and_data_check.ipynb` |
| `notebooks/01_ayarlama.ipynb` | `notebooks/01_hyperparameter_tuning.ipynb` |
| `notebooks/02_final.ipynb` | `notebooks/02_train_baselines.ipynb` |
| `notebooks/03_ablasyon_duyarlilik.ipynb` | `notebooks/03_ablation_without_current_input.ipynb` |
| `notebooks/03b_akim_girisi.ipynb` | `notebooks/03b_train_main_model_factorial.ipynb` |
| `notebooks/03c_anamodel_C_ablasyon.ipynb` | `notebooks/03c_main_model_ablation_sensitivity.ipynb` |
| `notebooks/04_ekf.ipynb` | `notebooks/04_ecm_ekf_baseline.ipynb` |
| `notebooks/04b_basamak_direnci.ipynb` | `notebooks/04b_step_resistance_from_data.ipynb` |
| `notebooks/05_saglamlik_sizinti.ipynb` | `notebooks/05_robustness_and_leakage_baselines.ipynb` |
| `notebooks/05c_anamodel_C_degerlendirme.ipynb` | `notebooks/05c_main_model_evaluation.ipynb` |
| `notebooks/06c_ozet_grafikler_C.ipynb` | `notebooks/06_summary_tables_and_figures.ipynb` |
| `notebooks/07_gecikme_olcumu.ipynb` | `notebooks/07_latency_measurement.ipynb` |
| `results/excel/V26C_ozet.xlsx` | `results/summary_tables.xlsx` |
| `results/excel/00_veri.xlsx` | `results/00_dataset_files.xlsx` |
| `results/excel/01_ayarlama.xlsx` | `results/01_tuning.xlsx` |
| `results/excel/03b_akim_girisi.xlsx` | `results/03b_factorial_current_input.xlsx` |
| `results/excel/03c_anamodel_C_ablasyon.xlsx` | `results/03c_main_model_ablation.xlsx` |
| `results/excel/04_ekf.xlsx` | `results/04_ecm_ekf.xlsx` |
| `results/excel/04_ekf_satirlar.csv` | `results/04_ecm_ekf_rows.csv` |
| `results/excel/04b_basamak_direnci.xlsx` | `results/04b_step_resistance.xlsx` |
| `results/series/04b_basamak_olaylari.csv.gz` | `results/04b_step_events.csv.gz` |
| `results/excel/05_saglamlik_sizinti.xlsx` | `results/05_robustness_leakage.xlsx` |
| `results/excel/05_saglamlik_satirlar.csv` | `results/05_robustness_rows_baselines.csv` |
| `results/excel/05c_saglamlik_satirlar.csv` | `results/05c_robustness_rows_main.csv` |
| `results/excel/05_sizinti_pencereler.csv` | `results/05_leakage_windows.csv` |
| `results/excel/05c_anamodel_C.xlsx` | `results/05c_main_model.xlsx` |
| `results/excel/07_gecikme.xlsx` | `results/07_latency.xlsx` |
| `results/configs/selected.json` | `results/configs/selected_hyperparameters.json` |
| `results/configs/normalizer.json` | `results/configs/normalizer.json` |
| `results/configs/ecm_parametreleri.json` | `results/configs/ecm_parameters.json` |
| `results/configs/verimlilik.json` | `results/configs/efficiency_baselines.json` |
| `results/configs/verimlilik_c.json` | `results/configs/efficiency_main_model.json` |
| `results/configs/verimlilik_cs.json` | `results/configs/efficiency_main_model_h64.json` |
| `results/configs/verimlilik_yeniden.json` | `results/configs/latency_remeasured.json` |
| `results/configs/ortam.json` | `results/configs/environment.json` |

## License
MIT (see LICENSE). If you use this code, please cite the article (citation will be added after acceptance).
