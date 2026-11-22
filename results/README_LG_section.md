## Generalization to a second cell (LG 18650HG2)

`notebooks/08_generalization_LG_HG2.ipynb` repeats the unseen-temperature protocol on a second cell
(LG 18650HG2, 3 Ah NMC) with **unchanged hyperparameters**. All models are trained from scratch on this cell
(training Mixed 1–7, validation Mixed 8, at 40/25/10/0/−10 °C; test UDDS, HWFET, LA92 and US06 at six temperatures;
−20 °C is never used for training, model selection or tuning; seeds 101/202/303). The notebook runs on Google Colab
(GPU); the Drive folder layout is described in its first cell.

Data (not included): P. Kollmeyer, M. Naguib, and M. Skells, *LG 18650HG2 Li-ion Battery Data*, Mendeley Data, V2,
2020, doi: 10.17632/b5mj79w5w9.2.

Results: `results/generalization_LG_HG2.xlsx` (sheets: Summary, By temperature, Significance, R0, Files,
File mapping, Raw rows). These are the values reported in Section III-H and Table 6 of the article.
