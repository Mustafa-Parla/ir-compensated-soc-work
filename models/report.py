"""Özet tablolar (Excel) ve yayın kalitesinde grafikler (V26)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

from models.engine import ci95
from models.experiment import BASE_SPECS, SHORT
from data_utils import TEMPS

OURS = 'Önerilen (RC-Hibrit)'
MODEL_ORDER = list(BASE_SPECS) + ['2RC-ECM + EKF']
METRICS = ['rmse', 'mae', 'maxe', 'r2', 'jitter', 'rmse_ema', 'mae_ema', 'maxe_ema', 'jitter_ema']


def _pct(df, cols):
    df = df.copy()
    for c in cols:
        if c in df and c not in ('r2', 'r2_ema'): df[c] = df[c] * 100
    return df


def seed_level(df, groups=('model',), subset=None):
    d = df if subset is None else df[subset(df)]
    cols = [c for c in METRICS if c in d]
    return _pct(d.groupby(list(groups) + ['seed'])[cols].mean().reset_index(), cols)


def mean_std_ci(sl, groups=('model',)):
    cols = [c for c in METRICS if c in sl]
    g = sl.groupby(list(groups))[cols]
    out = g.mean().add_suffix('_ort').join(g.std().add_suffix('_std')).join(g.agg(ci95).add_suffix('_GA95'))
    out['n_seed'] = sl.groupby(list(groups)).seed.nunique()
    return out[[f'{c}_{s}' for c in cols for s in ('ort', 'std', 'GA95')] + ['n_seed']]


def main_table(final):
    parts = {'Eğitim sıcaklıkları (25/10/0/−10°C)': lambda d: d.temp != '-20degC', 'Görülmemiş −20°C': lambda d: d.temp == '-20degC',
             'Tümü': lambda d: d.temp == d.temp}
    out = []
    for k, f in parts.items():
        t = mean_std_ci(seed_level(final, subset=f)); t.insert(0, 'kapsam', k); out.append(t)
    t = pd.concat(out).reset_index()
    present = list(dict.fromkeys(t.model))
    t['model'] = pd.Categorical(t.model, [m for m in MODEL_ORDER if m in present] + [m for m in present if m not in MODEL_ORDER], ordered=True)
    return t.sort_values(['kapsam', 'model'])


def by_key(final, key):
    sl = _pct(final.groupby(['model', 'seed', key])[['rmse', 'mae', 'rmse_ema']].mean().reset_index(), ['rmse', 'mae', 'rmse_ema'])
    g = sl.groupby(['model', key]).rmse
    return pd.DataFrame({'RMSE_ort': g.mean(), 'RMSE_std': g.std(), 'RMSE_GA95': g.agg(ci95)}).reset_index()


def per_file(final):
    g = (final.groupby(['temp', 'cycle', 'model']).rmse.agg(['mean', 'std']) * 100).reset_index()
    piv = g.pivot_table(index=['temp', 'cycle'], columns='model', values='mean')
    piv['kazanan'] = piv.idxmin(1)
    return g, piv


def significance(final, ref=OURS):
    """Hücre düzeyinde (dosya, seed ort.) Wilcoxon ve seed düzeyinde Welch: ref vs her model."""
    cell = final.groupby(['model', 'temp', 'cycle']).rmse.mean() * 100
    sl = seed_level(final, subset=lambda d: d.temp != '-20degC').set_index(['model', 'seed']).rmse
    rows = []
    for m in [x for x in MODEL_ORDER if x in set(final.model) and x != ref]:
        for scope, f in [('20 dosya', lambda i: True), ('16 dosya (−20 hariç)', lambda i: i[0] != '-20degC')]:
            a, b = cell.loc[ref], cell.loc[m]
            idx = [i for i in a.index if f(i) and i in b.index]
            d = a.loc[idx] - b.loc[idx]
            rows.append({'karşılaştırma': f'{ref} vs {m}', 'kapsam': scope, 'Δ RMSE ort (puan)': d.mean(), 'ref daha iyi (dosya)': f'{(d < 0).sum()}/{len(d)}',
                         'Wilcoxon p': stats.wilcoxon(d).pvalue if len(d) > 5 and (d != 0).any() else np.nan})
        if m in sl.index.get_level_values(0):
            rows.append({'karşılaştırma': f'{ref} vs {m}', 'kapsam': 'seed düzeyi (−20 hariç), Welch',
                         'Δ RMSE ort (puan)': sl.loc[ref].mean() - sl.loc[m].mean(),
                         'Wilcoxon p': stats.ttest_ind(sl.loc[ref], sl.loc[m], equal_var=False).pvalue})
    return pd.DataFrame(rows)


def significance_cold(final, ref=OURS, temp='-20degC'):
    """Görülmemiş sıcaklık için ayrı test: seed düzeyi Welch (n=5/5) ve seed×çevrim eşleştirilmiş Wilcoxon (n=20)."""
    m = final[final.temp == temp]
    sl = m.groupby(['model', 'seed']).rmse.mean() * 100
    cell = m.groupby(['model', 'seed', 'cycle']).rmse.mean() * 100
    rows = []
    for k in [x for x in MODEL_ORDER if x in set(m.model) and x != ref]:
        a = cell.loc[ref]; b = cell.loc[k].reindex(a.index); d = (a - b).dropna()
        rows.append({'karşılaştırma': f'{ref} vs {k}', 'kapsam': f'{temp} (görülmemiş)', 'Δ RMSE ort (puan)': sl.loc[ref].mean() - sl.loc[k].mean(),
                     'Welch p (seed)': stats.ttest_ind(sl.loc[ref], sl.loc[k], equal_var=False).pvalue,
                     'ref daha iyi (seed×çevrim)': f'{(d < 0).sum()}/{len(d)}',
                     'Wilcoxon p (seed×çevrim)': stats.wilcoxon(d).pvalue if (d != 0).any() else np.nan})
    return pd.DataFrame(rows)


def rc_param_table(records, norm, ecm=None):
    tn = lambda T: (T - norm.stats['temp']['min']) / (norm.stats['temp']['max'] - norm.stats['temp']['min'])
    rows = []
    for rec in records:
        p = rec.get('rc_params')
        if not p: continue
        row = {'model': rec['name'], 'seed': rec['seed']}
        for T in [25, 10, 0, -10, -20]:
            row[f'R0 {T}°C (mΩ)'] = np.exp(p['r0_a'] + p['r0_b'] * (tn(T) - 1)) * 1e3
            row[f'ΣRk {T}°C (mΩ)'] = sum(np.exp(a + b * (tn(T) - 1)) for a, b in zip(p['rk_a'], p['rk_b'])) * 1e3
        for j, t in enumerate(p['tau']): row[f'τ{j + 1} (s)'] = t
        rows.append(row)
    t = pd.DataFrame(rows)
    if ecm is not None:
        er = {'model': '2RC-ECM (tanımlanan)', 'seed': np.nan}
        for T in [25, 10, 0, -10, -20]:
            er[f'R0 {T}°C (mΩ)'] = np.exp(ecm['R_fit']['R0'][0] + ecm['R_fit']['R0'][1] * T) * 1e3
            er[f'ΣRk {T}°C (mΩ)'] = sum(np.exp(ecm['R_fit'][k][0] + ecm['R_fit'][k][1] * T) for k in ('R1', 'R2')) * 1e3
        t = pd.concat([t, pd.DataFrame([er])], ignore_index=True)
    return t


# ───────────────────────────── grafikler ─────────────────────────────
def style():
    import matplotlib as mpl
    mpl.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'legend.fontsize': 8, 'figure.dpi': 110,
                         'savefig.dpi': 300, 'axes.grid': True, 'grid.alpha': .3, 'axes.spines.top': False, 'axes.spines.right': False,
                         'font.family': 'DejaVu Sans'})


COLORS = {OURS: '#c0392b', 'Hibrit (GRU→Transformer)': '#2c3e50', 'LSTM': '#2980b9', 'GRU': '#16a085', 'CNN-LSTM': '#8e44ad',
          'Transformer': '#d35400', '2RC-ECM + EKF': '#7f8c8d'}
TCOL = dict(zip(TEMPS, ['#2c3e50', '#2980b9', '#27ae60', '#e67e22', '#c0392b']))


def save(fig, gdir, name):
    gdir = Path(gdir); fig.savefig(gdir / f'{name}.png', bbox_inches='tight'); fig.savefig(gdir / f'{name}.pdf', bbox_inches='tight')
