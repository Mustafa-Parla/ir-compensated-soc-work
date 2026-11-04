"""
06c · Ana model C (IR-telafili hibrit) için özet tablolar ve yayın şekilleri (İngilizce).

Bağımlılık yalnız numpy / pandas / scipy / matplotlib (torch gerekmez). Veri, Experiment'in kayıt klasörlerinden okunur:
  results/runs/<aşama>/*.json, results/series/<aşama>/*.parquet, results/excel/*.csv|xlsx, results/configs/*.json
Kullanım (06c defteri):  from models import c_report as R;  D = R.gather(exp);  tabs = R.tables(D);  R.fig_...(D, G)
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter

# ───────────────────────────── adlar ─────────────────────────────
C = 'Önerilen (IR-telafili hibrit)'
CS = 'C: eşit boyut (h=64)'
RCONST = 'C: R sıcaklıktan bağımsız'
HYB = 'Hibrit (GRU→Transformer)'
OLD = 'Önerilen (RC-Hibrit)'                 # akımsız omurga + 3 RC (eski önerilen)
RCI = 'RC + akım girişi'                     # çok ölçekli RC uzantısı (C + 3 RC kolu)
OHM = 'RC: yalnız ohmik kol'                 # C − omurgaya akım
NOI = 'Akımsız (noI)'
EKF = '2RC-ECM + EKF'
BASE = ['LSTM', 'GRU', 'CNN-LSTM', 'Transformer']
MAIN = [C, CS, HYB] + BASE + [EKF]
SENS = ['C duyarlılık: ohmik yumuşatma yok', 'C duyarlılık: ohmik yumuşatma 5 s', 'C duyarlılık: W = 128']
ABL_ORDER = [C, RCONST, CS, RCI, OHM, OLD, HYB, NOI] + SENS
TEMPS = ['25degC', '10degC', '0degC', '-10degC', '-20degC']
TLAB = {'25degC': '25 °C', '10degC': '10 °C', '0degC': '0 °C', '-10degC': '−10 °C', '-20degC': '−20 °C'}
CYC = ['US06', 'HWFET', 'UDDS', 'LA92']

EN = {C: 'Proposed (IR-compensated hybrid)', CS: 'Proposed-S (h = 64)', HYB: 'Hybrid (GRU→Transformer)', 'LSTM': 'LSTM', 'GRU': 'GRU',
      'CNN-LSTM': 'CNN-LSTM', 'Transformer': 'Transformer', EKF: '2RC-ECM + EKF',
      RCONST: 'Temperature-independent R', RCI: '+ three RC branches (multi-scale extension)', OHM: '− current input to backbone',
      OLD: '− current input, + three RC branches', NOI: 'No compensation, no current input',
      'C duyarlılık: ohmik yumuşatma yok': 'No ohmic smoothing (1 s)', 'C duyarlılık: ohmik yumuşatma 5 s': 'Ohmic smoothing 5 s',
      'C duyarlılık: W = 128': 'Window W = 128 s'}
EN_SHORT = {C: 'Proposed', CS: 'Proposed-S', HYB: 'Hybrid', EKF: '2RC-ECM+EKF'}
ABL_EN = dict(EN); ABL_EN.update({HYB: 'No compensation (= Hybrid)', C: 'Proposed (full)'})
COLORS = {C: '#c0392b', CS: '#e8836f', HYB: '#2c3e50', 'LSTM': '#2980b9', 'GRU': '#16a085', 'CNN-LSTM': '#8e44ad',
          'Transformer': '#d35400', EKF: '#7f8c8d', RCI: '#b9770e', OLD: '#f5b041'}
TCOL = dict(zip(TEMPS, ['#2c3e50', '#2980b9', '#27ae60', '#e67e22', '#c0392b']))
SCEN_EN = {'temiz': 'Clean', 'gürültü düşük': 'Noise (low)', 'gürültü orta': 'Noise (medium)', 'gürültü yüksek': 'Noise (high)',
           'I ofset +0.1 A': 'I offset +0.1 A', 'V ofset +10 mV': 'V offset +10 mV', 'T ofset +2 °C': 'T offset +2 °C',
           'eksik örnek %5': '5 % dropped samples', 'eksik örnek %20': '20 % dropped samples',
           'başlangıç SoC 0.8': 'Start at SoC 0.8', 'başlangıç SoC 0.6': 'Start at SoC 0.6'}
METRICS = ['rmse', 'mae', 'maxe', 'r2', 'jitter', 'bias', 'rmse_ema', 'mae_ema', 'maxe_ema', 'jitter_ema', 'bias_ema']
en = lambda m: EN.get(m, m)

MM = 1 / 25.4
W1, W15, W2 = 88 * MM, 140 * MM, 183 * MM
DPI = 600
FORMATS = ('pdf', 'png')


# ───────────────────────────── yardımcılar ─────────────────────────────
def ci95(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; n = len(x)
    return float(stats.t.ppf(.975, n - 1) * x.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan


def load_table(path):
    path = Path(path)
    if path.exists():
        try: return pd.read_parquet(path)
        except Exception: pass
    c = path.with_suffix('.csv')
    return pd.read_csv(c) if c.exists() else None


def _wilcox(d):
    d = np.asarray(d, float); d = d[~np.isnan(d)]
    return float(stats.wilcoxon(d).pvalue) if len(d) > 1 and np.any(d != 0) else np.nan


def _scope(df):
    return np.where(df.temp == '-20degC', '−20 °C', 'görülen')


def _collect(exp, stage):
    try: return exp.collect(stage) if (exp.p['runs'] / stage).exists() else pd.DataFrame()
    except Exception: return pd.DataFrame()


def _records(exp, stage):
    try: return exp.records(stage) if (exp.p['runs'] / stage).exists() else []
    except Exception: return []


def _sub(df, names):
    return df[df.model.isin(names)] if len(df) else df


# ───────────────────────────── veri toplama ─────────────────────────────
def gather(exp, step_xlsx=None):
    """Tüm aşamaların satırlarını, kayıtlarını ve ek tabloları tek sözlükte toplar."""
    st = {s: _collect(exp, s) for s in ['final', 'final_c', 'ablation_c', 'sensitivity_c', 'ablation', 'ablation_b']}
    rec = {s: _records(exp, s) for s in st}
    X = exp.p['excel']
    ekf = pd.read_csv(X / '04_ekf_satirlar.csv') if (X / '04_ekf_satirlar.csv').exists() else pd.DataFrame()
    if len(ekf) and 'başlangıç' in ekf:
        ekf = ekf[ekf['başlangıç'] == 'doğru SoC0'].drop(columns=[c for c in ['başlangıç', 'ms_adım'] if c in ekf])
    allm = pd.concat([d for d in [st['final_c'], _sub(st['ablation_c'], [CS]), _sub(st['final'], [HYB] + BASE), ekf] if len(d)], ignore_index=True)
    abl = pd.concat([d for d in [st['final_c'], st['ablation_c'], st['sensitivity_c'], _sub(st['ablation_b'], [RCI]),
                                 _sub(st['ablation'], [OHM, NOI]), _sub(st['final'], [HYB, OLD])] if len(d)], ignore_index=True)
    # ablasyon karşılaştırmaları C'nin seed'leriyle eşleştirilir
    recs = {}
    for s, rr in rec.items():
        for r in rr: recs.setdefault((s, r['name']), []).append(r)
    rob = [pd.read_csv(X / f) for f in ['05_saglamlik_satirlar.csv', '05c_saglamlik_satirlar.csv'] if (X / f).exists()]
    rob = pd.concat(rob, ignore_index=True) if rob else pd.DataFrame()
    eff = []
    for f in ['verimlilik.json', 'verimlilik_c.json', 'verimlilik_cs.json']:
        p = exp.p['configs'] / f
        if p.exists(): eff.append(pd.read_json(p))
    eff = pd.concat(eff, ignore_index=True).drop_duplicates('model', keep='last') if eff else pd.DataFrame()
    re_p = exp.p['configs'] / 'verimlilik_yeniden.json'          # 07: bütün modeller tek oturumda yeniden ölçüldüyse onu kullan
    if re_p.exists():
        new = pd.read_json(re_p)
        keep = [c for c in eff.columns if c not in new.columns or c == 'model'] if len(eff) else ['model']
        eff = (eff[keep].merge(new, on='model', how='outer') if len(eff) else new)
    ecm_p = exp.p['configs'] / 'ecm_parametreleri.json'
    step_xlsx = Path(step_xlsx) if step_xlsx else X / '04b_basamak_direnci.xlsx'
    step = pd.read_excel(step_xlsx, sheet_name='Özet') if step_xlsx.exists() else None
    aud = pd.read_csv(X / '05_sizinti_pencereler.csv') if (X / '05_sizinti_pencereler.csv').exists() else None
    return {'exp': exp, 'stage': st, 'recs': recs, 'allm': allm, 'abl': abl, 'ekf': ekf, 'rob': rob, 'eff': eff,
            'ecm': json.load(open(ecm_p)) if ecm_p.exists() else None, 'step': step, 'aud': aud, 'norm': exp.norm.stats}


def n_params(D):
    out = {}
    for (s, name), rr in D['recs'].items():
        if rr and 'n_params' in rr[0]: out.setdefault(name, rr[0]['n_params'])
    return out


def series(D, stage, name, seed=None):
    """Zaman serisi (seed verilmezse en düşük val seed). Yoksa None."""
    exp = D['exp']; rr = D['recs'].get((stage, name), [])
    if not rr: return None
    if seed is None:
        rr = sorted(rr, key=lambda r: r.get('val_rmse_final', r.get('train', {}).get('best_val', 9)))
        for r in rr:
            s = load_table(exp.paths(stage, name, r['seed'])[2])
            if s is not None: return s.assign(seed=r['seed'])
        return None
    s = load_table(exp.paths(stage, name, seed)[2])
    return None if s is None else s.assign(seed=seed)


def all_series(D, stage, name):
    rr = D['recs'].get((stage, name), [])
    ss = [series(D, stage, name, r['seed']) for r in rr]
    ss = [s for s in ss if s is not None]
    return pd.concat(ss, ignore_index=True) if ss else None


def ekf_series(D):
    return load_table(D['exp'].p['series'] / 'ekf' / 'ekf_dogru_soc0.parquet')


# ───────────────────────────── tablolar ─────────────────────────────
def seed_level(df, extra=()):
    cols = [c for c in METRICS if c in df]
    sl = df.groupby(['model', 'seed', *extra])[cols].mean().reset_index()
    for c in cols:
        if not c.startswith('r2'): sl[c] *= 100
    return sl


def main_table(allm, order=MAIN, params=None):
    out = []
    for k, f in [('Eğitim sıcaklıkları (25/10/0/−10 °C)', allm.temp != '-20degC'), ('Görülmemiş −20 °C', allm.temp == '-20degC'), ('Tümü', allm.temp == allm.temp)]:
        sl = seed_level(allm[f]); cols = [c for c in METRICS if c in sl]
        g = sl.groupby('model')[cols]
        t = g.mean().add_suffix('_ort').join(g.std().add_suffix('_std')).join(g.agg(ci95).add_suffix('_GA95'))
        t = t[[f'{c}_{s}' for c in cols for s in ('ort', 'std', 'GA95')]]
        t.insert(0, 'n_seed', sl.groupby('model').seed.nunique()); t.insert(0, 'kapsam', k)
        out.append(t.reset_index())
    t = pd.concat(out, ignore_index=True)
    if params: t.insert(2, 'parametre', t.model.map(params))
    t['model'] = pd.Categorical(t.model, [m for m in order if m in set(t.model)] + [m for m in t.model.unique() if m not in order], ordered=True)
    return t.sort_values(['kapsam', 'model']).reset_index(drop=True)


def compact_table(allm, order=MAIN, params=None):
    """Makale Tablo 2 biçimi: görülen / −20 RMSE ort ± std, MAE, maks., sapma, titreme."""
    rows = []
    for m in [x for x in order if x in set(allm.model)]:
        d = allm[allm.model == m]; r = {'model': m, 'Model (EN)': en(m), 'parametre': (params or {}).get(m, np.nan), 'n_seed': d.seed.nunique()}
        for k, f in [('görülen', d.temp != '-20degC'), ('−20', d.temp == '-20degC'), ('tümü', d.temp == d.temp)]:
            sl = seed_level(d[f])
            r[f'RMSE {k}'] = sl.rmse.mean(); r[f'RMSE {k} std'] = sl.rmse.std()
            if k != 'tümü':
                r[f'MAE {k}'] = sl.mae.mean(); r[f'MaxE {k}'] = sl.maxe.mean(); r[f'sapma {k}'] = sl.bias.mean() if 'bias' in sl else np.nan
        r['titreme (EMA)'] = seed_level(d).jitter_ema.mean() if 'jitter_ema' in d else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def by_key(allm, key, order=MAIN):
    sl = seed_level(allm, extra=(key,))
    g = sl.groupby(['model', key]).rmse
    t = pd.DataFrame({'RMSE_ort': g.mean(), 'RMSE_std': g.std(), 'RMSE_GA95': g.agg(ci95)}).reset_index()
    t['model'] = pd.Categorical(t.model, [m for m in order if m in set(t.model)], ordered=True)
    return t.sort_values(['model', key]).reset_index(drop=True)


def per_file(allm, order=MAIN):
    g = (allm.groupby(['temp', 'cycle', 'model']).rmse.agg(['mean', 'std']) * 100).reset_index()
    piv = g.pivot_table(index=['temp', 'cycle'], columns='model', values='mean')
    piv = piv[[m for m in order if m in piv.columns]]; piv['kazanan'] = piv.idxmin(1)
    return g, piv


def paired(df, ref, other):
    """ref vs other, iki kapsamda: seed düzeyi Welch + hücre düzeyi eşleştirilmiş Wilcoxon.
    Hücre = (seed, temp, cycle); deterministik model (tek seed, ör. EKF) için ref seed ortalaması alınır."""
    rows = []
    for k, f in [('−20 °C', df.temp == '-20degC'), ('görülen', df.temp != '-20degC')]:
        m = df[f]
        a = m[m.model == ref]; b = m[m.model == other]
        if not len(a) or not len(b): continue
        ca = a.groupby(['seed', 'temp', 'cycle']).rmse.mean() * 100; cb = b.groupby(['seed', 'temp', 'cycle']).rmse.mean() * 100
        if b.seed.nunique() == 1 and a.seed.nunique() > 1:
            ca = ca.groupby(['temp', 'cycle']).mean(); cb = cb.droplevel('seed')
        d = (ca - cb.reindex(ca.index)).dropna()
        sa = a.groupby('seed').rmse.mean() * 100; sb = b.groupby('seed').rmse.mean() * 100
        welch = stats.ttest_ind(sa, sb, equal_var=False).pvalue if len(sa) > 1 and len(sb) > 1 else np.nan
        rows.append({'karşılaştırma': f'{en(ref)} vs {en(other)}', 'model': other, 'kapsam': k,
                     'ref RMSE': sa.mean(), 'model RMSE': sb.mean(), 'iyileşme (puan)': sb.mean() - sa.mean(),
                     'ref daha iyi': f'{(d < 0).sum()}/{len(d)}', 'Wilcoxon p (eşleştirilmiş)': _wilcox(d), 'Welch p (seed)': welch})
    return rows


def significance(allm, ref=C, order=MAIN):
    return pd.DataFrame([r for m in order if m != ref and m in set(allm.model) for r in paired(allm, ref, m)])


TESTS = {RCONST: 'Kazanç R(T)\'den mi? (mekanizma)', CS: 'Kazanç model boyutundan mı? (h=64, Hibrit ile eşit)',
         RCI: 'Çok ölçekli RC kolları katkı veriyor mu?', OHM: 'Omurgaya akım vermek gerekli mi?',
         OLD: 'Akımsız omurga + RC (eski önerilen)', HYB: 'Telafi kanalının etkisi (h=64)', NOI: 'Telafi de akım da yok',
         'C duyarlılık: ohmik yumuşatma yok': 'Duyarlılık', 'C duyarlılık: ohmik yumuşatma 5 s': 'Duyarlılık', 'C duyarlılık: W = 128': 'Duyarlılık'}


def ablation_table(abl, params=None):
    rows = []
    for m in [x for x in ABL_ORDER if x in set(abl.model)]:
        d = abl[abl.model == m]; sl = seed_level(d.assign(k=_scope(d)), extra=('k',))
        r = {'varyant': m, 'Variant (EN)': ABL_EN.get(m, m), 'neyi test eder': TESTS.get(m, ''), 'parametre': (params or {}).get(m, np.nan),
             'n_seed': d.seed.nunique(), 'val (%)': d.groupby('seed').val_rmse.first().mean() * 100}
        for k in ['görülen', '−20 °C']:
            s = sl[sl.k == k]
            r[f'RMSE {k}'] = s.rmse.mean(); r[f'RMSE {k} std'] = s.rmse.std(); r[f'sapma {k}'] = s.bias.mean() if 'bias' in s else np.nan
        if m != C:
            for p in paired(abl[abl.seed.isin(abl[abl.model == m].seed.unique())], C, m):
                tag = 'gör' if p['kapsam'] == 'görülen' else '−20'
                r[f'C daha iyi ({tag})'] = p['ref daha iyi']; r[f'p Wilcoxon ({tag})'] = p['Wilcoxon p (eşleştirilmiş)']
        rows.append(r)
    return pd.DataFrame(rows)


def factorial_table(abl):
    """2 × 3 faktöriyel tablo: omurgaya akım (yok/var) × fizik dalı (yok / yalnız ohmik / ohmik + 3 RC)."""
    cell = {('akımsız', 'yok'): NOI, ('akımsız', 'yalnız ohmik'): OHM, ('akımsız', 'ohmik + 3 RC'): OLD,
            ('akımlı', 'yok'): HYB, ('akımlı', 'yalnız ohmik'): C, ('akımlı', 'ohmik + 3 RC'): RCI}
    rows = []
    for (i, j), m in cell.items():
        d = abl[abl.model == m]
        if not len(d): continue
        sl = seed_level(d.assign(k=_scope(d)), extra=('k',))
        rows.append({'omurgaya akım': i, 'fizik dalı': j, 'model': m, 'n_seed': d.seed.nunique(),
                     'görülen': sl[sl.k == 'görülen'].rmse.mean(), '−20 °C': sl[sl.k == '−20 °C'].rmse.mean(),
                     '−20 sapma': sl[sl.k == '−20 °C'].bias.mean() if 'bias' in sl else np.nan,
                     'val (%)': d.groupby('seed').val_rmse.first().mean() * 100})
    t = pd.DataFrame(rows)
    txt = t.assign(v=lambda x: x['görülen'].map('{:.3f}'.format) + ' / ' + x['−20 °C'].map('{:.2f}'.format) + ' · val ' + x['val (%)'].map('{:.3f}'.format))
    return t, txt.pivot(index='omurgaya akım', columns='fizik dalı', values='v').reindex(columns=['yok', 'yalnız ohmik', 'ohmik + 3 RC'])


def test_time_ablation(D):
    d = pd.concat([x for x in [D['stage']['final_c'], _sub(D['stage']['ablation_b'], [RCI])] if len(x)], ignore_index=True)
    dc = [c for c in d if c.startswith('drmse_')]
    if not dc: return None
    return (d.groupby(['model', 'temp'])[dc].mean() * 100).reset_index()


def _tn(norm, T):
    return (np.asarray(T, float) - norm['temp']['min']) / (norm['temp']['max'] - norm['temp']['min'] + 1e-8)


def learned_R(rec, norm, T, horizon=None):
    """Öğrenilen direnç (mΩ) T'de. horizon=None → R0 ve ΣRk ayrı; horizon (s) → birim basamak yanıtı."""
    p = rec.get('rc_params') or {}
    if not p: return None
    tn = _tn(norm, T)
    r0 = np.exp(p['r0_a'] + p['r0_b'] * (tn - 1)) * 1e3
    rk = [np.exp(a + b * (tn - 1)) * 1e3 for a, b in zip(p.get('rk_a', []), p.get('rk_b', []))]
    if horizon is None:
        return r0, (sum(rk) if rk else np.zeros_like(r0))
    s = max(int(rec.get('spec', {}).get('rc', {}).get('ohm_smooth', 3)), 1)
    out = r0 * min(horizon, s) / s
    for r, tau in zip(rk, p.get('tau', [])): out = out + r * (1 - np.exp(-horizon / tau))
    return out


def rc_param_table(D, T=(25, 10, 0, -10, -20)):
    rows = []
    for (s, name), rr in D['recs'].items():
        if name not in (C, CS, RCONST, RCI, OLD, OHM): continue
        for rec in rr:
            if not rec.get('rc_params'): continue
            r = {'aşama': s, 'model': name, 'seed': rec['seed']}
            for t in T:
                r0, rk = learned_R(rec, D['norm'], t); r[f'R0 {t}°C (mΩ)'] = float(r0); r[f'ΣRk {t}°C (mΩ)'] = float(rk)
            for j, tau in enumerate(rec['rc_params'].get('tau', [])): r[f'τ{j + 1} (s)'] = tau
            rows.append(r)
    t = pd.DataFrame(rows)
    ecm = D['ecm']
    if ecm and 'R_fit' in ecm:
        er = {'aşama': '04', 'model': '2RC-ECM (tanımlanan)', 'seed': np.nan}
        for tt in T:
            er[f'R0 {tt}°C (mΩ)'] = np.exp(ecm['R_fit']['R0'][0] + ecm['R_fit']['R0'][1] * tt) * 1e3
            er[f'ΣRk {tt}°C (mΩ)'] = sum(np.exp(ecm['R_fit'][k][0] + ecm['R_fit'][k][1] * tt) for k in ('R1', 'R2')) * 1e3
        t = pd.concat([t, pd.DataFrame([er])], ignore_index=True)
    return t


def step_compare(D):
    """04b basamak direnci (veri, ölçülen hücre sıcaklığında) ↔ öğrenilen direnç (aynı sıcaklık ve ufuk)."""
    st = D['step']
    if st is None: return None, None
    st = st.assign(ufuk=st.ufuk.astype(str), etiket=st.etiket.astype(str))
    Tc = st.drop_duplicates('etiket').set_index('etiket')['T_batt medyan (°C)']
    rows = []
    for lab, T in Tc.items():
        r = {'etiket': lab, 'hücre T medyan (°C)': T}
        for h in ['1s', '3s']:
            g = st[(st.etiket == lab) & (st.ufuk == h)]
            if len(g): r.update({f'veri {h} (mΩ)': g['medyan (mΩ)'].iloc[0], f'veri {h} CI alt': g['CI95 alt'].iloc[0],
                                 f'veri {h} CI üst': g['CI95 üst'].iloc[0], f'n {h}': g['n'].iloc[0]})
        for name, stage, key in [(C, 'final_c', 'C'), (CS, 'ablation_c', 'C-S'), (RCI, 'ablation_b', 'RC uzantısı')]:
            vals = [learned_R(rec, D['norm'], T, horizon=3) for rec in D['recs'].get((stage, name), [])]
            vals = [float(v) for v in vals if v is not None]
            if vals:
                r[f'{key} öğrenilen 3 s (mΩ)'] = np.mean(vals); r[f'{key} sd'] = np.std(vals, ddof=1) if len(vals) > 1 else np.nan
                if 'veri 3s (mΩ)' in r: r[f'{key} / veri 3 s'] = r[f'{key} öğrenilen 3 s (mΩ)'] / r['veri 3s (mΩ)']
        rows.append(r)
    t = pd.DataFrame(rows).sort_values('hücre T medyan (°C)', ascending=False).reset_index(drop=True)
    # sıcaklık duyarlılığı (ln R'nin °C başına eğimi) ve 25 → −20 artış oranı
    sl = []
    warm, cold = t.iloc[0], t.iloc[-1]
    for key, col in [('veri 1 s', 'veri 1s (mΩ)'), ('veri 3 s', 'veri 3s (mΩ)'), ('C öğrenilen', 'C öğrenilen 3 s (mΩ)'),
                     ('C-S öğrenilen', 'C-S öğrenilen 3 s (mΩ)'), ('RC uzantısı', 'RC uzantısı öğrenilen 3 s (mΩ)')]:
        if col not in t: continue
        x, y = t['hücre T medyan (°C)'].values, np.log(t[col].values)
        tr = t.index[:-1]                       # eğitim aralığı (−20 hariç)
        b_all = np.polyfit(x, y, 1)[0]; b_tr = np.polyfit(x[tr], y[tr], 1)[0]
        sl.append({'kaynak': key, 'eğim tümü (%/°C)': -100 * b_all, 'eğim eğitim aralığı (%/°C)': -100 * b_tr,
                   f'oran {cold.etiket} / {warm.etiket}': cold[col] / warm[col]})
    return t, pd.DataFrame(sl)


def soc_bias(D, temps=('-10degC', '-20degC'), bins=np.arange(0, 101, 5)):
    """Sıcaklık başına, referans SoC aralığına göre ortalama hata (tüm seed'ler) — soğuk sapma mekanizması."""
    src = [(C, 'final_c'), (CS, 'ablation_c'), (HYB, 'final')] + [(b, 'final') for b in BASE]
    out = []
    for name, stage in src:
        s = all_series(D, stage, name)
        if s is None: continue
        out.append(s[s.temp.isin(temps)][['temp', 'seed', 'soc_true', 'err']].assign(model=name))
    e = ekf_series(D)
    if e is not None: out.append(e[e.temp.isin(temps)][['temp', 'soc_true', 'err']].assign(model=EKF, seed=0))
    if not out: return None
    s = pd.concat(out, ignore_index=True)
    s['SoC aralığı'] = pd.cut(s.soc_true * 100, bins, include_lowest=True)
    s['mid'] = s['SoC aralığı'].apply(lambda iv: iv.mid).astype(float)
    t = (s.groupby(['model', 'temp', 'seed', 'mid'], observed=True).err.mean() * 100).groupby(['model', 'temp', 'mid']).agg(['mean', 'std']).reset_index()
    return t


def robustness(D):
    rob = D['rob']
    if not len(rob): return None, None, None
    rob = rob.assign(k=_scope(rob))
    sl = rob.groupby(['senaryo', 'model', 'k', 'seed']).rmse.mean() * 100
    order = [s for s in SCEN_EN if s in set(rob.senaryo)]
    tab = sl.groupby(['senaryo', 'model', 'k']).mean().unstack(['model', 'k']).reindex(order)
    tab = tab[[c for m in [C, RCI, OLD, HYB] + BASE + [EKF] for c in tab.columns if c[0] == m]]
    clean = sl.xs('temiz', level='senaryo')
    inc = (sl - sl.index.droplevel('senaryo').map(clean)).groupby(['senaryo', 'model', 'k']).mean().unstack(['model', 'k']).reindex(order)
    inc = inc[[c for c in tab.columns if c in inc.columns]]
    sig = []
    for scen in order:
        r = rob[rob.senaryo == scen]
        for m in [x for x in [RCI, OLD, HYB] + BASE + [EKF] if x in set(r.model)]:
            for p in paired(r, C, m): sig.append({'senaryo': scen, **p})
    return tab, inc, pd.DataFrame(sig)


def leakage(D):
    aud = D['aud']; s = series(D, 'final_c', C)
    if aud is None or s is None: return None, None
    aud = aud.copy(); aud['C_rmse_60s'] = np.nan
    for (t, c), g in aud.groupby(['temp', 'cycle']):
        e = s[(s.temp == t) & (s.cycle == c)].err.values
        aud.loc[g.index, 'C_rmse_60s'] = [float(np.sqrt(np.mean(e[a:a + 60] ** 2))) if len(e[a:a + 60]) else np.nan for a in g.start.values]
    rho = aud.groupby('temp').apply(lambda g: stats.spearmanr(g.max_corr, g.C_rmse_60s, nan_policy='omit').correlation).rename('spearman')
    return aud, rho


def efficiency(D, allm):
    e = D['eff']
    if not len(e): return None
    k = seed_level(allm.assign(k=_scope(allm)), extra=('k',)).groupby(['model', 'k']).rmse.mean().unstack('k')
    e = e[e.model.isin(MAIN)].merge(k.reset_index(), on='model', how='left')
    e['model'] = pd.Categorical(e.model, [m for m in MAIN if m in set(e.model)], ordered=True)
    return e.sort_values('model').reset_index(drop=True)


def learning_curves(D):
    out = []
    for (s, name), rr in D['recs'].items():
        if name not in MAIN: continue
        for r in rr:
            h = r.get('train', {}).get('hist')
            if h: out.append(pd.DataFrame(h).assign(model=name, seed=r['seed']))
    return pd.concat(out, ignore_index=True) if out else None


def tables(D):
    P = n_params(D)
    allm, abl = D['allm'], D['abl']
    T = {}
    T['T1 Ana tablo (makale)'] = compact_table(allm, params=P)
    T['T1b Ana sonuçlar (tam)'] = main_table(allm, params=P)
    T['T2 Sıcaklık'] = by_key(allm, 'temp'); T['T3 Çevrim'] = by_key(allm, 'cycle')
    g, piv = per_file(allm); T['T4 Dosya (ort)'] = piv.reset_index(); T['T4b Dosya ort-std'] = g
    T['T5 Anlamlılık (C vs)'] = significance(allm)
    fac, fac_txt = factorial_table(abl); T['T6 Faktöriyel (2x3)'] = fac; T['T6b Faktöriyel metin'] = fac_txt.reset_index()
    T['T7 Ablasyon + duyarlılık'] = ablation_table(abl, P)
    T['T8 Test anı ablasyon'] = test_time_ablation(D)
    T['T9 Öğrenilen R ve τ'] = rc_param_table(D)
    sc, slope = step_compare(D); T['T10 Basamak direnci vs öğrenilen'] = sc; T['T10b Sıcaklık duyarlılığı'] = slope
    tab, inc, sig = robustness(D)
    T['T11 Sağlamlık RMSE'] = None if tab is None else tab.round(4).pipe(lambda x: x.set_axis([f'{a} | {b}' for a, b in x.columns], axis=1)).reset_index()
    T['T11b Sağlamlık artış'] = None if inc is None else inc.round(4).pipe(lambda x: x.set_axis([f'{a} | {b}' for a, b in x.columns], axis=1)).reset_index()
    T['T11c Sağlamlık anlamlılık'] = sig
    aud, rho = leakage(D); T['T12 Sızıntı-hata (C)'] = None if rho is None else rho.reset_index()
    T['T13 Verimlilik'] = efficiency(D, allm)
    T['T14 Soğuk sapma vs SoC'] = soc_bias(D)
    T['T15 Öğrenme eğrileri'] = learning_curves(D)
    T['Ham ana satırlar'] = allm; T['Ham ablasyon satırları'] = abl
    D['_aud'] = aud
    return T


def write_excel(T, path):
    with pd.ExcelWriter(path) as xw:
        for k, v in T.items():
            if v is not None and len(v): v.to_excel(xw, sheet_name=k[:31], index=False)
    return path


# ───────────────────────────── şekiller ─────────────────────────────
def style():
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'], 'mathtext.fontset': 'stixsans',
                         'font.size': 7.5, 'axes.titlesize': 8, 'axes.labelsize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'axes.linewidth': .6, 'lines.linewidth': 1.0, 'grid.linewidth': .4, 'xtick.major.width': .6, 'ytick.major.width': .6,
                         'axes.grid': True, 'grid.alpha': .3, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': DPI,
                         'figure.dpi': 120, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'legend.frameon': False})


def save(fig, gdir, name):
    gdir = Path(gdir); gdir.mkdir(parents=True, exist_ok=True)
    for f in FORMATS:
        fig.savefig(gdir / f'{name}.{f}', dpi=DPI, bbox_inches='tight', pad_inches=.02)
    return fig


def letter(ax, s):
    ax.set_title(s, loc='left', fontweight='bold', fontsize=8)


def plain_log(ax, axis='y'):
    a = ax.yaxis if axis == 'y' else ax.xaxis
    f = ScalarFormatter(); f.set_scientific(False); a.set_major_formatter(f); a.set_minor_formatter(NullFormatter())


def _models(df, order=MAIN):
    return [m for m in order if m in set(df.model)]


def fig_rmse_temperature(D, gdir, name='C_Fig01_rmse_by_temperature'):
    TT = by_key(D['allm'], 'temp'); models = _models(TT)
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 58 * MM), gridspec_kw={'width_ratios': [4, 1.6], 'wspace': .18})
    w = .84 / len(models)
    for ax, temps in [(a, TEMPS[:4]), (b, TEMPS[4:])]:
        xs = np.arange(len(temps))
        for i, m in enumerate(models):
            d = TT[TT.model == m].set_index('temp').reindex(temps)
            ax.bar(xs + (i - (len(models) - 1) / 2) * w, d.RMSE_ort, w, yerr=d.RMSE_GA95, color=COLORS.get(m), label=en(m),
                   error_kw={'elinewidth': .6, 'capsize': 1.2}, linewidth=0, hatch='////' if m == CS else None, edgecolor='white')
        ax.set_xticks(xs); ax.set_xticklabels([TLAB[t] for t in temps]); ax.set_ylim(bottom=0)
    a.set_ylabel('RMSE (SoC %)'); a.set_title('Training temperatures'); b.set_title('Unseen temperature')
    letter(a, '(a)'); letter(b, '(b)')
    h, l = a.get_legend_handles_labels(); fig.legend(h, l, ncol=4, loc='upper center', bbox_to_anchor=(.5, 1.13))
    return save(fig, gdir, name)


def fig_cold_cycles(D, gdir, ycap=20, name='C_Fig02_rmse_minus20_by_cycle'):
    allm = D['allm']; models = _models(allm)
    sl = (allm[allm.temp == '-20degC'].groupby(['model', 'seed', 'cycle']).rmse.mean() * 100).groupby(['model', 'cycle']).agg(['mean', ci95]).reset_index()
    fig, ax = plt.subplots(figsize=(W2, 55 * MM)); xs = np.arange(4); w = .84 / len(models)
    for i, m in enumerate(models):
        d = sl[sl.model == m].set_index('cycle').reindex(CYC); x = xs + (i - (len(models) - 1) / 2) * w
        mu = d['mean'].clip(upper=ycap); ci = d['ci95'].fillna(0)
        ax.bar(x, mu, w, color=COLORS.get(m), label=en(m), linewidth=0, hatch='////' if m == CS else None, edgecolor='white')
        ax.errorbar(x, mu, yerr=[ci.where(d['mean'] < ycap, 0), np.minimum(ci, (ycap - d['mean']).clip(lower=0))], fmt='none', ecolor='k', elinewidth=.6, capsize=1.2)
        for xi, v in zip(x, d['mean']):
            if np.isfinite(v) and v > ycap: ax.annotate(f'{v:.0f}', (xi, ycap), xytext=(0, 1), textcoords='offset points', ha='center', fontsize=5.5)
    ax.set_xticks(xs); ax.set_xticklabels(CYC); ax.set_ylabel('RMSE at −20 °C (SoC %)'); ax.set_ylim(0, ycap * 1.06)
    ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(.5, 1.27))
    return save(fig, gdir, name)


def fig_tracking(D, gdir, cases=(('25degC', 'US06'), ('0degC', 'LA92'), ('-10degC', 'HWFET'), ('-20degC', 'UDDS')),
                 name='C_Fig03_soc_tracking'):
    S = {C: series(D, 'final_c', C), HYB: series(D, 'final', HYB)}
    for b in BASE: S[b] = series(D, 'final', b)
    S[EKF] = ekf_series(D); S = {k: v for k, v in S.items() if v is not None}
    emph = (C, HYB, EKF)
    fig, axs = plt.subplots(2, len(cases), figsize=(W2, 95 * MM), sharex='col', gridspec_kw={'height_ratios': [1.7, 1], 'hspace': .12, 'wspace': .32})
    for j, (t, c) in enumerate(cases):
        o = S[C]; o = o[(o.temp == t) & (o.cycle == c)]
        if not len(o): continue
        t0 = o.t_s.min()
        axs[0, j].plot((o.t_s - t0) / 3600, o.soc_true * 100, 'k', lw=1.1, label='Reference')
        if HYB in S:
            h = S[HYB]; h = h[(h.temp == t) & (h.cycle == c)]
            axs[0, j].plot((h.t_s - t0) / 3600, h.soc_pred_ema * 100, color=COLORS[HYB], lw=.6, ls='--', label=en(HYB))
        axs[0, j].plot((o.t_s - t0) / 3600, o.soc_pred_ema * 100, color=COLORS[C], lw=.7, label=en(C))
        axs[0, j].set_title(f'{TLAB[t]} · {c}' + (' (unseen)' if t == '-20degC' else ''))
        for m, s in sorted(S.items(), key=lambda kv: kv[0] in emph):
            s = s[(s.temp == t) & (s.cycle == c)]
            if not len(s): continue
            e = ((s.soc_pred_ema if 'soc_pred_ema' in s else s.soc_pred) - s.soc_true) * 100
            strong = m in emph
            axs[1, j].plot((s.t_s - t0) / 3600, e, color=COLORS.get(m), lw=.7 if strong else .4, alpha=1 if strong else .35, label=en(m), zorder=3 if m == C else 2)
        axs[1, j].axhline(0, color='k', lw=.4); axs[1, j].set_xlabel('Time (h)')
    axs[0, 0].set_ylabel('SoC (%)'); axs[1, 0].set_ylabel('Error (SoC %)')
    h0, l0 = axs[0, 0].get_legend_handles_labels(); h1, l1 = axs[1, -1].get_legend_handles_labels()
    hl = dict(zip(l0, h0)); hl.update({l: h for h, l in zip(h1, l1) if l not in hl})
    fig.legend(list(hl.values()), list(hl.keys()), ncol=4, loc='upper center', bbox_to_anchor=(.5, .02), fontsize=6.5)
    return save(fig, gdir, name)


def fig_voltage_collapse(D, gdir, step=4, name='C_Fig04_voltage_vs_compensated'):
    o = series(D, 'final_c', C)
    fig, axs = plt.subplots(1, 2, figsize=(W2, 68 * MM), sharey=True, gridspec_kw={'wspace': .08})
    for t in TEMPS:
        g = o[o.temp == t].iloc[::step]
        axs[0].scatter(g.soc_true * 100, g.V, s=.4, color=TCOL[t], alpha=.3, rasterized=True, label=TLAB[t] + (' (unseen)' if t == '-20degC' else ''))
        axs[1].scatter(g.soc_true * 100, g.ocv_hat, s=.4, color=TCOL[t], alpha=.3, rasterized=True)
    axs[0].set_title('Measured terminal voltage $V$'); axs[1].set_title('IR-compensated voltage $V-\\hat{R}_0(T)\\,\\bar{I}$')
    axs[0].set_ylabel('Voltage (V)')
    for a, l in zip(axs, ['(a)', '(b)']): a.set_xlabel('Reference SoC (%)'); a.set_ylim(2.8, 4.4); letter(a, l)
    axs[0].legend(markerscale=12, loc='upper left', fontsize=6.5)
    return save(fig, gdir, name)


def fig_ablation(D, gdir, name='C_Fig05_ablation'):
    abl = D['abl']; rows = []
    for m in [x for x in ABL_ORDER if x in set(abl.model)]:
        d = abl[abl.model == m]; sl = seed_level(d.assign(k=_scope(d)), extra=('k',))
        r = {'model': m, 'grp': 'ref' if m == C else ('sens' if m in SENS else 'abl')}
        for k, kk in [('görülen', 'seen'), ('−20 °C', 'cold')]:
            v = sl[sl.k == k].rmse; r[kk] = v.mean(); r[f'{kk}_ci'] = ci95(v)
        rows.append(r)
    tab = pd.DataFrame(rows)
    y, pos, prev = [], 0.0, None
    for g in tab.grp:
        if prev is not None and g != prev: pos += .6
        y.append(-pos); pos += 1; prev = g
    y = np.array(y)
    col = {'ref': COLORS[C], 'abl': '#5d6d7e', 'sens': '#aab7b8'}
    colors = [COLORS[HYB] if m == HYB else col[g] for m, g in zip(tab.model, tab.grp)]
    fig, axs = plt.subplots(1, 2, figsize=(W2, (22 + 6.2 * len(tab)) * MM), sharey=True, gridspec_kw={'wspace': .08})
    for ax, k, title in [(axs[0], 'seen', 'Training temperatures'), (axs[1], 'cold', 'Unseen −20 °C')]:
        ax.barh(y, tab[k], .7, xerr=tab[f'{k}_ci'].fillna(0), color=colors, error_kw={'elinewidth': .6, 'capsize': 1.2})
        ax.axvline(float(tab.loc[tab.model == C, k].iloc[0]), color=COLORS[C], lw=.6, ls='--')
        right = (tab[k] + tab[f'{k}_ci'].fillna(0))
        for yi, v, rr in zip(y, tab[k], right): ax.text(rr, yi, f'  {v:.2f}', va='center', fontsize=5.8)
        ax.set_xlabel('RMSE (SoC %)'); ax.set_title(title); ax.set_xlim(0, right.max() * 1.15); ax.grid(axis='y', visible=False)
    axs[0].set_yticks(y); axs[0].set_yticklabels([ABL_EN.get(m, m) for m in tab.model])
    letter(axs[0], '(a)'); letter(axs[1], '(b)')
    return save(fig, gdir, name)


def fig_resistance(D, gdir, name='C_Fig06_resistance_vs_temperature'):
    """Öğrenilen R0(T) (C, 5 seed) ↔ veriden ölçülen basamak direnci (1 s, 3 s; ölçülen hücre sıcaklığında) ↔ 2RC-ECM."""
    norm = D['norm']; Tg = np.linspace(-16, 28, 80)
    fig, ax = plt.subplots(figsize=(W15 * .75, 66 * MM))
    for name_, stage, lab, ls in [(C, 'final_c', 'Learned $R_0(T)$, proposed (5 seeds)', '-'), (CS, 'ablation_c', 'Learned $R_0(T)$, Proposed-S', ':')]:
        cu = [learned_R(r, norm, Tg)[0] for r in D['recs'].get((stage, name_), []) if r.get('rc_params')]
        if cu:
            cu = np.array(cu); ax.plot(Tg, cu.mean(0), color=COLORS[name_], ls=ls, label=lab)
            if name_ == C: ax.fill_between(Tg, cu.min(0), cu.max(0), color=COLORS[C], alpha=.22, lw=0)
    rci = [learned_R(r, norm, Tg, horizon=3) for r in D['recs'].get(('ablation_b', RCI), []) if r.get('rc_params')]
    if rci: ax.plot(Tg, np.mean(rci, 0), color=COLORS[RCI], ls='--', lw=.8, label='Multi-scale RC extension, 3-s response')
    sc, _ = step_compare(D)
    if sc is not None:
        for h, mk, c in [('1s', 'v', '#5d6d7e'), ('3s', 'o', '#111111')]:
            if f'veri {h} (mΩ)' not in sc: continue
            y = sc[f'veri {h} (mΩ)']; lo = y - sc[f'veri {h} CI alt']; hi = sc[f'veri {h} CI üst'] - y
            ax.errorbar(sc['hücre T medyan (°C)'], y, yerr=[lo.clip(lower=0), hi.clip(lower=0)], fmt=mk, ms=3.2, color=c, mfc='white' if h == '1s' else c,
                        elinewidth=.6, capsize=1.5, lw=0, label=f'Measured step resistance $\\Delta V/\\Delta I$, {h[:-1]} s')
        cold = sc.iloc[-1]
        # etiket boş bölgeye (hücre T ≈ −9…−1 °C, ~34 mΩ) konur ve okla −20 °C oda noktasına bağlanır (veri noktalarıyla çakışmaz)
        ax.annotate('−20 °C chamber (unseen),\ncell ≈ −13 °C', xy=(cold['hücre T medyan (°C)'], cold.get('veri 1s (mΩ)', np.nan) * .9),
                    xytext=(-9.0, 34), textcoords='data', fontsize=5.8, ha='left', va='center', color='#333333',
                    arrowprops={'arrowstyle': '->', 'lw': .6, 'color': '#333333', 'shrinkA': 1, 'shrinkB': 2})
    ecm = D['ecm']
    if ecm and 'per_temp' in ecm and sc is not None:
        Tc = dict(zip(sc.etiket, sc['hücre T medyan (°C)']))
        pts = [(Tc.get(k, p.get('T')), (p['R0'] + p['R1']) * 1e3) for k, p in ecm['per_temp'].items() if 'R0' in p and 'R1' in p]
        if pts: ax.plot(*zip(*pts), 'x', color='#2980b9', ms=4, mew=1, label='Identified 2RC-ECM $R_0+R_1$')
    tmin = norm['temp']['min']
    ax.axvspan(Tg.min(), tmin, color='gray', alpha=.12, lw=0)
    ax.text((Tg.min() + tmin) / 2, .03, 'below\ntraining\nrange', transform=ax.get_xaxis_transform(), fontsize=5.8, va='bottom', ha='center', color='#555555')
    ax.set_yscale('log'); ax.set_ylim(15, 350); ax.set_yticks([20, 30, 50, 100, 200, 300]); plain_log(ax)
    ax.set_xlabel('Measured cell temperature (°C)'); ax.set_ylabel('Resistance (mΩ)'); ax.set_xlim(Tg.min(), Tg.max())
    ax.legend(fontsize=5.6, loc='upper right')
    return save(fig, gdir, name)


def fig_cold_bias(D, gdir, name='C_Fig07_error_vs_soc'):
    t = soc_bias(D)
    if t is None: return None
    temps = [x for x in ['-10degC', '-20degC'] if x in set(t.temp)]
    fig, axs = plt.subplots(1, len(temps), figsize=(W2 * (.5 * len(temps) + .1), 60 * MM), sharey=True, gridspec_kw={'wspace': .08}, squeeze=False)
    for ax, tt, l in zip(axs[0], temps, ['(a)', '(b)']):
        for m in _models(t):
            g = t[(t.model == m) & (t.temp == tt)].sort_values('mid')
            ax.plot(g.mid, g['mean'], color=COLORS.get(m), lw=1.3 if m == C else .7, ls=':' if m == CS else '-', label=en(m), zorder=3 if m == C else 2)
        ax.axhline(0, color='k', lw=.5); ax.set_xlabel('Reference SoC (%)'); ax.set_title(TLAB[tt] + (' (unseen)' if tt == '-20degC' else ' (training)'))
        letter(ax, l); ax.invert_xaxis()
    axs[0, 0].set_ylabel('Mean error (SoC %)')
    h, l = axs[0, 0].get_legend_handles_labels(); fig.legend(h, l, ncol=4, loc='upper center', bbox_to_anchor=(.5, 1.12))
    return save(fig, gdir, name)


def fig_robustness(D, gdir, name='C_Fig08_robustness'):
    rob = D['rob']
    if not len(rob): return None
    rob = rob.assign(k=np.where(rob.temp == '-20degC', 'cold', 'seen'))
    t = (rob.groupby(['k', 'senaryo', 'model', 'seed']).rmse.mean() * 100).groupby(['k', 'senaryo', 'model']).mean()
    order = [s for s in SCEN_EN if s in set(rob.senaryo)]; xs = np.arange(len(order))
    models = [m for m in [C, HYB] + BASE + [EKF] if m in set(rob.model)]
    fig, axs = plt.subplots(1, 2, figsize=(W2, 72 * MM), gridspec_kw={'wspace': .18})
    for ax, k, title in [(axs[0], 'seen', 'Training temperatures'), (axs[1], 'cold', 'Unseen −20 °C')]:
        for m in models:
            v = [t.get((k, s, m), np.nan) for s in order]
            ax.plot(xs, v, marker='o', ms=2.2, lw=1.2 if m == C else .7, color=COLORS.get(m), label=en(m), zorder=3 if m == C else 2)
        ax.set_xticks(xs); ax.set_xticklabels([SCEN_EN[s] for s in order], rotation=40, ha='right'); ax.set_title(title); ax.set_ylim(bottom=0)
    axs[0].set_ylabel('RMSE (SoC %)'); letter(axs[0], '(a)'); letter(axs[1], '(b)')
    h, l = axs[1].get_legend_handles_labels(); fig.legend(h, l, ncol=4, loc='upper center', bbox_to_anchor=(.5, 1.1))
    return save(fig, gdir, name)


def fig_rc_extension(D, gdir, temp='-10degC', cycle='US06', t_from=600, length=300, name='C_Fig09_rc_extension'):
    """Çok ölçekli RC uzantısı: (a) aşırı gerilim ayrıştırması, (b) öğrenilen R0 ve R0+ΣRk vs sıcaklık (C ile)."""
    s = series(D, 'ablation_b', RCI); src = 'RC + current'
    if s is None or 'eta_rc1' not in s: s = series(D, 'final', OLD); src = 'RC (no current)'
    norm = D['norm']; Tg = np.linspace(-16, 28, 80)
    fig = plt.figure(figsize=(W2, 72 * MM)); gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1], height_ratios=[1, 1.6], hspace=.1, wspace=.28)
    a0 = fig.add_subplot(gs[0, 0]); a1 = fig.add_subplot(gs[1, 0], sharex=a0); b = fig.add_subplot(gs[:, 1])
    if s is not None:
        g = s[(s.temp == temp) & (s.cycle == cycle)]; t0 = g.t_s.min()
        g = g[(g.t_s >= t0 + t_from) & (g.t_s < t0 + t_from + length)]; tt = g.t_s - t0
        a0.step(tt, g.I, where='pre', color='#27ae60', lw=.7); a0.set_ylabel('Current (A)'); plt.setp(a0.get_xticklabels(), visible=False)
        base = np.zeros(len(g))
        for col, lab, c in [('eta_ohm', 'Ohmic $R_0(T)\\bar I$', '#95a5a6')] + [(f'eta_rc{k}', f'RC branch {k}', c) for k, c in zip(range(1, 4), ['#5dade2', '#f0b27a', '#c39bd3'])]:
            if col not in g: continue
            v = g[col].values * 1e3; a1.fill_between(tt, base, base + v, color=c, lw=0, label=lab); base = base + v
        a1.plot(tt, g.eta * 1e3, 'k', lw=.7, label='Total $\\hat{\\eta}$')
        a1.set_ylabel('Overpotential (mV)'); a1.set_xlabel('Time (s)'); a1.legend(ncol=3, fontsize=6, loc='upper center', bbox_to_anchor=(.5, -.3))
        a0.set_title(f'(a) {src}, {TLAB[temp]} · {cycle}', loc='left', fontsize=8, fontweight='bold')
    for name_, stage, part, lab, c, ls in [(RCI, 'ablation_b', 0, 'Extension $R_0$', COLORS[RCI], '-'), (RCI, 'ablation_b', 1, 'Extension $R_0+\\Sigma R_k$', COLORS[RCI], '--'),
                                          (C, 'final_c', 0, 'Proposed $R_0$ (single branch)', COLORS[C], '-')]:
        cu = [learned_R(r, norm, Tg) for r in D['recs'].get((stage, name_), []) if r.get('rc_params')]
        if not cu: continue
        cu = np.array([x[0] if part == 0 else x[0] + x[1] for x in cu])
        b.plot(Tg, cu.mean(0), color=c, ls=ls, label=lab); b.fill_between(Tg, cu.min(0), cu.max(0), color=c, alpha=.18, lw=0)
    taus = [r['rc_params']['tau'] for r in D['recs'].get(('ablation_b', RCI), []) if r.get('rc_params')]
    if taus:
        tm = np.mean(taus, 0); b.text(.03, .03, 'Learned τ: ' + ' / '.join(f'{x:.1f}' for x in tm) + ' s', transform=b.transAxes, fontsize=6)
    b.axvspan(Tg.min(), norm['temp']['min'], color='gray', alpha=.12, lw=0)
    b.set_yscale('log'); b.set_yticks([20, 30, 50, 100, 200, 300, 500]); plain_log(b); b.set_xlabel('Cell temperature (°C)'); b.set_ylabel('Resistance (mΩ)')
    b.legend(fontsize=6, loc='upper right'); b.set_title('(b) Learned resistances', loc='left', fontsize=8, fontweight='bold')
    return save(fig, gdir, name)


def fig_leakage(D, gdir, name='C_Fig10_leakage_audit'):
    aud = D.get('_aud')
    if aud is None: return None
    fig, axs = plt.subplots(1, 2, figsize=(W2, 58 * MM), gridspec_kw={'wspace': .25})
    axs[0].hist(aud.max_corr, bins=40, color='#7f8c8d'); axs[0].set_xlabel('Max. current-profile correlation\n(test window vs. training/validation)'); axs[0].set_ylabel('Test windows')
    for t in TEMPS:
        g = aud[aud.temp == t]; axs[1].scatter(g.max_corr, g.C_rmse_60s * 100, s=1.2, color=TCOL[t], alpha=.45, rasterized=True, label=TLAB[t])
    axs[1].set_xlabel('Max. current-profile correlation'); axs[1].set_ylabel('Proposed model RMSE, 60 s (SoC %)'); axs[1].legend(markerscale=5, fontsize=6.5)
    letter(axs[0], '(a)'); letter(axs[1], '(b)')
    return save(fig, gdir, name)


def fig_learning(D, gdir, name='C_Fig11_learning_curves'):
    fig, ax = plt.subplots(figsize=(W1, 62 * MM))
    for i, r in enumerate(D['recs'].get(('final_c', C), [])):
        h = pd.DataFrame(r.get('train', {}).get('hist', []))
        if not len(h): continue
        ax.plot(h.epoch, h.val_rmse * 100, color=COLORS[C], lw=.6, alpha=.75, label='Validation (NN)' if i == 0 else None)
        if 'test_rmse_diag' in h:
            d = h.dropna(subset=['test_rmse_diag'])
            ax.plot(d.epoch, d.test_rmse_diag * 100, 'o--', color='#2c3e50', ms=1.8, lw=.4, alpha=.7, label='Test (monitoring only)' if i == 0 else None)
        ax.axvline(r['train']['best_epoch'], color=COLORS[C], lw=.3, alpha=.4)
    ax.set_xlabel('Epoch'); ax.set_ylabel('RMSE (SoC %)'); ax.set_ylim(0, None); ax.legend(fontsize=6.5)
    return save(fig, gdir, name)


def fig_efficiency(D, gdir, name='C_Fig12_accuracy_vs_size'):
    e = efficiency(D, D['allm'])
    if e is None or 'parametre' not in e: return None
    e = e.dropna(subset=['parametre'])
    fig, axs = plt.subplots(1, 2, figsize=(W2 * .8, 60 * MM), sharex=True, gridspec_kw={'wspace': .3})
    for ax, k, title in [(axs[0], 'görülen', 'Training temperatures'), (axs[1], '−20 °C', 'Unseen −20 °C')]:
        for _, r in e.iterrows():
            ax.scatter(r['parametre'], r[k], s=14, color=COLORS.get(r['model']), zorder=3, edgecolor='white', lw=.4)
            ax.annotate(EN_SHORT.get(r['model'], r['model']), (r['parametre'], r[k]), xytext=(3, 2), textcoords='offset points', fontsize=6)
        ax.set_xscale('log'); ax.set_xlabel('Trainable parameters'); ax.set_title(title)
        if k == '−20 °C': ax.set_ylim(bottom=0)
        ax.set_xlim(e.parametre.min() / 1.6, e.parametre.max() * 2.4)
    axs[0].set_ylabel('RMSE (SoC %)'); letter(axs[0], '(a)'); letter(axs[1], '(b)')
    return save(fig, gdir, name)


FIGS = [fig_rmse_temperature, fig_cold_cycles, fig_tracking, fig_voltage_collapse, fig_ablation, fig_resistance,
        fig_cold_bias, fig_robustness, fig_rc_extension, fig_leakage, fig_learning, fig_efficiency]
