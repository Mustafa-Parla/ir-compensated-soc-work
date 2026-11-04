"""Publication figures in English (JESTECH template: 88 mm column / 183 mm full width; ≥7 pt Arial). V26.

Every figure is saved as vector PDF (preferred for submission) and as PNG/TIFF at `DPI` (default 600).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter

MM = 1 / 25.4
W1, W15, W2 = 88 * MM, 140 * MM, 183 * MM          # JESTECH template: column 88 mm, text width 183 mm (A4, margins 12/15 mm)
DPI = 600
FORMATS = ('pdf', 'png')                             # add 'tiff' if the journal asks for TIFF

OURS = 'Önerilen (RC-Hibrit)'
OURS_S = 'Önerilen-küçük (h=64)'
EN = {OURS: 'Proposed (RC-hybrid)', OURS_S: 'Proposed-S (h = 64)', 'Hibrit (GRU→Transformer)': 'Hybrid (GRU→Transformer)',
      'LSTM': 'LSTM', 'GRU': 'GRU', 'CNN-LSTM': 'CNN-LSTM', 'Transformer': 'Transformer', '2RC-ECM + EKF': '2RC-ECM + EKF',
      # ablations / sensitivity
      'Akımsız (noI)': 'No current input, no physics branch', 'RC: R sıcaklıktan bağımsız': 'Temperature-independent R',
      'RC: yalnız ohmik kol': 'Ohmic branch only', 'RC: tek RC kolu': 'Single RC branch', 'RC + serbest CNN detector': 'RC + free CNN detector',
      'RC: ohmik yumuşatma yok': 'No ohmic smoothing', 'Duyarlılık: W = 128': 'Window W = 128 s',
      'Duyarlılık: ohmik yumuşatma 5 s': 'Ohmic smoothing 5 s', 'Duyarlılık: τ başlangıcı 1/10/60 s': 'τ initialization 1/10/60 s',
      'RC + akım girişi': 'Proposed + current input', 'RC: yalnız ohmik + akım girişi': 'Ohmic only + current input'}
SCEN_EN = {'temiz': 'Clean', 'gürültü düşük': 'Noise (low)', 'gürültü orta': 'Noise (medium)', 'gürültü yüksek': 'Noise (high)',
           'I ofset +0.1 A': 'I offset +0.1 A', 'V ofset +10 mV': 'V offset +10 mV', 'T ofset +2 °C': 'T offset +2 °C',
           'eksik örnek %5': '5 % dropped samples', 'eksik örnek %20': '20 % dropped samples',
           'başlangıç SoC 0.8': 'Start at SoC 0.8', 'başlangıç SoC 0.6': 'Start at SoC 0.6'}
COLORS = {OURS: '#c0392b', OURS_S: '#e8836f', 'Hibrit (GRU→Transformer)': '#2c3e50', 'LSTM': '#2980b9', 'GRU': '#16a085',
          'CNN-LSTM': '#8e44ad', 'Transformer': '#d35400', '2RC-ECM + EKF': '#7f8c8d'}
TEMPS = ['25degC', '10degC', '0degC', '-10degC', '-20degC']
TLAB = {'25degC': '25 °C', '10degC': '10 °C', '0degC': '0 °C', '-10degC': '−10 °C', '-20degC': '−20 °C'}
TCOL = dict(zip(TEMPS, ['#2c3e50', '#2980b9', '#27ae60', '#e67e22', '#c0392b']))
CYC = ['US06', 'HWFET', 'UDDS', 'LA92']
en = lambda m: EN.get(m, m)


def style():
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'], 'mathtext.fontset': 'stixsans', 'font.size': 7.5, 'axes.titlesize': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': .6, 'lines.linewidth': 1.0, 'grid.linewidth': .4,
                         'xtick.major.width': .6, 'ytick.major.width': .6, 'axes.grid': True, 'grid.alpha': .3,
                         'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': DPI, 'figure.dpi': 120,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'legend.frameon': False, 'axes.titleweight': 'normal'})


def save(fig, gdir, name):
    gdir = Path(gdir); gdir.mkdir(parents=True, exist_ok=True)
    for f in FORMATS:
        kw = {'pil_kwargs': {'compression': 'tiff_lzw'}} if f == 'tiff' else {}
        fig.savefig(gdir / f'{name}.{f}', dpi=DPI, bbox_inches='tight', pad_inches=.02, **kw)


def plain_log(ax, axis='y'):
    a = ax.yaxis if axis == 'y' else ax.xaxis
    f = ScalarFormatter(); f.set_scientific(False); a.set_major_formatter(f); a.set_minor_formatter(NullFormatter())


def letter(ax, s):
    ax.set_title(s, loc='left', fontweight='bold', fontsize=8)


# ─────────────── Fig. 1 — RMSE by temperature (two panels: seen / unseen) ───────────────
def fig_rmse_temperature(TT, models, gdir, name='Fig01_rmse_by_temperature'):
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 58 * MM), gridspec_kw={'width_ratios': [4, 1.6], 'wspace': .18})
    w = .82 / len(models)
    for ax, temps in [(a, TEMPS[:4]), (b, TEMPS[4:])]:
        xs = np.arange(len(temps))
        for i, m in enumerate(models):
            d = TT[TT.model == m].set_index('temp').reindex(temps)
            ax.bar(xs + (i - (len(models) - 1) / 2) * w, d.RMSE_ort, w, yerr=d.RMSE_GA95, color=COLORS.get(m), label=en(m),
                   error_kw={'elinewidth': .6, 'capsize': 1.2}, linewidth=0, hatch='////' if m == OURS_S else None, edgecolor='white')
        ax.set_xticks(xs); ax.set_xticklabels([TLAB[t] for t in temps]); ax.set_ylim(bottom=0)
    a.set_ylabel('RMSE (SoC %)'); a.set_title('Training temperatures'); b.set_title('Unseen temperature')
    letter(a, '(a)'); letter(b, '(b)')
    h, l = a.get_legend_handles_labels(); fig.legend(h, l, ncol=4, loc='upper center', bbox_to_anchor=(.5, 1.13))
    save(fig, gdir, name); return fig


# ─────────────── Fig. 2 — unseen −20 °C per drive cycle ───────────────
def fig_cold_cycles(allm, models, gdir, ci95, ycap=20, name='Fig02_rmse_minus20_by_cycle'):
    sl = (allm[allm.temp == '-20degC'].groupby(['model', 'seed', 'cycle']).rmse.mean() * 100).groupby(['model', 'cycle']).agg(['mean', ci95]).reset_index()
    fig, ax = plt.subplots(figsize=(W2, 55 * MM)); xs = np.arange(4); w = .82 / len(models)
    for i, m in enumerate(models):
        d = sl[sl.model == m].set_index('cycle').reindex(CYC); x = xs + (i - (len(models) - 1) / 2) * w
        lo = d['mean'] - d['ci95']; hi = np.minimum(d['mean'] + d['ci95'], ycap)
        ax.bar(x, d['mean'].clip(upper=ycap), w, color=COLORS.get(m), label=en(m), linewidth=0, hatch='////' if m == OURS_S else None, edgecolor='white')
        ax.errorbar(x, d['mean'].clip(upper=ycap), yerr=[(d['mean'] - lo).clip(lower=0).where(d['mean'] < ycap, 0), (hi - d['mean']).clip(lower=0)],
                    fmt='none', ecolor='k', elinewidth=.6, capsize=1.2)
        for xi, (mu, up) in zip(x, zip(d['mean'], d['mean'] + d['ci95'])):
            if np.isfinite(up) and up > ycap: ax.annotate('', (xi, ycap), (xi, ycap - 1.2), arrowprops={'arrowstyle': '->', 'lw': .6})
    ax.set_xticks(xs); ax.set_xticklabels(CYC); ax.set_ylabel('RMSE at −20 °C (SoC %)'); ax.set_ylim(0, ycap)
    ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(.5, 1.25))
    save(fig, gdir, name); return fig


# ─────────────── Fig. 3 — SoC tracking and error ───────────────
def fig_tracking(S, gdir, cases=(('25degC', 'US06'), ('0degC', 'LA92'), ('-10degC', 'HWFET'), ('-20degC', 'UDDS')),
                 emph=(OURS, 'Hibrit (GRU→Transformer)', '2RC-ECM + EKF'), name='Fig03_soc_tracking'):
    fig, axs = plt.subplots(2, len(cases), figsize=(W2, 95 * MM), sharex='col', gridspec_kw={'height_ratios': [1.7, 1], 'hspace': .12, 'wspace': .32})
    for j, (t, c) in enumerate(cases):
        o = S[OURS]; o = o[(o.temp == t) & (o.cycle == c)]; t0 = o.t_s.min()
        axs[0, j].plot((o.t_s - t0) / 3600, o.soc_true * 100, 'k', lw=1.1, label='Reference')
        h = S.get('Hibrit (GRU→Transformer)')
        if h is not None:
            h = h[(h.temp == t) & (h.cycle == c)]; axs[0, j].plot((h.t_s - t0) / 3600, h.soc_pred_ema * 100, color=COLORS['Hibrit (GRU→Transformer)'], lw=.6, ls='--', label=en('Hibrit (GRU→Transformer)'))
        axs[0, j].plot((o.t_s - t0) / 3600, o.soc_pred_ema * 100, color=COLORS[OURS], lw=.7, label=en(OURS))
        axs[0, j].set_title(f'{TLAB[t]} · {c}' + (' (unseen)' if t == '-20degC' else ''))
        for m, s in sorted(S.items(), key=lambda kv: kv[0] in emph):
            s = s[(s.temp == t) & (s.cycle == c)]
            if not len(s): continue
            e = ((s.soc_pred_ema if 'soc_pred_ema' in s else s.soc_pred) - s.soc_true) * 100
            strong = m in emph
            axs[1, j].plot((s.t_s - t0) / 3600, e, color=COLORS.get(m), lw=.7 if strong else .4, alpha=1 if strong else .35, label=en(m), zorder=3 if m == OURS else 2)
        axs[1, j].axhline(0, color='k', lw=.4); axs[1, j].set_xlabel('Time (h)')
    axs[0, 0].set_ylabel('SoC (%)'); axs[1, 0].set_ylabel('Error (SoC %)')
    h0, l0 = axs[0, 0].get_legend_handles_labels(); h1, l1 = axs[1, -1].get_legend_handles_labels()
    hl = dict(zip(l0, h0)); hl.update({l: h for h, l in zip(h1, l1) if l not in hl})
    fig.legend(list(hl.values()), list(hl.keys()), ncol=4, loc='upper center', bbox_to_anchor=(.5, .02), fontsize=6.5)
    save(fig, gdir, name); return fig


# ─────────────── Fig. 4 — voltage collapse ───────────────
def fig_voltage_collapse(o, gdir, step=4, name='Fig04_voltage_vs_compensated'):
    fig, axs = plt.subplots(1, 2, figsize=(W2, 68 * MM), sharey=True, gridspec_kw={'wspace': .08})
    for t in TEMPS:
        g = o[o.temp == t].iloc[::step]
        axs[0].scatter(g.soc_true * 100, g.V, s=.4, color=TCOL[t], alpha=.3, rasterized=True, label=TLAB[t] + (' (unseen)' if t == '-20degC' else ''))
        axs[1].scatter(g.soc_true * 100, g.ocv_hat, s=.4, color=TCOL[t], alpha=.3, rasterized=True)
    axs[0].set_title('Measured terminal voltage $V$'); axs[1].set_title('Overpotential-compensated voltage $V-\\hat{\\eta}$')
    axs[0].set_ylabel('Voltage (V)')
    for a, l in zip(axs, ['(a)', '(b)']): a.set_xlabel('Reference SoC (%)'); a.set_ylim(2.8, 4.4); letter(a, l)
    axs[0].legend(markerscale=12, loc='upper left', fontsize=6.5)
    save(fig, gdir, name); return fig


# ─────────────── Fig. 5 — overpotential decomposition (stacked) ───────────────
def fig_eta(o, gdir, temp='-10degC', cycle='US06', t_from=600, length=300, name='Fig05_overpotential_decomposition'):
    g = o[(o.temp == temp) & (o.cycle == cycle)]; t0 = g.t_s.min()
    g = g[(g.t_s >= t0 + t_from) & (g.t_s < t0 + t_from + length)]; t = g.t_s - t0
    fig, axs = plt.subplots(2, 1, figsize=(W15, 80 * MM), sharex=True, gridspec_kw={'height_ratios': [1, 1.6], 'hspace': .1})
    axs[0].step(t, g.I, where='pre', color='#27ae60', lw=.7); axs[0].set_ylabel('Current (A)')
    parts = [('eta_ohm', 'Ohmic $R_0(T)\\bar I$', '#95a5a6')] + [(f'eta_rc{k}', f'RC branch {k}', c) for k, c in zip(range(1, 4), ['#5dade2', '#f0b27a', '#c39bd3'])]
    base = np.zeros(len(g))
    for col, lab, c in parts:
        if col not in g: continue
        v = g[col].values * 1e3; axs[1].fill_between(t, base, base + v, step=None, color=c, lw=0, label=lab); base = base + v
    axs[1].plot(t, g.eta * 1e3, 'k', lw=.7, label='Total $\\hat{\\eta}$')
    axs[1].set_ylabel('Overpotential (mV)'); axs[1].set_xlabel('Time (s)')
    axs[1].legend(ncol=5, loc='upper center', bbox_to_anchor=(.5, -.28), fontsize=6.5)
    axs[0].set_title(f'{TLAB[temp]} · {cycle}')
    save(fig, gdir, name); return fig


# ─────────────── Fig. 6 — learned resistances vs identified ECM ───────────────
def fig_resistance(recs, norm, ecm, gdir, name='Fig06_resistance_vs_temperature'):
    Tg = np.linspace(-20, 27, 60)
    tn = lambda T: (T - norm.stats['temp']['min']) / (norm.stats['temp']['max'] - norm.stats['temp']['min'])
    R0, RT = [], []
    for r in recs:
        p = r.get('rc_params')
        if r['name'] != OURS or not p: continue
        r0 = np.exp(p['r0_a'] + p['r0_b'] * (tn(Tg) - 1)) * 1e3
        rk = sum(np.exp(a + b * (tn(Tg) - 1)) for a, b in zip(p['rk_a'], p['rk_b'])) * 1e3
        R0.append(r0); RT.append(r0 + rk)
    fig, axs = plt.subplots(1, 2, figsize=(W2 * .78, 60 * MM), sharey=True, gridspec_kw={'wspace': .1})
    for ax, cu, title, key in [(axs[0], R0, 'Fast (ohmic) resistance', 'fast'), (axs[1], RT, 'Total resistance', 'total')]:
        if cu:
            cu = np.array(cu); ax.plot(Tg, cu.mean(0), color=COLORS[OURS], label='Learned (proposed, 5 seeds)')
            ax.fill_between(Tg, cu.min(0), cu.max(0), color=COLORS[OURS], alpha=.25, lw=0)
        if ecm:
            pts = sorted(ecm['per_temp'].values(), key=lambda p: p['T'])
            y = [(p['R0'] + p['R1']) * 1e3 if key == 'fast' else (p['R0'] + p['R1'] + p['R2']) * 1e3 for p in pts]
            ax.plot([p['T'] for p in pts], y, 'x', color='#2c3e50', ms=4, mew=1, label='Identified 2RC-ECM' + (' ($R_0+R_1$)' if key == 'fast' else ''))
        ax.axvspan(-21.5, -15, color='gray', alpha=.12, lw=0); ax.text(-21, .97, 'unseen', transform=ax.get_xaxis_transform(), fontsize=6.5, va='top')
        ax.set_yscale('log'); ax.set_yticks([20, 30, 50, 100, 200, 300]); plain_log(ax); ax.set_xlabel('Temperature (°C)'); ax.set_title(title)
    axs[0].set_ylabel('Resistance (mΩ)'); axs[0].legend(fontsize=6.5, loc='upper right'); letter(axs[0], '(a)'); letter(axs[1], '(b)')
    save(fig, gdir, name); return fig


# ─────────────── Fig. 7 — ablation and sensitivity (horizontal) ───────────────
def fig_ablation(tab, gdir, name='Fig07_ablation_sensitivity'):
    """tab: rows = variant (Turkish key), columns: 'grp' ('ref'|'abl'|'sens'), 'seen', 'seen_ci', 'cold', 'cold_ci'."""
    tab = tab.copy(); order = {'ref': 0, 'abl': 1, 'sens': 2}
    tab['_o'] = [(-1 if i == OURS else 0) for i in tab.index]
    tab = tab.sort_values(['grp', '_o', 'cold'], key=lambda s: s.map(order) if s.name == 'grp' else s)
    y, pos, prev = [], 0.0, None
    for g in tab.grp:
        if prev is not None and g != prev: pos += .6
        y.append(-pos); pos += 1; prev = g
    y = np.array(y)
    fig, axs = plt.subplots(1, 2, figsize=(W2, (22 + 6.5 * len(tab)) * MM), sharey=True, gridspec_kw={'wspace': .08})
    col = {'ref': '#c0392b', 'abl': '#5d6d7e', 'sens': '#aab7b8'}
    colors = [COLORS['Hibrit (GRU→Transformer)'] if i == 'Hibrit (GRU→Transformer)' else col[g] for i, g in zip(tab.index, tab.grp)]
    for ax, k, title in [(axs[0], 'seen', 'Training temperatures'), (axs[1], 'cold', 'Unseen −20 °C')]:
        ax.barh(y, tab[k], .7, xerr=tab[f'{k}_ci'], color=colors, error_kw={'elinewidth': .6, 'capsize': 1.2})
        ref = tab[tab.index == OURS][k]
        if len(ref): ax.axvline(float(ref.iloc[0]), color='#c0392b', lw=.6, ls='--')
        ax.set_xlabel('RMSE (SoC %)'); ax.set_title(title); ax.set_xlim(left=0); ax.grid(axis='y', visible=False)
    axs[0].set_yticks(y); axs[0].set_yticklabels([en(i) for i in tab.index])
    letter(axs[0], '(a)'); letter(axs[1], '(b)')
    save(fig, gdir, name); return fig


# ─────────────── Fig. 8 — robustness ───────────────
def fig_robustness(rob, scen_order, models, gdir, name='Fig08_robustness'):
    rob = rob.assign(k=np.where(rob.temp == '-20degC', 'cold', 'seen'))
    t = (rob.groupby(['k', 'senaryo', 'model', 'seed']).rmse.mean() * 100).groupby(['k', 'senaryo', 'model']).mean()
    fig, axs = plt.subplots(1, 2, figsize=(W2, 72 * MM), gridspec_kw={'wspace': .18})
    xs = np.arange(len(scen_order))
    for ax, k, title in [(axs[0], 'seen', 'Training temperatures'), (axs[1], 'cold', 'Unseen −20 °C')]:
        for m in models:
            if (k, scen_order[0], m) not in t.index: continue
            v = [t.get((k, s, m), np.nan) for s in scen_order]
            ax.plot(xs, v, marker='o', ms=2.2, lw=1.1 if m == OURS else .7, color=COLORS.get(m), label=en(m), zorder=3 if m == OURS else 2)
        ax.set_xticks(xs); ax.set_xticklabels([SCEN_EN.get(s, s) for s in scen_order], rotation=40, ha='right'); ax.set_title(title); ax.set_ylim(bottom=0)
    axs[0].set_ylabel('RMSE (SoC %)'); letter(axs[0], '(a)'); letter(axs[1], '(b)')
    h, l = axs[1].get_legend_handles_labels(); fig.legend(h, l, ncol=4, loc='upper center', bbox_to_anchor=(.5, 1.1))
    save(fig, gdir, name); return fig


# ─────────────── Fig. 9 — leakage audit ───────────────
def fig_leakage(aud, gdir, name='Fig09_leakage_audit'):
    fig, axs = plt.subplots(1, 2, figsize=(W2, 58 * MM), gridspec_kw={'wspace': .25})
    axs[0].hist(aud.max_corr, bins=40, color='#7f8c8d'); axs[0].set_xlabel('Max. current-profile correlation\n(test window vs. training/validation)'); axs[0].set_ylabel('Test windows')
    for t in TEMPS:
        g = aud[aud.temp == t]; axs[1].scatter(g.max_corr, g.model_rmse_60s * 100, s=1.2, color=TCOL[t], alpha=.45, rasterized=True, label=TLAB[t])
    axs[1].set_xlabel('Max. current-profile correlation'); axs[1].set_ylabel('Proposed model RMSE, 60 s (SoC %)'); axs[1].legend(markerscale=5, fontsize=6.5)
    letter(axs[0], '(a)'); letter(axs[1], '(b)')
    save(fig, gdir, name); return fig


# ─────────────── Fig. 10 — learning curves ───────────────
def fig_learning(recs, gdir, model=OURS, name='Fig10_learning_curves'):
    fig, ax = plt.subplots(figsize=(W1, 62 * MM))
    for i, r in enumerate([r for r in recs if r['name'] == model]):
        h = pd.DataFrame(r['train']['hist'])
        ax.plot(h.epoch, h.val_rmse * 100, color=COLORS[model], lw=.6, alpha=.75, label='Validation (NN)' if i == 0 else None)
        if 'test_rmse_diag' in h:
            d = h.dropna(subset=['test_rmse_diag'])
            ax.plot(d.epoch, d.test_rmse_diag * 100, 'o--', color='#2c3e50', ms=1.8, lw=.4, alpha=.7, label='Test (monitoring only)' if i == 0 else None)
        ax.axvline(r['train']['best_epoch'], color=COLORS[model], lw=.3, alpha=.4)
    ax.set_xlabel('Epoch'); ax.set_ylabel('RMSE (SoC %)'); ax.set_ylim(0, None); ax.legend(fontsize=6.5)
    save(fig, gdir, name); return fig


OFFS = {('LSTM', 'seen'): (-3, 3), ('GRU', 'seen'): (3, 3), ('Transformer', 'seen'): (3, 3), (OURS, 'seen'): (3, -9),
        (OURS_S, 'seen'): (-3, 3), (OURS_S, 'cold'): (3, -9)}


# ─────────────── Fig. 11 — accuracy vs size ───────────────
def fig_efficiency(eff_rmse, gdir, name='Fig11_accuracy_vs_size'):
    """eff_rmse: columns model, parametre, GPU_ms_medyan, CPU_ms_medyan, seen, cold."""
    fig, axs = plt.subplots(1, 2, figsize=(W2 * .8, 60 * MM), sharex=True, gridspec_kw={'wspace': .3})
    for ax, k, title in [(axs[0], 'seen', 'Training temperatures'), (axs[1], 'cold', 'Unseen −20 °C')]:
        for _, r in eff_rmse.iterrows():
            ax.scatter(r['parametre'], r[k], s=14, color=COLORS.get(r['model']), zorder=3, edgecolor='white', lw=.4)
            off = OFFS.get((r['model'], k), (3, 2))
            ax.annotate(en(r['model']).replace(' (GRU→Transformer)', '').replace(' (RC-hybrid)', ''), (r['parametre'], r[k]), xytext=off,
                        textcoords='offset points', fontsize=6, ha='right' if off[0] < 0 else 'left')
        ax.set_xscale('log'); ax.set_xlabel('Trainable parameters'); ax.set_title(title)
        if k == 'cold': ax.set_ylim(bottom=0)
        else: lo, hi = eff_rmse[k].min(), eff_rmse[k].max(); ax.set_ylim(lo - .25 * (hi - lo) - .02, hi + .25 * (hi - lo) + .02)
        ax.set_xlim(eff_rmse.parametre.min() / 1.6, eff_rmse.parametre.max() * 2.2)
    axs[0].set_ylabel('RMSE (SoC %)'); letter(axs[0], '(a)'); letter(axs[1], '(b)')
    save(fig, gdir, name); return fig


# ─────────────── Fig. 12 — current-input control experiment (03b) ───────────────
CI_ORDER = [OURS, 'RC + akım girişi', 'RC: yalnız ohmik kol', 'RC: yalnız ohmik + akım girişi', 'Hibrit (GRU→Transformer)']
CI_LAB = {OURS: 'Proposed', 'RC + akım girişi': 'Proposed\n+ current', 'RC: yalnız ohmik kol': 'Ohmic\nonly',
          'RC: yalnız ohmik + akım girişi': 'Ohmic only\n+ current', 'Hibrit (GRU→Transformer)': 'Hybrid\n(no comp.)'}
CI_COL = {OURS: '#c0392b', 'RC + akım girişi': '#c0392b', 'RC: yalnız ohmik kol': '#e67e22',
          'RC: yalnız ohmik + akım girişi': '#e67e22', 'Hibrit (GRU→Transformer)': '#2c3e50'}


def fig_current_input(D, gdir, pairs_p=None, name='Fig12_current_input_control'):
    """D: rows with model, seed, temp, rmse, bias. Hatched bars = current also fed to the backbone.
    pairs_p: {(a, b): p} brackets on the −20 °C panel (paired Wilcoxon, seed × cycle)."""
    from scipy import stats as st
    models = [m for m in CI_ORDER if m in set(D.model)]
    D = D.assign(k=np.where(D.temp == '-20degC', 'cold', 'seen'))
    sl = D.groupby(['model', 'k', 'seed'])[['rmse', 'bias']].mean() * 100
    fig, axs = plt.subplots(1, 3, figsize=(W2, 52 * MM), sharey=True, gridspec_kw={'wspace': .12})
    ys = np.arange(len(models))[::-1]; rng = np.random.default_rng(0)
    for ax, (k, col, ttl) in zip(axs, [('seen', 'rmse', '(a) RMSE, training temperatures'), ('cold', 'rmse', '(b) RMSE, unseen −20 °C'),
                                      ('cold', 'bias', '(c) Bias, unseen −20 °C')]):
        right = 0
        for y, m in zip(ys, models):
            v = sl.loc[(m, k), col].values; n = len(v)
            ci = st.t.ppf(.975, n - 1) * v.std(ddof=1) / np.sqrt(n) if n > 1 else 0
            hatch = '////' if 'akım girişi' in m else None
            ax.barh(y, v.mean(), .66, xerr=ci, color=CI_COL[m], alpha=.85 if hatch is None else .5, hatch=hatch,
                    edgecolor='white', linewidth=0, error_kw={'elinewidth': .6, 'capsize': 1.5})
            ax.scatter(v, y + rng.uniform(-.15, .15, n), s=5, color='k', zorder=3, linewidths=0)
            right = max(right, v.max(), v.mean() + ci)
        ax.set_title(ttl, loc='left', fontsize=8); ax.grid(axis='y', visible=False)
        if col == 'bias': ax.axvline(0, color='k', lw=.6); ax.set_xlabel('Mean error (SoC %)')
        else: ax.set_xlim(left=0); ax.set_xlabel('RMSE (SoC %)')
        if k == 'cold' and col == 'rmse' and pairs_p:
            for j, ((a, b), p) in enumerate(pairs_p.items()):
                if a not in models or b not in models: continue
                y1, y2 = ys[models.index(a)], ys[models.index(b)]; x = right * (1.05 + .22 * j); t = right * .03
                ax.plot([x - t, x, x, x - t], [y1, y1, y2, y2], color='k', lw=.6)
                ax.text(x + t, (y1 + y2) / 2, f'p = {p:.2g}' if p >= 1e-3 else 'p < 0.001', va='center', fontsize=6.3)
            ax.set_xlim(0, right * (1.45 + .22 * len(pairs_p)))
    axs[0].set_yticks(ys); axs[0].set_yticklabels([CI_LAB[m].replace('\n', ' ') for m in models])
    save(fig, gdir, name); return fig
