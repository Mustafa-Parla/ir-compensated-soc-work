"""Eğitim, tahmin, değerlendirme, verimlilik ölçümleri (V26)."""
import io, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from scipy.signal import lfilter


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


# ───────────────────────────── metrikler ─────────────────────────────
def metrics(pred, true):
    """SoC birim aralığında (0–1); raporlarda ×100 → % puan. Etiket normalize edilmez."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    e = pred - true
    return {'rmse': float(np.sqrt(np.mean(e ** 2))), 'mae': float(np.mean(np.abs(e))), 'maxe': float(np.max(np.abs(e))),
            'r2': float(1 - np.sum(e ** 2) / (np.sum((true - true.mean()) ** 2) + 1e-12)),
            'jitter': float(np.sqrt(np.mean((np.diff(pred) - np.diff(true)) ** 2))) if len(e) > 1 else np.nan,
            'bias': float(np.mean(e))}


def ema(x, tau):
    """Nedensel birinci derece üstel yumuşatma, zaman sabiti tau (s)."""
    if not tau: return np.asarray(x)
    al = 1 - np.exp(-1.0 / tau)
    return lfilter([al], [1, -(1 - al)], x, zi=[(1 - al) * x[0]])[0]


def decompose(err):
    u = np.linspace(0, 1, len(err)); b, a = np.polyfit(u, err, 1)
    return {'ofset': float(a), 'sürüklenme': float(b), 'artık_rmse': float(np.sqrt(np.mean((err - a - b * u) ** 2)))}


def ci95(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; n = len(x)
    if n < 2: return np.nan
    return float(stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n))


# ───────────────────────────── tahmin / eğitim ─────────────────────────────
@torch.no_grad()
def predict(model, data, bs=1024, ablate=None, keep=()):
    model.eval(); P, Y, K = [], [], {k: [] for k in keep}
    for X, y in data.batches(bs, False):
        p, o = model(X, ablate)
        P.append(p.float().cpu().numpy()); Y.append(y.cpu().numpy())
        for k in keep:
            v = o.get(k) if o else None
            if v is not None: K[k].append(v[:, -1].float().cpu().numpy())
    out = {'pred': np.concatenate(P), 'true': np.concatenate(Y)}
    out.update({k: np.concatenate(v) for k, v in K.items() if v})
    return out


def mean_file_rmse(model, sets, bs=1024):
    return float(np.mean([metrics(**{k: v for k, v in predict(model, d, bs).items() if k in ('pred', 'true')})['rmse']
                          for d in sets.values()])) if sets else float('nan')


def fit(model, tr_data, val_sets, cfg, lr, ckpt_path, gen, test_sets=None, log=print, log_extra=None):
    """Adam + ReduceLROnPlateau(0.5, 5), grad clip 1.0, MSE; seçim ve erken durma = val (dosya ort. RMSE).
    test_sets verilirse her cfg['TEST_EVERY'] epoch'ta YALNIZ TANI amaçlı test RMSE kaydedilir."""
    dev = next(model.parameters()).device
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=cfg['LR_PATIENCE'])
    mse = nn.MSELoss()
    best, best_ep, wait, hist = float('inf'), 0, 0, []
    if dev.type == 'cuda': torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for ep in range(1, cfg['EPOCHS'] + 1):
        te = time.time(); model.train(); tot = nb = 0
        for X, y in tr_data.batches(cfg['BATCH'], True, gen):
            opt.zero_grad(set_to_none=True); p, _ = model(X); loss = mse(p, y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item(); nb += 1
        v = mean_file_rmse(model, val_sets, cfg['PRED_BS'])
        row = {'epoch': ep, 'train_mse': tot / nb, 'val_rmse': v, 'lr': opt.param_groups[0]['lr'], 'epoch_s': time.time() - te}
        if test_sets and cfg.get('TEST_EVERY') and (ep % cfg['TEST_EVERY'] == 0 or ep == 1):
            row['test_rmse_diag'] = mean_file_rmse(model, {k: d for k, d in test_sets.items() if k[0] != '-20degC'}, cfg['PRED_BS'])
        hist.append(row)
        mark = ''
        if v < best:
            best, best_ep, wait, mark = v, ep, 0, ' ✅'
            torch.save(model.state_dict(), ckpt_path)
        else:
            wait += 1
        log(f"  ep {ep:3d} | loss {tot/nb:.5f} | val {v*100:.3f}%" + (f" | (tanı) test {row['test_rmse_diag']*100:.3f}%" if 'test_rmse_diag' in row else '')
            + f" | lr {row['lr']:.1e} | {row['epoch_s']:.0f}s{mark}" + (log_extra(model) if log_extra else ''))
        sch.step(v)
        if wait >= cfg['PATIENCE']: log('  ⏹ erken durma'); break
    peak = torch.cuda.max_memory_allocated() / 1e9 if dev.type == 'cuda' else float('nan')
    return {'hist': hist, 'best_epoch': best_ep, 'best_val': best, 'train_time_s': time.time() - t0, 'epochs_run': len(hist),
            'epoch_time_mean_s': float(np.mean([h['epoch_s'] for h in hist])), 'train_peak_mem_gb': peak}


def evaluate(model, test_sets, te_dfs, cfg, ablations=(), keep=(), series=False):
    """Dosya bazında metrikler (ham + çıkış EMA'sı) + hata ayrıştırması + ablasyon ΔRMSE. series=True → zaman serileri."""
    rows, ser = [], []
    H, tau = cfg['HEAD_S'], cfg['EMA_TAU']
    for (t, c), dat in test_sets.items():
        r = predict(model, dat, cfg['PRED_BS'], keep=keep)
        pe = ema(r['pred'], tau); err, erre = r['pred'] - r['true'], pe - r['true']
        m, me = metrics(r['pred'], r['true']), metrics(pe, r['true'])
        row = {'temp': t, 'cycle': c, **m, **{f'{k}_ema': v for k, v in me.items()},
               'rmse_head': float(np.sqrt(np.mean(err[:H] ** 2))), 'rmse_tail': float(np.sqrt(np.mean(err[H:] ** 2))), **decompose(err)}
        for ab in ablations:
            ra = predict(model, dat, cfg['PRED_BS'], ablate=ab)
            row[f'drmse_{ab}'] = metrics(ra['pred'], ra['true'])['rmse'] - m['rmse']
        rows.append(row)
        if series:
            d = te_dfs[(t, c)]; n = len(r['pred']); o0 = 0 if getattr(dat, 'pad', False) else dat.W - 1; sl = slice(o0, o0 + n)
            s = pd.DataFrame({'temp': t, 'cycle': c, 't_s': d.time.values[sl] - d.attrs['time0'], 'V': d.v10.values[sl],
                              'I': d.i10.values[sl], 'T': d.temp.values[sl], 'soc_true': r['true'], 'soc_pred': r['pred'],
                              'soc_pred_ema': pe, 'err': err, 'err_ema': erre})
            if 'eta' in r:
                s['eta'] = r['eta']; s['eta_ohm'] = r['eta0']
                for j in range(r['etak'].shape[1] if r['etak'].ndim == 2 else 0): s[f'eta_rc{j + 1}'] = r['etak'][:, j]
                s['ocv_hat'] = s['V'] - s['eta']
            ser.append(s)
    return rows, (pd.concat(ser, ignore_index=True) if ser else None)


# ───────────────────────────── verimlilik ─────────────────────────────
def count_macs(model, X):
    """Kanca tabanlı MAC sayımı (bir pencere). Linear, Conv1d, GRU/LSTM, MultiheadAttention + RC detector analitik."""
    macs = [0]; hooks = []
    def lin(m, i, o): macs[0] += i[0].numel() // i[0].shape[-1] * m.in_features * m.out_features
    def conv(m, i, o): macs[0] += o.numel() * (m.in_channels // m.groups) * m.kernel_size[0]
    def rnn(m, i, o):
        x = i[0]; B, T = x.shape[:2]; g = 4 if isinstance(m, nn.LSTM) else 3; h = m.hidden_size
        for l in range(m.num_layers): macs[0] += B * T * g * ((m.input_size if l == 0 else h) * h + h * h)
    def mha(m, i, o):
        q = i[0]; B, T, E = q.shape; macs[0] += B * T * E * E * 4 + 2 * B * T * T * E
    for mod in model.modules():
        if isinstance(mod, nn.MultiheadAttention): hooks.append(mod.register_forward_hook(mha))
        elif isinstance(mod, nn.Linear) and type(mod).__name__ == 'Linear': hooks.append(mod.register_forward_hook(lin))
        elif isinstance(mod, nn.Conv1d): hooks.append(mod.register_forward_hook(conv))
        elif isinstance(mod, (nn.GRU, nn.LSTM)): hooks.append(mod.register_forward_hook(rnn))
    was = model.training; model.train()                     # hızlı yol (fastpath) kancaları atlamasın
    with torch.no_grad(): model(X)
    model.train(was)
    for h in hooks: h.remove()
    extra = model.rc.macs(X.shape[1]) if getattr(model, 'rc', None) is not None else 0
    return int(macs[0] + extra)


def efficiency(model, sample_X, n_warm=20, n_rep=200, cpu_threads=None):
    """Parametre, model boyutu, MAC, tek pencere gecikmesi (GPU ve CPU, ms; medyan ve p95), çıkarım bellek tepesi."""
    dev = next(model.parameters()).device
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buf = io.BytesIO(); torch.save(model.state_dict(), buf); size_mb = buf.tell() / 1e6
    X1 = sample_X[:1]
    out = {'parametre': n_par, 'boyut_MB': size_mb, 'MAC_pencere': count_macs(model, X1)}
    model.eval()
    def lat(m, x, sync):
        ts = []
        with torch.no_grad():
            for i in range(n_warm + n_rep):
                if sync: torch.cuda.synchronize()
                t0 = time.perf_counter(); m(x)
                if sync: torch.cuda.synchronize()
                if i >= n_warm: ts.append((time.perf_counter() - t0) * 1e3)
        return float(np.median(ts)), float(np.percentile(ts, 95))
    if dev.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(); out['GPU_ms_medyan'], out['GPU_ms_p95'] = lat(model, X1, True)
        out['GPU_çıkarım_bellek_MB'] = torch.cuda.max_memory_allocated() / 1e6
    import copy
    mc = copy.deepcopy(model).cpu()
    if cpu_threads: torch.set_num_threads(cpu_threads)
    out['CPU_ms_medyan'], out['CPU_ms_p95'] = lat(mc, X1.cpu(), False)
    out['CPU_iş_parçacığı'] = torch.get_num_threads()
    return out
