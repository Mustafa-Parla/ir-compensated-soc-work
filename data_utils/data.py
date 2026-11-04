"""
Veri hattı (V26) — Panasonic 18650PF.

1. load_raw         : .mat → ham 10 Hz diziler
2. step_sync_index  : basamak senkron 1 Hz indeksleri (kayıt 10 Hz, akım 1 Hz basamaklı; basamak kenarından önceki
                      oturmuş örnek seçilir; baştaki seyrek 60 s dinlenme kaydı atılır)
3. to_frame         : 1 Hz tablo; V, I = o saniyedeki 10 ham örneğin ortalaması; türetilmiş özellikler (derive)
4. Normalizer       : min-max, yalnız eğitim dosyalarından
5. WindowData       : dosya sınırını geçmeyen kayan pencereler (stride = 1); hedef SoC(t+1)
6. perturb          : sağlamlık senaryoları (gürültü, sensör ofseti, eksik örnek, ortadan başlama)
7. leakage_audit    : akım profili kopya/korelasyon denetimi (eğitim+val ↔ test)
"""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
import torch

from .files import DATA_FOLDERS, TEMPS, ROLE_FILES, TEST_FILES

NOMINAL_CAPACITY = 2.9                     # Ah — etiket: SoC = (2.9 + Ah) / 2.9
EN_ALL = ['v10', 'i10', 'temp', 'v10_avg']            # ana kol özellikleri (normalize)
DET_COLS = ['i10', 'i_ma30', 'i_ma60', 'temp', 'di_abs']  # detector özellikleri (normalize)
NE, ND = len(EN_ALL), len(DET_COLS)
IV, II = NE + ND, NE + ND + 1              # X içinde ham V (V) ve ham I (A)
NX = NE + ND + 2


# ───────────────────────────── yükleme ─────────────────────────────
def load_raw(filepath):
    try:
        raw = sio.loadmat(str(filepath))
    except Exception:
        import mat73
        raw = mat73.loadmat(str(filepath))
    ds = raw['meas']
    get = lambda k: np.asarray(ds[k][0][0]).flatten().astype(float) if isinstance(ds, np.ndarray) \
        else np.asarray(ds[k]).flatten().astype(float)
    return {k: get(k) for k in ['Time', 'Voltage', 'Current', 'Ah', 'Battery_Temp_degC']}


def step_sync_index(t, I, rate=10, block=600, min_dt=0.5):
    n = len(t); dt = np.diff(t)
    start = 0
    while start < n - 1 and dt[start] > min_dt:
        start += 1
    cuts = [start] + [k + 1 for k in np.where(dt > min_dt)[0] if k + 1 > start] + [n]
    dI = np.abs(np.diff(I, prepend=I[0]))
    keep = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        for a2 in range(a, b, block):
            b2 = min(b, a2 + block)
            if b2 - a2 < 2 * rate:
                keep.extend(range(min(a2 + rate - 1, b2 - 1), b2, rate)); continue
            edge = int(np.argmax([dI[a2 + p:b2:rate].mean() for p in range(rate)]))
            keep.extend(range(a2 + (edge - 1) % rate, b2, rate))
    idx = np.unique(np.array(keep, dtype=int))
    return idx[idx >= start], start


def derive(df, ma=(30, 60), vavg=10):
    """Türetilmiş özellikler (bozulma sonrası da yeniden hesaplanır)."""
    df['v10_avg'] = df['v10'].rolling(vavg, min_periods=1).mean()
    for w in ma:
        df[f'i_ma{w}'] = df['i10'].rolling(w, min_periods=1).mean()
    df['di_abs'] = df['i10'].diff().abs().fillna(0)
    return df


def to_frame(raw, rate=10, avg=10, ma=(30, 60), vavg=10):
    idx, start = step_sync_index(raw['Time'], raw['Current'], rate)
    V, I = raw['Voltage'], raw['Current']
    win = np.clip(idx[:, None] - (avg - 1) + np.arange(avg), start, None)
    ah = raw['Ah'][idx]
    df = pd.DataFrame({'time': raw['Time'][idx], 'v10': V[win].mean(1), 'i10': I[win].mean(1), 'ah': ah,
                       'temp': raw['Battery_Temp_degC'][idx],
                       'soc': np.clip((NOMINAL_CAPACITY + ah) / NOMINAL_CAPACITY, 0.0, 1.0)})
    df.attrs['time0'] = float(raw['Time'][0])
    df.attrs['n_raw'] = int(len(raw['Time']))
    return derive(df, ma, vavg)


def label_check(df):
    return {'süre (h)': (df.time.iloc[-1] - df.time.iloc[0]) / 3600, 'örnek (1 Hz)': len(df), 'SoC baş': df.soc.iloc[0],
            'SoC son': df.soc.iloc[-1], 'deşarj Ah': -(df.ah.min() - df.ah.iloc[0]),
            'şarj Ah (rejen.)': float(np.clip(np.diff(df.ah.values), 0, None).sum()),
            'V min': df.v10.min(), 'V maks': df.v10.max(), 'T ort': df.temp.mean(), 'T maks−min': df.temp.max() - df.temp.min()}


def load_all(data_dir, rate=10, avg=10, verbose=True):
    """Tüm rol ve test dosyalarını yükler. Dönüş: role_dfs {(temp, rol): df}, te_dfs {(temp, çevrim): df}, dosya tablosu,
    ön işleme süresi (ms / 1 Hz örnek)."""
    data_dir = Path(data_dir)
    role_dfs, te_dfs, rows, missing = {}, {}, [], []
    t_pre, n_pre = 0.0, 0
    if not data_dir.exists():
        raise FileNotFoundError(f'Veri klasörü bulunamadı: {data_dir}')
    def _load(t, fn):
        nonlocal t_pre, n_pre
        fp = data_dir / DATA_FOLDERS[t] / fn
        if not fp.exists():
            missing.append(str(fp))
            return None
        raw = load_raw(fp); t0 = time.perf_counter(); d = to_frame(raw, rate, avg); t_pre += time.perf_counter() - t0; n_pre += len(d)
        return d
    for t, rf in ROLE_FILES.items():
        for role, fn in rf.items():
            d = _load(t, fn)
            if d is None: continue
            role_dfs[(t, role)] = d; rows.append({'küme': 'eğitim' if role != 'NN' else 'val', 'temp': t, 'kayıt': role, 'dosya': fn, **label_check(d)})
            if verbose: print(f'✅ {t} {role:3s} → {len(d):,}')
    for t, c, fn in TEST_FILES:
        d = _load(t, fn)
        if d is None: continue
        te_dfs[(t, c)] = d; rows.append({'küme': 'test', 'temp': t, 'kayıt': c, 'dosya': fn, **label_check(d)})
    if missing:
        print(f'⚠️  {len(missing)} dosya bulunamadı. İlk 5:'); [print('   ', m) for m in missing[:5]]
        print(f'   Veri klasöründekiler: {sorted(x.name for x in data_dir.iterdir())}')
    if not role_dfs:
        raise FileNotFoundError(f'Hiç eğitim dosyası yüklenemedi. Beklenen yapı: {data_dir}\\DriveCycles25deg\\<dosya>.mat (bkz. yukarıdaki liste)')
    if verbose: print(f'Rol dosyası {len(role_dfs)} | test {len(te_dfs)}')
    return role_dfs, te_dfs, pd.DataFrame(rows), 1e3 * t_pre / max(n_pre, 1)


# ───────────────────────────── normalizasyon / pencereler ─────────────────────────────
class Normalizer:
    def __init__(self, stats=None):
        self.stats = stats or {}
    def fit(self, dfs):
        d = pd.concat(dfs, ignore_index=True)
        for c in dict.fromkeys(EN_ALL + DET_COLS):
            self.stats[c] = {'min': float(d[c].min()), 'max': float(d[c].max())}
        self.stats['t_mean_norm'] = float(self.mm(d['temp'].values, 'temp').mean())
        return self
    def mm(self, x, c):
        mn, mx = self.stats[c]['min'], self.stats[c]['max']
        return (x - mn) / (mx - mn + 1e-8)
    def t_norm(self, T_c):
        return (T_c - self.stats['temp']['min']) / (self.stats['temp']['max'] - self.stats['temp']['min'] + 1e-8)
    def transform(self, df):
        en = np.stack([self.mm(df[c].values, c) for c in EN_ALL], 1)
        de = np.stack([self.mm(df[c].values, c) for c in DET_COLS], 1)
        return np.concatenate([en, de, df[['v10', 'i10']].values], 1).astype(np.float32)


class WindowData:
    """Kayan pencereler (stride = 1, dosya sınırını geçmez). batches → X (B, W, NX), y = SoC[t+1].
    pad_start=True: dosya başı W−1 kez ilk örnekle doldurulur (soğuk başlangıç: geçmiş yok) → tahmin t = 1'den başlar."""
    def __init__(self, arrays, socs, window_size, device, pad_start=False):
        X, soc, starts = [], [], []
        off = 0
        for a, s in zip(arrays, socs):
            if pad_start:
                a = np.concatenate([np.repeat(a[:1], window_size - 1, 0), a]); s = np.concatenate([np.repeat(s[:1], window_size - 1), s])
            n = len(a); X.append(a); soc.append(s)
            if n > window_size:
                starts.append(np.arange(off, off + n - window_size))
            off += n
        f32 = lambda a: torch.tensor(np.concatenate(a), dtype=torch.float32, device=device)
        self.X, self.soc = f32(X), f32(soc)
        self.starts = torch.tensor(np.concatenate(starts), dtype=torch.long, device=device)
        self.W, self.device, self.pad = window_size, device, pad_start
        self.ar = torch.arange(window_size, device=device)

    def __len__(self):
        return len(self.starts)

    def batches(self, batch_size, shuffle, generator=None):
        n = len(self.starts)
        order = (torch.randperm(n, generator=generator).to(self.device) if shuffle
                 else torch.arange(n, device=self.device))
        for i in range(0, n, batch_size):
            s = self.starts[order[i:i + batch_size]]
            yield self.X[s[:, None] + self.ar], self.soc[s + self.W]


# ───────────────────────────── sağlamlık senaryoları ─────────────────────────────
ROBUST_SCENARIOS = {
    'temiz': {},
    'gürültü düşük': {'noise_I': 0.02, 'noise_V': 0.002, 'noise_T': 0.2},
    'gürültü orta': {'noise_I': 0.05, 'noise_V': 0.005, 'noise_T': 0.5},
    'gürültü yüksek': {'noise_I': 0.10, 'noise_V': 0.010, 'noise_T': 1.0},
    'I ofset +0.1 A': {'bias_I': 0.1},
    'V ofset +10 mV': {'bias_V': 0.010},
    'T ofset +2 °C': {'bias_T': 2.0},
    'eksik örnek %5': {'drop': 0.05},
    'eksik örnek %20': {'drop': 0.20},
    'başlangıç SoC 0.8': {'start_soc': 0.8},
    'başlangıç SoC 0.6': {'start_soc': 0.6},
}


def perturb(df, scen, seed=0):
    """Ölçülen sinyalleri (v10, i10, temp) bozar; etiket (soc) değişmez; türetilmiş özellikler yeniden hesaplanır.
    start_soc: dosya, SoC'un ilk kez start_soc'un altına indiği andan başlatılır (geçmiş yok → soğuk başlangıç)."""
    rng = np.random.default_rng(seed)
    d = df.copy(); d.attrs = dict(df.attrs)
    n = len(d)
    for c, k in [('i10', 'I'), ('v10', 'V'), ('temp', 'T')]:
        if scen.get(f'noise_{k}'): d[c] = d[c] + rng.normal(0, scen[f'noise_{k}'], n)
        if scen.get(f'bias_{k}'): d[c] = d[c] + scen[f'bias_{k}']
    if scen.get('drop'):                                           # örnek kaybı → son geçerli değeri tut
        m = rng.random(n) < scen['drop']; m[0] = False
        for c in ['v10', 'i10', 'temp']:
            d[c] = d[c].mask(m).ffill()
    if scen.get('start_soc'):
        k = int(np.argmax(d.soc.values <= scen['start_soc']))
        d = d.iloc[k:].reset_index(drop=True); d.attrs = dict(df.attrs)
    return derive(d)


# ───────────────────────────── sızıntı denetimi ─────────────────────────────
def _zwin(i, L, stride, min_std):
    idx = np.arange(0, len(i) - L + 1, stride)
    w = np.stack([i[a:a + L] for a in idx]) if len(idx) else np.zeros((0, L))
    sd = w.std(1)
    ok = sd > min_std
    z = (w[ok] - w[ok].mean(1, keepdims=True)) / sd[ok, None]
    return z.astype(np.float32), idx[ok]


def leakage_audit(ref_dfs, te_dfs, L=256, stride_ref=4, stride_te=16, min_std=0.05, device='cpu'):
    """Her test penceresinin (akım profili, z-normalize) eğitim+val pencereleriyle en yüksek Pearson korelasyonu.
    Ayrıca en iyi eşleşmede ortalama mutlak akım farkı (A). Dönüş: pencere tablosu (temp, cycle, start, max_corr, mad_A, ref)."""
    R, Rraw, Rid = [], [], []
    for k, d in ref_dfs.items():
        z, idx = _zwin(d.i10.values, L, stride_ref, min_std)
        R.append(z); Rid += [(k, a) for a in idx]
        Rraw.append(np.stack([d.i10.values[a:a + L] for a in idx]).astype(np.float32) if len(idx) else np.zeros((0, L), np.float32))
    R = torch.tensor(np.concatenate(R), device=device); Rraw = torch.tensor(np.concatenate(Rraw), device=device)
    rows = []
    for (t, c), d in te_dfs.items():
        z, idx = _zwin(d.i10.values, L, stride_te, min_std)
        if not len(idx): continue
        Z = torch.tensor(z, device=device)
        Traw = torch.tensor(np.stack([d.i10.values[a:a + L] for a in idx]).astype(np.float32), device=device)
        for a0 in range(0, len(Z), 256):
            C = Z[a0:a0 + 256] @ R.T / L
            mc, am = C.max(1)
            mad = (Traw[a0:a0 + 256] - Rraw[am]).abs().mean(1).cpu().numpy()
            mc, am = mc.cpu().numpy(), am.cpu().numpy()
            for j in range(len(mc)):
                rk = Rid[int(am[j])]
                rows.append({'temp': t, 'cycle': c, 'start': int(idx[a0 + j]), 'max_corr': float(mc[j]), 'mad_A': float(mad[j]),
                             'ref': f'{rk[0][0]} {rk[0][1]} @{rk[1]}'})
    return pd.DataFrame(rows)


def window_counts(role_dfs, te_dfs, train_roles, val_roles, W):
    cnt = lambda d: max(len(d) - W, 0)
    return pd.DataFrame([{'küme': 'eğitim', 'pencere': sum(cnt(d) for (t, r), d in role_dfs.items() if r in train_roles)},
                         {'küme': 'val (NN)', 'pencere': sum(cnt(d) for (t, r), d in role_dfs.items() if r in val_roles)},
                         {'küme': 'test', 'pencere': sum(cnt(d) for d in te_dfs.values())}]).assign(W=W, stride=1)
