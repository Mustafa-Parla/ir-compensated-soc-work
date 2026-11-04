"""
Deney yöneticisi (V26): klasörler, veri, model tanımları (spec), ayarlama ızgaraları, koşu + kayıt + devam etme.

Klasörler (ROOT = C:\\Makale2):
  data\            ham .mat (mevcut)
  results\checkpoints\<aşama>\<model>_s<seed>.pt     ağırlıklar
  results\runs\<aşama>\<model>_s<seed>.json          spec, hiperparametreler, öğrenme eğrisi, süreler, dosya bazında metrikler
  results\series\<aşama>\<model>_s<seed>.parquet     tahmin / gerçek / hata / η zaman serileri
  results\configs\                                   seçilen hiperparametreler (selected.json), ortam bilgisi
  results\excel\                                     Excel tabloları
  graphs\                                            yüksek çözünürlüklü grafikler (PNG 300 dpi + PDF)
"""
import json, copy, time, platform, itertools
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from data_utils import (load_all, Normalizer, WindowData, TEMPS, perturb, ROBUST_SCENARIOS)
from models.nets import build
from models import engine

def save_table(df, path):
    """Parquet (pyarrow varsa), yoksa aynı adla CSV."""
    path = Path(path)
    try: df.to_parquet(path, index=False)
    except Exception: df.to_csv(path.with_suffix('.csv'), index=False)


def load_table(path):
    path = Path(path)
    if path.exists():
        try: return pd.read_parquet(path)
        except Exception: pass
    return pd.read_csv(path.with_suffix('.csv'))


def table_exists(path):
    path = Path(path); return path.exists() or path.with_suffix('.csv').exists()


TRAIN_ROLES = ['C1', 'C2', 'C3', 'C4']
VAL_ROLES = ['NN']

CFG = {'W': 256, 'BATCH': 256, 'EPOCHS': 150, 'PATIENCE': 20, 'LR_PATIENCE': 5, 'PRED_BS': 1024,
       'HEAD_S': 600, 'EMA_TAU': 5, 'TEST_EVERY': 5,
       'TUNE_EPOCHS': 80, 'TUNE_PATIENCE': 12,
       'TUNE_SEED': 7, 'FINAL_SEEDS': [101, 202, 303, 404, 505], 'ABL_SEEDS': [101, 202, 303]}

HYB = {'hidden': 128, 'gru_layers': 3, 'd_model': 128, 'nhead': 4, 'trans_layers': 3, 'dropout': 0.1}
RC_DEFAULT = {'taus': [3, 7, 12], 'ohm_smooth': 3, 'use_T': True, 'det': False, 'mod_max': 0.3}

# Model tanımları (başlangıç; ayarlama ızgaraları bunların üzerine yazar)
BASE_SPECS = {
    'Önerilen (RC-Hibrit)': {'family': 'rc', 'params': dict(HYB), 'rc': dict(RC_DEFAULT), 'lr': 8e-4},
    'Hibrit (GRU→Transformer)': {'family': 'hyb', 'params': dict(HYB), 'lr': 8e-4},
    'LSTM': {'family': 'lstm', 'params': {'hidden': 128, 'layers': 2, 'dropout': 0.1}, 'lr': 8e-4},
    'GRU': {'family': 'gru', 'params': {'hidden': 128, 'layers': 2, 'dropout': 0.1}, 'lr': 8e-4},
    'CNN-LSTM': {'family': 'cnn_lstm', 'params': {'conv_ch': 64, 'kernel': 3, 'conv_layers': 2, 'hidden': 128, 'layers': 2, 'dropout': 0.1}, 'lr': 8e-4},
    'Transformer': {'family': 'transformer', 'params': {'d_model': 128, 'nhead': 4, 'layers': 3, 'dropout': 0.1}, 'lr': 8e-4},
}
SHORT = {'Önerilen (RC-Hibrit)': 'rc_hyb', 'Hibrit (GRU→Transformer)': 'hyb', 'LSTM': 'lstm', 'GRU': 'gru',
         'CNN-LSTM': 'cnn_lstm', 'Transformer': 'transformer'}

# Eşit bütçeli ayarlama ızgaraları (her model 8 kombinasyon; seçim yalnız NN val)
GRIDS = {
    'Önerilen (RC-Hibrit)': {'params.hidden': [64, 128], 'params.dropout': [0.1, 0.2], 'lr': [4e-4, 8e-4]},
    'Hibrit (GRU→Transformer)': {'params.hidden': [64, 128], 'params.dropout': [0.1, 0.2], 'lr': [4e-4, 8e-4]},
    'LSTM': {'params.hidden': [64, 128], 'params.layers': [2, 3], 'lr': [4e-4, 8e-4]},
    'GRU': {'params.hidden': [64, 128], 'params.layers': [2, 3], 'lr': [4e-4, 8e-4]},
    'CNN-LSTM': {'params.conv_ch': [32, 64], 'params.hidden': [64, 128], 'lr': [4e-4, 8e-4]},
    'Transformer': {'params.d_model': [64, 128], 'params.layers': [2, 3], 'lr': [4e-4, 8e-4]},
}

RC_ABLATIONS_TEST = ['eta0', 'ohm0', 'rc1', 'rc2', 'rc3', 'tconst']


def set_path(spec, key, val):
    d = spec
    *ks, last = key.split('.')
    for k in ks: d = d.setdefault(k, {})
    d[last] = val


def grid_specs(name):
    base, g = BASE_SPECS[name], GRIDS[name]
    keys = list(g)
    out = []
    for vals in itertools.product(*[g[k] for k in keys]):
        s = copy.deepcopy(base)
        for k, v in zip(keys, vals): set_path(s, k, v)
        if s['family'] in ('rc', 'hyb'): s['params']['d_model'] = s['params']['hidden']
        tag = '_'.join(f"{k.split('.')[-1]}{v}" for k, v in zip(keys, vals))
        out.append((f'{SHORT[name]}__{tag}', s))
    return out


def ablation_specs(sel):
    """Bileşen ablasyonu ve duyarlılık: seçilen önerilen modelin hiperparametreleri üzerine tek değişiklik."""
    b = sel['Önerilen (RC-Hibrit)']; h = sel['Hibrit (GRU→Transformer)']
    def v(**kw):
        s = copy.deepcopy(b)
        for k, x in kw.items(): set_path(s, k, x)
        return s
    noI = copy.deepcopy(h); noI['noI'] = True
    abl = {'Akımsız (noI)': noI,
           'RC: R sıcaklıktan bağımsız': v(**{'rc.use_T': False}),
           'RC: yalnız ohmik kol': v(**{'rc.taus': []}),
           'RC: tek RC kolu': v(**{'rc.taus': [7]}),
           'RC + serbest CNN detector': v(**{'rc.det': True}),
           'RC: ohmik yumuşatma yok': v(**{'rc.ohm_smooth': 1})}
    sens = {'Duyarlılık: W = 128': v(window=128),
            'Duyarlılık: ohmik yumuşatma 5 s': v(**{'rc.ohm_smooth': 5}),
            'Duyarlılık: τ başlangıcı 1/10/60 s': v(**{'rc.taus': [1, 10, 60]})}
    return abl, sens


def current_input_specs(sel):
    """03b kontrol deneyi: telafi KORUNUR, ana kola akım da verilir (akımsız omurga iddiasının nedensel testi)."""
    b = sel['Önerilen (RC-Hibrit)']
    def v(**kw):
        s = copy.deepcopy(b)
        for k, x in kw.items(): set_path(s, k, x)
        return s
    return {'RC + akım girişi': v(**{'rc.use_I': True}),
            'RC: yalnız ohmik + akım girişi': v(**{'rc.taus': [], 'rc.use_I': True})}


# ── Ana model C (03.11.2026 kararı): yalnız ohmik kol + akım omurgaya da girer (IR-telafili hibrit) ──
C_NAME = 'Önerilen (IR-telafili hibrit)'
C_SRC = ('ablation_b', 'RC: yalnız ohmik + akım girişi')     # 03b'de eğitilen aynı model (aynı spec, aynı seed'ler)


def c_spec(sel):
    return current_input_specs(sel)['RC: yalnız ohmik + akım girişi']


def c_specs(sel):
    """03c: C tabanında ablasyon (5 seed) ve duyarlılık (3 seed) — her biri tek değişiklik."""
    b = c_spec(sel)
    def v(**kw):
        s = copy.deepcopy(b)
        for k, x in kw.items(): set_path(s, k, x)
        return s
    abl = {'C: R sıcaklıktan bağımsız': v(**{'rc.use_T': False}),
           'C: eşit boyut (h=64)': v(**{'params.hidden': 64, 'params.d_model': 64})}
    sens = {'C duyarlılık: ohmik yumuşatma yok': v(**{'rc.ohm_smooth': 1}),
            'C duyarlılık: ohmik yumuşatma 5 s': v(**{'rc.ohm_smooth': 5}),
            'C duyarlılık: W = 128': v(window=128)}
    return abl, sens


def register_c(exp, seeds, series=True):
    """03b'deki eğitilmiş C ağırlıklarını 'final_c' aşamasına ana model adıyla kopyalar (yeniden eğitim YOK),
    sonra seriler + test anı ablasyonlarıyla yeniden değerlendirir."""
    import shutil
    spec = c_spec(exp.load_selected())
    for seed in seeds:
        sck, sjs, _ = exp.paths(C_SRC[0], C_SRC[1], seed)
        dck, djs, _ = exp.paths('final_c', C_NAME, seed)
        if not djs.exists():
            if not (sck.exists() and sjs.exists()):
                raise FileNotFoundError(f'03b koşusu yok: {sjs} — önce 03b tamamlanmalı')
            shutil.copy(sck, dck)
            rec = json.load(open(sjs, encoding='utf-8')); rec.update({'name': C_NAME, 'stage': 'final_c'}); rec.pop('rows', None)
            json.dump(rec, open(djs, 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
        exp.run(C_NAME, spec, seed, 'final_c', series=series, ablations=RC_ABLATIONS_TEST)


class Experiment:
    def __init__(self, root, smoke=False, device=None, log=print):
        self.root = Path(root); self.smoke = smoke; self.log = log
        self.dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.cfg = dict(CFG)
        if smoke:
            self.cfg.update({'EPOCHS': 2, 'PATIENCE': 2, 'TUNE_EPOCHS': 1, 'TUNE_PATIENCE': 1, 'FINAL_SEEDS': [101, 202],
                             'ABL_SEEDS': [101], 'TEST_EVERY': 1, 'W': 64})
        R = self.root / 'results'
        self.p = {'data': self.root / 'data', 'ckpt': R / 'checkpoints', 'runs': R / 'runs', 'series': R / 'series',
                  'configs': R / 'configs', 'excel': R / 'excel', 'graphs': self.root / 'graphs'}
        for k, d in self.p.items():
            if k != 'data': d.mkdir(parents=True, exist_ok=True)
        if self.dev.type == 'cuda': torch.backends.cudnn.benchmark = True
        self._ds = {}

    # ── ortam ──
    def env_info(self):
        info = {'python': platform.python_version(), 'os': platform.platform(), 'torch': torch.__version__,
                'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                'cpu': platform.processor(), 'numpy': np.__version__, 'pandas': pd.__version__}
        json.dump(info, open(self.p['configs'] / 'ortam.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        return info

    # ── veri ──
    def load_data(self):
        self.role_dfs, self.te_dfs, self.file_table, self.pre_ms = load_all(self.p['data'], verbose=not self.smoke)
        tr = [d for (t, r), d in self.role_dfs.items() if r in TRAIN_ROLES]
        self.norm = Normalizer().fit(tr)
        json.dump(self.norm.stats, open(self.p['configs'] / 'normalizer.json', 'w'), indent=1)
        self.log(f"Veri: rol {len(self.role_dfs)} · test {len(self.te_dfs)} dosya · ön işleme {self.pre_ms:.4f} ms/örnek")

    def datasets(self, W):
        if W not in self._ds:
            one = lambda d: WindowData([self.norm.transform(d)], [d.soc.values], W, self.dev)
            tr = [d for (t, r), d in self.role_dfs.items() if r in TRAIN_ROLES]
            self._ds[W] = {'train': WindowData([self.norm.transform(d) for d in tr], [d.soc.values for d in tr], W, self.dev),
                           'val': {k: one(d) for k, d in self.role_dfs.items() if k[1] in VAL_ROLES},
                           'test': {k: one(d) for k, d in self.te_dfs.items()}}
        return self._ds[W]

    def test_sets_from(self, dfs, W, pad=False):
        return {k: WindowData([self.norm.transform(d)], [d.soc.values], W, self.dev, pad_start=pad) for k, d in dfs.items()}

    # ── koşu ──
    def spec_full(self, spec):
        s = copy.deepcopy(spec); s.setdefault('window', self.cfg['W'])
        if self.smoke:
            s['window'] = min(s['window'], self.cfg['W']); p = s['params']
            for k in ('hidden', 'd_model', 'conv_ch'):
                if k in p: p[k] = 16
            for k in ('gru_layers', 'trans_layers', 'layers'):
                if k in p: p[k] = 1
            if 'nhead' in p: p['nhead'] = 2
        return s

    def paths(self, stage, name, seed):
        safe = name.replace(' ', '_').replace('/', '-').replace(':', '').replace('→', '-').replace('(', '').replace(')', '').replace('=', '')
        for k in ('ckpt', 'runs', 'series'): (self.p[k] / stage).mkdir(parents=True, exist_ok=True)
        return (self.p['ckpt'] / stage / f'{safe}_s{seed}.pt', self.p['runs'] / stage / f'{safe}_s{seed}.json',
                self.p['series'] / stage / f'{safe}_s{seed}.parquet')

    def model_for(self, spec):
        return build(self.spec_full(spec), self.norm.stats).to(self.dev)

    def rc_log(self, model):
        rc = getattr(model, 'rc', None)
        if rc is None: return ''
        r0 = lambda T: float(torch.exp(rc.r0_a + rc.r0_b * (self.norm.t_norm(T) - 1))) * 1e3
        return f" | R0 25°C {r0(25):.0f} · −10°C {r0(-10):.0f} mΩ" + (f" | τ {'/'.join(f'{x:.1f}' for x in rc.taus().tolist())} s" if rc.nk else '')

    def run(self, name, spec, seed, stage, evaluate=True, series=False, ablations=(), tune=False):
        """Eğit (yoksa) + değerlendir + kaydet. Tamamlanmış koşu atlanır (JSON'dan okunur)."""
        ck, js, pq = self.paths(stage, name, seed)
        if js.exists():
            rec = json.load(open(js, encoding='utf-8'))
            if (not evaluate or 'rows' in rec) and (not series or table_exists(pq)):
                self.log(f'✔ {stage} | {name} | s{seed} tamamlanmış'); return rec
        sf = self.spec_full(spec); W = sf['window']; ds = self.datasets(W)
        engine.set_seed(seed); model = build(sf, self.norm.stats).to(self.dev)
        cfg = dict(self.cfg)
        if tune: cfg.update({'EPOCHS': cfg['TUNE_EPOCHS'], 'PATIENCE': cfg['TUNE_PATIENCE'], 'TEST_EVERY': 0})
        rec = {'name': name, 'stage': stage, 'seed': seed, 'spec': sf, 'cfg': {k: cfg[k] for k in
               ('BATCH', 'EPOCHS', 'PATIENCE', 'LR_PATIENCE', 'EMA_TAU')}, 'train_roles': TRAIN_ROLES, 'val_roles': VAL_ROLES,
               'n_params': sum(p.numel() for p in model.parameters() if p.requires_grad)}
        if js.exists() and ck.exists():
            old = json.load(open(js, encoding='utf-8')); rec.update({k: old[k] for k in ('train',) if k in old})
        if 'train' not in rec:
            self.log(f"\n{'=' * 70}\n{stage} | {name} | seed {seed} | W {W} | lr {sf['lr']} | parametre {rec['n_params']:,}\n{'=' * 70}")
            info = engine.fit(model, ds['train'], ds['val'], cfg, sf['lr'], ck, torch.Generator().manual_seed(seed),
                              test_sets=None if tune else ds['test'], log=self.log, log_extra=self.rc_log)
            rec['train'] = info
            json.dump(rec, open(js, 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
        model.load_state_dict(torch.load(ck, map_location=self.dev, weights_only=True))
        if model.rc is not None:
            rc = model.rc; rec['rc_params'] = {'r0_a': rc.r0_a.item(), 'r0_b': rc.r0_b.item(),
                                               'rk_a': rc.rk_a.tolist() if rc.nk else [], 'rk_b': rc.rk_b.tolist() if rc.nk else [],
                                               'tau': rc.taus().tolist()}
        if evaluate:
            keep = ('eta', 'eta0', 'etak') if model.rc is not None else ()
            abl = [a for a in ablations if model.rc is not None and (a != 'rc3' or model.rc.nk >= 3) and (a not in ('rc1', 'rc2') or model.rc.nk >= int(a[-1]))]
            rows, ser = engine.evaluate(model, ds['test'], self.te_dfs, cfg, abl, keep, series)
            rec['rows'] = rows; rec['val_rmse_final'] = engine.mean_file_rmse(model, ds['val'], cfg['PRED_BS'])
            if ser is not None: save_table(ser, pq)
            m = [r['rmse'] for r in rows if r['temp'] != '-20degC']; m20 = [r['rmse'] for r in rows if r['temp'] == '-20degC']
            self.log(f"→ {name} s{seed}: val {rec['val_rmse_final']*100:.3f}% | test (−20 hariç) {np.mean(m)*100:.3f}% | −20°C {np.mean(m20)*100:.2f}%")
        json.dump(rec, open(js, 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
        return rec

    def load_model(self, stage, name, seed):
        ck, js, _ = self.paths(stage, name, seed); rec = json.load(open(js, encoding='utf-8'))
        m = build(rec['spec'], self.norm.stats).to(self.dev); m.load_state_dict(torch.load(ck, map_location=self.dev, weights_only=True)); m.eval()
        return m, rec

    def collect(self, stage):
        rows = []
        for f in sorted((self.p['runs'] / stage).glob('*.json')):
            rec = json.load(open(f, encoding='utf-8'))
            for r in rec.get('rows', []):
                rows.append({'model': rec['name'], 'seed': rec['seed'], 'val_rmse': rec.get('val_rmse_final', rec.get('train', {}).get('best_val')),
                             'best_epoch': rec.get('train', {}).get('best_epoch'), **r})
        return pd.DataFrame(rows)

    def records(self, stage):
        return [json.load(open(f, encoding='utf-8')) for f in sorted((self.p['runs'] / stage).glob('*.json'))]

    # ── seçilen hiperparametreler ──
    def save_selected(self, sel):
        json.dump(sel, open(self.p['configs'] / 'selected.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    def load_selected(self):
        f = self.p['configs'] / 'selected.json'
        if f.exists(): return json.load(open(f, encoding='utf-8'))
        self.log('⚠️ selected.json yok → varsayılan hiperparametreler kullanılıyor (önce 01_ayarlama çalıştırılmalı)')
        return copy.deepcopy(BASE_SPECS)
