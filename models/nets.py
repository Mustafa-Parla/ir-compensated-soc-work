"""
Modeller (V26). Hepsi aynı girdi tensörünü alır: X (B, W, NX) — bkz. data_utils.data.

Aileler (spec['family']):
  'rc'          : ÖNERİLEN — fiziksel çok ölçekli RC detector + hibrit GRU → Transformer.
                  η(t) = R0(T)·Ī_s(t) + Σ_k R_k(T)·(h_k ∗ I)(t),  h_k(s) = (1−a_k)a_k^s, a_k = e^(−1/τ_k), τ_k öğrenilir.
                  log R = a + b·(T_n − 1) (soğukta büyür). Ana kol akım görmez: [V, T, V_ort, V − η].
                  spec['rc'] seçenekleri: taus (başlangıç τ'lar; boş → yalnız ohmik), ohm_smooth (s), use_T, det (serbest CNN
                  modülasyonu, sıcaklıksız), mod_max, use_I (kontrol deneyi: ana kola akım da verilir → [V, I, T, V_ort, V − η]).
  'hyb'         : hibrit GRU → Transformer, girdi [V, I, T, V_ort] (spec['noI'] = True → akımsız)
  'lstm' / 'gru': yığılmış LSTM / GRU → son adım → MLP
  'cnn_lstm'    : nedensel 1D-CNN katmanları → LSTM → MLP
  'transformer' : doğrusal gömme + konum gömmesi + Transformer kodlayıcı → son adım → MLP
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils.data import NE, ND, IV, II

FEAT4 = [0, 1, 2, 3]          # v10, i10, temp, v10_avg
FEAT_NOI = [0, 2, 3]          # v10, temp, v10_avg
T_IDX = 2


def head(d, dropout):
    return nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))


class HybBackbone(nn.Module):
    def __init__(self, n_in, W, hidden=128, gru_layers=3, d_model=128, nhead=4, trans_layers=3, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, gru_layers, batch_first=True, dropout=dropout if gru_layers > 1 else 0)
        self.proj = nn.Linear(hidden, d_model)
        self.pos = nn.Embedding(W, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout, batch_first=True)
        self.trans = nn.TransformerEncoder(layer, trans_layers, enable_nested_tensor=False)
        self.head = head(d_model, dropout)

    def forward(self, x):
        g, _ = self.gru(x)
        tok = self.proj(g)
        h = self.trans(tok + self.pos(torch.arange(tok.shape[1], device=tok.device)))
        return self.head(h[:, -1]).squeeze(-1)


class RNNBackbone(nn.Module):
    def __init__(self, n_in, W, cell='lstm', hidden=128, layers=2, dropout=0.1):
        super().__init__()
        R = nn.LSTM if cell == 'lstm' else nn.GRU
        self.rnn = R(n_in, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0)
        self.head = head(hidden, dropout)

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.head(h[:, -1]).squeeze(-1)


class CNNLSTMBackbone(nn.Module):
    def __init__(self, n_in, W, conv_ch=64, kernel=3, conv_layers=2, hidden=128, layers=2, dropout=0.1):
        super().__init__()
        self.k = kernel
        self.convs = nn.ModuleList([nn.Conv1d(n_in if i == 0 else conv_ch, conv_ch, kernel) for i in range(conv_layers)])
        self.rnn = nn.LSTM(conv_ch, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0)
        self.head = head(hidden, dropout)

    def forward(self, x):
        h = x.transpose(1, 2)
        for c in self.convs:
            h = F.relu(c(F.pad(h, (self.k - 1, 0))))                 # nedensel dolgu
        h, _ = self.rnn(h.transpose(1, 2))
        return self.head(h[:, -1]).squeeze(-1)


class TransformerBackbone(nn.Module):
    def __init__(self, n_in, W, d_model=128, nhead=4, layers=3, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(n_in, d_model)
        self.pos = nn.Embedding(W, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = head(d_model, dropout)
        self.register_buffer('mask', torch.triu(torch.full((W, W), float('-inf')), 1), persistent=False)

    def forward(self, x):
        T = x.shape[1]
        h = self.enc(self.inp(x) + self.pos(torch.arange(T, device=x.device)), mask=self.mask[:T, :T])
        return self.head(h[:, -1]).squeeze(-1)


class RCDetector(nn.Module):
    """Fiziksel çok ölçekli RC detector: η = R0(T)·Ī + Σ R_k(T)·(h_k ∗ I). Opsiyonel olay modülasyonu (det)."""
    def __init__(self, taus=(3., 7., 12.), ohm_smooth=3, use_T=True, det=False, mod_max=0.3, ch=16,
                 kernels=(3, 7, 12), det_idx=(0, 1, 2, 4), tau_max=60.):
        super().__init__()
        self.nk = len(taus)
        self.ohm_smooth, self.use_T, self.det, self.mod_max, self.tau_max = int(ohm_smooth), bool(use_T), bool(det), float(mod_max), float(tau_max)
        self.kernels, self.det_idx = tuple(kernels), list(det_idx)
        self.log_tau = nn.Parameter(torch.log(torch.tensor(taus, dtype=torch.float32))) if self.nk else None
        self.r0_a = nn.Parameter(torch.tensor(float(np.log(0.025))))
        self.r0_b = nn.Parameter(torch.tensor(-1.2), requires_grad=self.use_T)
        if self.nk:
            self.rk_a = nn.Parameter(torch.full((self.nk,), float(np.log(0.008))))
            self.rk_b = nn.Parameter(torch.full((self.nk,), -1.2), requires_grad=self.use_T)
        if not self.use_T:
            with torch.no_grad():
                self.r0_b.zero_()
                if self.nk: self.rk_b.zero_()
        if self.det:
            self.convs = nn.ModuleList([nn.Conv1d(len(self.det_idx), ch, k) for k in self.kernels])
            self.heads = nn.ModuleList([nn.Sequential(nn.Linear(ch, 16), nn.Tanh(), nn.Linear(16, 1)) for _ in range(self.nk)])
            self.head0 = nn.Sequential(nn.Linear(ch * len(self.kernels), 16), nn.Tanh(), nn.Linear(16, 1))
            for h in list(self.heads) + [self.head0]: nn.init.zeros_(h[-1].weight); nn.init.zeros_(h[-1].bias)

    def taus(self):
        return torch.exp(self.log_tau).clamp(1.0, self.tau_max) if self.nk else torch.zeros(0)

    @staticmethod
    def box(I, s):
        B, T = I.shape
        x = torch.cat([I[:, :1].expand(B, s - 1), I], 1).unsqueeze(1)
        return F.conv1d(x, torch.full((1, 1, s), 1.0 / s, device=I.device, dtype=I.dtype)).squeeze(1)

    @staticmethod
    def iir(I, tau):
        """(h ∗ I): pencere başında kararlı durum (I_0) varsayımı. Akış (streaming) uygulamada eşdeğer özyineleme:
        y_t = a·y_(t−1) + (1−a)·I_t (adım başına 2 çarpma)."""
        B, T = I.shape
        a = torch.exp(-1.0 / tau)
        s = torch.arange(T, device=I.device, dtype=I.dtype)
        w = ((1 - a) * a ** s).flip(0).view(1, 1, T)
        x = torch.cat([I[:, :1].expand(B, T - 1), I], 1).unsqueeze(1)
        return F.conv1d(x, w).squeeze(1)

    def R(self, t_n):
        """Sıcaklığa bağlı dirençler (Ω): r0 (…), rk (…, nk)."""
        r0 = torch.exp(self.r0_a + self.r0_b * (t_n - 1))
        rk = torch.exp(self.rk_a + self.rk_b * (t_n.unsqueeze(-1) - 1)) if self.nk else None
        return r0, rk

    def forward(self, I, t_n, x_det=None, ablate=None, t_mean=0.5):
        if ablate == 'tconst': t_n = torch.full_like(t_n, t_mean)
        r0, rk = self.R(t_n)
        m = None
        if self.det and x_det is not None:
            xt = x_det[..., self.det_idx].transpose(1, 2); hs = []
            for c, k in zip(self.convs, self.kernels):
                hs.append(torch.tanh(c(F.pad(xt, (k - 1, 0)))).transpose(1, 2))
            mk = [self.mod_max * torch.tanh(h(f).squeeze(-1)) for h, f in zip(self.heads, hs)]
            m0 = self.mod_max * torch.tanh(self.head0(torch.cat(hs, -1)).squeeze(-1))
            m = torch.stack([m0] + mk, -1)
            if ablate == 'det0': m = torch.zeros_like(m)
            r0 = r0 * torch.exp(m[..., 0])
            if self.nk: rk = rk * torch.exp(m[..., 1:])
        eta0 = r0 * (self.box(I, self.ohm_smooth) if self.ohm_smooth > 1 else I)
        if ablate == 'ohm0': eta0 = torch.zeros_like(eta0)
        if self.nk:
            taus = self.taus()
            etak = rk * torch.stack([self.iir(I, taus[j]) for j in range(self.nk)], -1)
            for j in range(self.nk):
                if ablate == f'rc{j + 1}': etak = etak.clone(); etak[..., j] = 0.
            eta = eta0 + etak.sum(-1)
        else:
            etak = torch.zeros(*I.shape, 0, device=I.device); eta = eta0
        if ablate == 'eta0': eta = torch.zeros_like(eta)
        return {'eta': eta, 'eta0': eta0, 'etak': etak, 'r0': r0, 'rk': rk, 'm': m}

    def macs(self, T):
        """Uygulanan biçimdeki çarpma-toplama (pencere başına). Akış biçimi: adım başına 2·nk + ohm_smooth."""
        return self.nk * T * T + T * max(self.ohm_smooth, 1) + (self.det and sum(len(self.det_idx) * 16 * k * T for k in self.kernels) or 0)


class SOCNet(nn.Module):
    """spec: {'family', 'params' (omurga), 'rc' (rc ailesi için), 'noI', 'window'}"""
    def __init__(self, spec, norm_stats):
        super().__init__()
        fam, p, W = spec['family'], dict(spec.get('params', {})), spec['window']
        self.family, self.W = fam, W
        self.v_min, self.v_max = norm_stats['v10']['min'], norm_stats['v10']['max']
        self.t_mean = norm_stats['t_mean_norm']
        self.rc = None
        if fam == 'rc':
            r = dict(spec.get('rc', {}))
            self.rc = RCDetector(taus=tuple(float(x) for x in r.get('taus', (3, 7, 12))), ohm_smooth=r.get('ohm_smooth', 3),
                                 use_T=r.get('use_T', True), det=r.get('det', False), mod_max=r.get('mod_max', 0.3))
            self.cols = FEAT4 if r.get('use_I', False) else FEAT_NOI; n_in = len(self.cols) + 1
        else:
            self.cols = FEAT_NOI if spec.get('noI') else FEAT4; n_in = len(self.cols)
        if fam in ('rc', 'hyb'):
            self.backbone = HybBackbone(n_in, W, **p)
        elif fam in ('lstm', 'gru'):
            self.backbone = RNNBackbone(n_in, W, cell=fam, **p)
        elif fam == 'cnn_lstm':
            self.backbone = CNNLSTMBackbone(n_in, W, **p)
        elif fam == 'transformer':
            self.backbone = TransformerBackbone(n_in, W, **p)
        else:
            raise ValueError(fam)

    def forward(self, X, ablate=None):
        x = X[..., self.cols]
        o = {}
        if self.rc is not None:
            o = self.rc(X[..., II], X[..., T_IDX], X[..., NE:NE + ND], ablate, self.t_mean)
            ocv = (X[..., IV] - o['eta'] - self.v_min) / (self.v_max - self.v_min)
            x = torch.cat([x, ocv.unsqueeze(-1)], -1)
        return self.backbone(x), o


def build(spec, norm_stats):
    return SOCNet(spec, norm_stats)
