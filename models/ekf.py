"""
Klasik kıyas: 2RC eşdeğer devre modeli (ECM) + Genişletilmiş Kalman Filtresi (EKF).

Tanımlama (her eğitim sıcaklığı için, YALNIZ eğitim dosyaları C1–C4; etiket SoC kullanılır):
    V_t = OCV(SoC_t) + R0·I_t + R1·f1_t + R2·f2_t,   f_k,t = a_k f_k,t−1 + (1−a_k) I_t,  a_k = e^(−Δt/τ_k)
    τ1, τ2 ızgarada aranır; her (τ1, τ2) için problem doğrusal → sınırlı en küçük kareler (R ≥ 0), OCV = 8. derece polinom.
Sıcaklık: log R = a + b·T (4 eğitim sıcaklığından), OCV katsayıları ve τ'lar sıcaklıkta doğrusal ara değerleme
(aralık dışı → en yakın uç; −20°C için −10°C OCV'si ve R(T) ekstrapolasyonu — DL modelleriyle aynı "görülmemiş sıcaklık" koşulu).
EKF durumu x = [SoC, V1, V2]; SoC_t+1 = SoC_t + I·Δt / (3600·Q), Q = 2.9 Ah (etiketle aynı).
"""
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

DEG = 8
TAU1_GRID = [1, 2, 3, 5, 8, 12]
TAU2_GRID = [15, 25, 40, 60, 100, 160, 250]


def _basis(soc):
    x = 2 * np.clip(soc, 0, 1) - 1
    return np.polynomial.chebyshev.chebvander(x, DEG)


def _filt(I, dt, tau):
    """f_t = a f_(t−1) + (1−a) I_t, a = e^(−1/τ) (1 Hz; dt yalnız imza uyumu için)."""
    from scipy.signal import lfilter
    a = np.exp(-1.0 / tau)
    return lfilter([1 - a], [1, -a], I, zi=[a * I[0]])[0]


def identify(dfs):
    """Bir sıcaklıktaki dosyalardan ECM parametreleri: dict(ocv_coef, R0, R1, R2, tau1, tau2, rmse_V)."""
    best = None
    soc = np.concatenate([d.soc.values for d in dfs]); V = np.concatenate([d.v10.values for d in dfs])
    I = np.concatenate([d.i10.values for d in dfs]); B = _basis(soc)
    dts = [np.diff(d.time.values, prepend=d.time.values[0] - 1).clip(0.5, 5) for d in dfs]
    F2 = {t2: np.concatenate([_filt(d.i10.values, dt, t2) for d, dt in zip(dfs, dts)]) for t2 in TAU2_GRID}
    for t1 in TAU1_GRID:
        f1 = np.concatenate([_filt(d.i10.values, dt, t1) for d, dt in zip(dfs, dts)])
        for t2 in TAU2_GRID:
            f2 = F2[t2]
            A = np.column_stack([B, I, f1, f2])
            lb = np.r_[np.full(B.shape[1], -np.inf), 0, 0, 0]
            res = lsq_linear(A, V, bounds=(lb, np.full(A.shape[1], np.inf)), method='bvls')
            rm = float(np.sqrt(np.mean((A @ res.x - V) ** 2)))
            if best is None or rm < best['rmse_V']:
                c = res.x
                best = {'ocv_coef': c[:B.shape[1]].tolist(), 'R0': c[-3], 'R1': c[-2], 'R2': c[-1], 'tau1': t1, 'tau2': t2, 'rmse_V': rm}
    return best


def fit_temperature_model(role_dfs, train_roles, temps_c):
    """temps_c: {'25degC': 25.0, ...}. Dönüş: sıcaklık bazında parametreler + R(T) üstel uyumları."""
    per = {}
    for t, Tc in temps_c.items():
        dfs = [d for (tt, r), d in role_dfs.items() if tt == t and r in train_roles]
        if dfs:
            p = identify(dfs); p['T'] = float(np.mean([d.temp.mean() for d in dfs])); per[t] = p
    Ts = np.array([p['T'] for p in per.values()])
    fits = {}
    for k in ['R0', 'R1', 'R2']:
        y = np.log(np.array([max(p[k], 1e-5) for p in per.values()])); b, a = np.polyfit(Ts, y, 1); fits[k] = (a, b)
    return {'per_temp': per, 'R_fit': fits}


class ECMParams:
    def __init__(self, tm):
        self.tm = tm
        per = sorted(tm['per_temp'].values(), key=lambda p: p['T'])
        self.Ts = np.array([p['T'] for p in per]); self.C = np.array([p['ocv_coef'] for p in per])
        self.tau1 = np.array([p['tau1'] for p in per], float); self.tau2 = np.array([p['tau2'] for p in per], float)

    def at(self, T_file):
        coef = np.array([np.interp(T_file, self.Ts, self.C[:, j]) for j in range(self.C.shape[1])])
        return coef, float(np.interp(T_file, self.Ts, self.tau1)), float(np.interp(T_file, self.Ts, self.tau2))

    def R(self, k, T):
        a, b = self.tm['R_fit'][k]; return np.exp(a + b * T)


def ocv(coef, s):
    return np.polynomial.chebyshev.chebval(2 * np.clip(s, 0, 1) - 1, coef)


def docv(coef, s):
    return 2 * np.polynomial.chebyshev.chebval(2 * np.clip(s, 0, 1) - 1, np.polynomial.chebyshev.chebder(coef))


def run_ekf(df, P, soc0, Q_ah=2.9, q=(1e-5, 1e-4, 1e-4), r_v=0.01, p0=(0.05, 0.01, 0.01)):
    """df: 1 Hz tablo (v10, i10, temp, time). Dönüş: SoC kestirimi (len(df))."""
    coef, t1, t2 = P.at(float(df.temp.median()))
    V, I, T = df.v10.values, df.i10.values, df.temp.values
    dt = np.diff(df.time.values, prepend=df.time.values[0] - 1).clip(0.5, 5)
    R0, R1, R2 = P.R('R0', T), P.R('R1', T), P.R('R2', T)
    x = np.array([soc0, 0.0, 0.0]); Pm = np.diag(np.square(p0)); Qm = np.diag(np.square(q)); out = np.empty(len(df))
    for k in range(len(df)):
        a1, a2 = np.exp(-dt[k] / t1), np.exp(-dt[k] / t2)
        F = np.diag([1.0, a1, a2])
        x = np.array([x[0] + I[k] * dt[k] / (3600 * Q_ah), a1 * x[1] + R1[k] * (1 - a1) * I[k], a2 * x[2] + R2[k] * (1 - a2) * I[k]])
        Pm = F @ Pm @ F.T + Qm
        H = np.array([docv(coef, x[0]), 1.0, 1.0])
        yk = V[k] - (ocv(coef, x[0]) + R0[k] * I[k] + x[1] + x[2])
        S = H @ Pm @ H + r_v ** 2; K = Pm @ H / S
        x = x + K * yk; x[0] = np.clip(x[0], 0, 1.05); Pm = (np.eye(3) - np.outer(K, H)) @ Pm
        out[k] = x[0]
    return out
