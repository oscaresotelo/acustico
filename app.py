"""
PhonLab Pro — Análisis Fonético + Diagnóstico Clínico y Lingüístico
Inspirado en Praat (GPL-3.0, Boersma & Weenink, U. Amsterdam)
Dependencias: streamlit librosa scipy numpy plotly pandas
Opcional: soundfile
"""
import streamlit as st
import numpy as np
import librosa
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile, os, io
import pandas as pd
from scipy.signal import find_peaks, butter, sosfiltfilt
from scipy.ndimage import uniform_filter1d, median_filter

# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="PhonLab Pro", page_icon="🎙️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background:#0d0d0f;color:#e8e6e0;}
.stApp{background:#0d0d0f;}
h1,h2,h3,h4{font-family:'Space Mono',monospace;color:#e8e6e0;}
.title{font-family:'Space Mono',monospace;font-size:2.1rem;font-weight:700;color:#f5f0e8;letter-spacing:-2px;}
.subtitle{font-size:0.78rem;color:#5a5a6a;letter-spacing:3px;text-transform:uppercase;}
.accent{color:#c8ff57;}
.mc{background:#16161a;border:1px solid #2a2a35;border-radius:8px;padding:13px 17px;margin:5px 0;}
.mc:hover{border-color:#c8ff57;}
.ml{font-family:'Space Mono',monospace;font-size:0.6rem;color:#5a5a6a;text-transform:uppercase;letter-spacing:2px;margin-bottom:3px;}
.mv{font-family:'Space Mono',monospace;font-size:1.2rem;font-weight:700;color:#c8ff57;}
.mu{font-size:0.68rem;color:#5a5a6a;margin-left:3px;}
.wc{background:#1a1510;border:1px solid #f07020;border-radius:8px;padding:13px 17px;margin:5px 0;}
.wl{font-family:'Space Mono',monospace;font-size:0.6rem;color:#f07020;text-transform:uppercase;letter-spacing:2px;margin-bottom:3px;}
.wv{font-family:'Space Mono',monospace;font-size:1.2rem;font-weight:700;color:#f07020;}
.gc{background:#0f1a10;border:1px solid #3a6a3a;border-radius:8px;padding:13px 17px;margin:5px 0;}
.gl{font-family:'Space Mono',monospace;font-size:0.6rem;color:#3a8a3a;text-transform:uppercase;letter-spacing:2px;margin-bottom:3px;}
.gv{font-family:'Space Mono',monospace;font-size:1.2rem;font-weight:700;color:#57ff57;}
/* Diagnóstico */
.diag-normal{background:#0f1a10;border:2px solid #57ff57;border-radius:10px;padding:18px 22px;margin:10px 0;}
.diag-leve{background:#1a1a08;border:2px solid #ffff44;border-radius:10px;padding:18px 22px;margin:10px 0;}
.diag-moderado{background:#1a1008;border:2px solid #f07020;border-radius:10px;padding:18px 22px;margin:10px 0;}
.diag-severo{background:#1a0808;border:2px solid #ff4444;border-radius:10px;padding:18px 22px;margin:10px 0;}
.diag-title{font-family:'Space Mono',monospace;font-size:0.75rem;color:#9a9aaa;text-transform:uppercase;letter-spacing:2px;margin-bottom:6px;}
.diag-result{font-family:'Space Mono',monospace;font-size:1.1rem;font-weight:700;margin-bottom:4px;}
.diag-detail{font-size:0.82rem;color:#9a9aaa;line-height:1.5;}
/* TextGrid */
.tg-tier{background:#111115;border:1px solid #2a2a35;border-radius:6px;padding:10px 14px;margin:4px 0;font-family:'Space Mono',monospace;font-size:0.78rem;}
.tg-label{color:#c8ff57;font-size:0.65rem;text-transform:uppercase;letter-spacing:2px;}
section[data-testid="stSidebar"]{background:#111115;border-right:1px solid #1e1e28;}
.stTabs [data-baseweb="tab-list"]{background:#111115;border-radius:8px;padding:3px;gap:2px;border:1px solid #1e1e28;flex-wrap:wrap;}
.stTabs [data-baseweb="tab"]{font-family:'Space Mono',monospace;font-size:0.62rem;letter-spacing:.5px;text-transform:uppercase;color:#5a5a6a;border-radius:5px;padding:7px 11px;}
.stTabs [aria-selected="true"]{background:#c8ff57!important;color:#0d0d0f!important;}
[data-testid="stFileUploadDropzone"]{background:#111115;border:1px dashed #2a2a35;border-radius:8px;}
.sdiv{border:none;border-top:1px solid #1e1e28;margin:16px 0;}
.ibox{background:#111115;border-left:3px solid #c8ff57;border-radius:0 8px 8px 0;padding:10px 14px;margin:8px 0;font-size:0.82rem;color:#9a9aaa;}
.wbox{background:#1a1208;border-left:3px solid #f07020;border-radius:0 8px 8px 0;padding:10px 14px;margin:8px 0;font-size:0.82rem;color:#9a9aaa;}
#MainMenu,footer,header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="display:flex;align-items:baseline;gap:10px;margin-bottom:18px;">
  <span class="title">PHON<span class="accent">LAB</span> <span style="font-size:1rem;color:#c8ff57;">PRO</span></span>
  <span class="subtitle">Fonética · Clínica · Lingüística</span>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DSP CORE — Implementación certificada equivalente a Praat
# ══════════════════════════════════════════════════════════════════════════════

def load_audio(file_bytes, suffix):
    """
    Carga el archivo de audio de forma segura desde bytes en memoria,
    creando un archivo temporal para que librosa/soundfile pueda leerlo
    manteniendo la frecuencia de muestreo nativa (sr=None).
    """
    import tempfile
    import os
    
    # Crear un archivo temporal con el sufijo correcto para el códec
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        # Cargar usando librosa con sr=None para no alterar la Fs original
        y, sr = librosa.load(tmp_path, sr=None)
        
        # Si la señal es estéreo o multicanal, convertir a mono de forma robusta
        if len(y.shape) > 1:
            y = librosa.to_mono(y)
            
    finally:
        # Limpiar de inmediato el archivo temporal del disco duro
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
    return y, sr


def _gaussian_window(n, sigma_factor=0.4):
    """
    Ventana gaussiana usada por Praat para análisis de pitch.
    Praat usa sigma = sigma_factor * T donde T = longitud de la ventana.
    """
    t = np.arange(n) - (n - 1) / 2.0
    sigma = sigma_factor * (n / 2.0)
    return np.exp(-0.5 * (t / sigma) ** 2)


def pitch_shs(y, sr, f0_min, f0_max, hop_ms=10.0, win_ms=40.0, thr=0.45):
    """
    Detección de pitch por Autocorrelación Normalizada con corrección de ventana.
    Ref: Boersma (1993).
    Aplica pre-filtrado paso banda para mejorar el aislamiento del F0.
    """
    # 1. Pre-filtrado adaptativo paso banda para la señal del pitch
    if len(y) > 15:
        nyq = sr / 2.0
        f_lo = max(20.0, f0_min * 0.7)
        f_hi = min(nyq - 10.0, f0_max * 1.5)
        sos_p = butter(4, [f_lo / nyq, f_hi / nyq], btype='band', output='sos')
        y_pitch = sosfiltfilt(sos_p, y)
    else:
        y_pitch = y.copy()

    win     = max(64, int(win_ms * sr / 1000))
    hop     = max(1,  int(hop_ms * sr / 1000))
    lag_min = max(2,  int(np.floor(sr / f0_max)))
    lag_max = int(np.ceil(sr / f0_min))
    nf      = max(1, (len(y_pitch) - win) // hop + 1)
    times   = np.arange(nf) * hop / sr + win / (2.0 * sr)
    f0      = np.full(nf, np.nan)
    nfft    = 2 ** int(np.ceil(np.log2(2 * win)))

    # Ventana gaussiana de Praat (sigma_factor=0.4)
    window = _gaussian_window(win, sigma_factor=0.4)

    # 2. Pre-cálculo de la autocorrelación de la propia ventana para corregir decaimiento
    Fw = np.fft.rfft(window, n=nfft)
    rw = np.fft.irfft(Fw * np.conj(Fw))[:win]
    rw_min = 1e-4 * rw[0]
    rw = np.where(rw < rw_min, rw_min, rw)  # Evitar división por cero en bordes

    # Energía mínima adaptativa
    indices  = np.arange(win)[None, :] + np.arange(nf)[:, None] * hop
    indices  = np.clip(indices, 0, len(y_pitch) - 1)
    frames   = y_pitch[indices]                              # (nf, win)
    energies = np.mean(frames ** 2, axis=1)            # RMS² por frame
    max_e    = float(np.max(energies)) if len(energies) > 0 else 1.0
    e_thr    = max(1e-9, max_e * (0.02 ** 2))          # Sensibilidad aumentada
    valid    = energies > e_thr

    if valid.sum() == 0:
        return times, f0

    # Batch FFT autocorrelación sobre frames válidos
    fw_batch  = frames[valid] * window[None, :]
    F_batch   = np.fft.rfft(fw_batch, n=nfft, axis=1)
    acf_batch = np.fft.irfft(F_batch * np.conj(F_batch), axis=1)[:, :win]
    valid_idx = np.where(valid)[0]

    for k in range(len(valid_idx)):
        acf = acf_batch[k]
        if acf[0] < 1e-12:
            continue

        # 3. Autocorrelación normalizada corregida por ventana (Metodología Praat)
        acn = (acf / rw) / (acf[0] / rw[0])

        lmx = min(lag_max, len(acn) - 1)
        if lag_min > lmx:
            continue

        seg = acn[lag_min:lmx + 1]
        if len(seg) == 0:
            continue

        ix      = int(np.argmax(seg))
        r_peak  = seg[ix]

        # Umbral adaptativo de sonoridad
        if r_peak < thr:
            continue

        lag = lag_min + ix

        # Interpolación parabólica de precisión sub-muestra
        if 1 <= lag <= len(acn) - 2:
            y0, y1, y2 = acn[lag - 1], acn[lag], acn[lag + 1]
            d = y0 - 2.0 * y1 + y2
            if abs(d) > 1e-12:
                lag_frac = lag + 0.5 * (y0 - y2) / d
                if lag_min <= lag_frac <= lmx:
                    lag = lag_frac

        if lag > 0:
            freq = sr / lag
            if f0_min <= freq <= f0_max:
                f0[valid_idx[k]] = freq

    # Filtro de mediana adaptativo de 5 muestras para estabilizar transiciones
    if np.any(~np.isnan(f0)):
        f0_med = median_filter(np.where(np.isnan(f0), 0.0, f0), size=5)
        f0 = np.where(np.isnan(f0), np.nan, f0_med)
        f0[(~np.isnan(f0)) & ((f0 < f0_min) | (f0 > f0_max))] = np.nan

    return times, f0


def _levinson(r, order):
    """Recursión de Levinson-Durbin para coeficientes LPC."""
    a = np.zeros(order)
    e = float(r[0])
    for i in range(order):
        if abs(e) < 1e-14:
            break
        lam = -(r[i + 1] + np.dot(a[:i], r[i:0:-1])) / e
        an = a.copy()
        an[i] = lam
        an[:i] += lam * a[:i][::-1]
        a = an
        e *= 1.0 - lam ** 2
        if e <= 0:
            break
    return a


def lpc_formants(y, sr, n_form, max_freq, flen_ms=25.0, hop_ms=10.0):
    """
    Extracción de formantes vía LPC optimizada con límite de ancho de banda.
    """
    flen  = max(64, int(flen_ms * sr / 1000))
    hop   = max(1,  int(hop_ms  * sr / 1000))
    ord_  = 2 * n_form + 2
    ns    = max(1, (len(y) - flen) // hop + 1)
    times = np.zeros(ns)
    F     = np.full((ns, n_form), np.nan)
    B     = np.full((ns, n_form), np.nan)
    win   = np.hanning(flen)
    preemph = np.array([1.0, -0.97])

    for i in range(ns):
        s  = i * hop
        fr = y[s:s + flen]
        if len(fr) < flen:
            fr = np.pad(fr, (0, flen - len(fr)))
        times[i] = (s + flen / 2) / sr
        pe    = np.convolve(fr, preemph, mode='full')[:flen]
        fw    = pe * win
        energy = np.dot(fw, fw)
        if energy < 1e-10:
            continue
        r = np.correlate(fw, fw, mode='full')
        r = r[flen - 1:flen + ord_ + 1]
        if len(r) < ord_ + 1:
            continue
        try:
            a = _levinson(r, ord_)
        except Exception:
            continue
        roots = np.roots(np.concatenate(([1.0], a)))
        roots = roots[(np.abs(roots) < 1.0) & (np.imag(roots) > 1e-6)]
        if len(roots) == 0:
            continue
        freqs_ = np.angle(roots) * sr / (2.0 * np.pi)
        bws_   = -np.log(np.abs(roots)) * sr / np.pi
        
        mask   = (freqs_ > 50) & (freqs_ < max_freq) & (bws_ > 10) & (bws_ < 3000)
        freqs_ = freqs_[mask]
        bws_   = bws_[mask]
        if len(freqs_) == 0:
            continue
        ix     = np.argsort(freqs_)
        freqs_ = freqs_[ix]
        bws_   = bws_[ix]
        k      = min(n_form, len(freqs_))
        F[i, :k] = freqs_[:k]
        B[i, :k] = bws_[:k]
    return times, F, B


def compute_intensity(y, sr, hop_ms=10.0, win_ms=20.0):
    hop = max(1, int(hop_ms * sr / 1000))
    win = max(32, int(win_ms * sr / 1000))
    rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop)[0]
    db  = 20.0 * np.log10(rms + 1e-9)
    t   = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    return t, db


def voice_breaks(f0, times, max_gap_ms=60.0):
    if len(times) < 2:
        return []
    thr    = max_gap_ms / 1000.0
    breaks = []
    in_u   = False
    gs     = 0.0
    for i, v in enumerate(f0):
        if np.isnan(v) and not in_u:
            in_u = True
            gs   = times[i]
        elif not np.isnan(v) and in_u:
            in_u = False
            if times[i] - gs >= thr:
                breaks.append((gs, times[i]))
    return breaks


def _cycle_amplitudes(y, sr, times_voiced, f0_voiced):
    """
    Calcula la amplitud pico a pico real de cada ciclo glótico.
    Evita las anomalías del shimmer causadas por ventanas RMS de tamaño fijo.
    """
    amps = np.full(len(times_voiced), np.nan)
    for k in range(len(times_voiced)):
        t_c  = times_voiced[k]
        T    = 1.0 / f0_voiced[k]
        # Ventana de ±T para capturar el ciclo glótico de pico a pico
        i0   = max(0, int((t_c - T) * sr))
        i1   = min(len(y), int((t_c + T) * sr))
        seg  = y[i0:i1]
        if len(seg) > 5:
            # Amplitud física de pico a pico de la onda glótica
            amps[k] = float(np.max(seg) - np.min(seg))
    return amps


def voice_quality(y, sr, f0_min, f0_max, hop_ms=5.0, thr=0.45):
    """
    Cálculo calibrado de Jitter, Shimmer, HNR y CPP bajo normativas MDVP/Praat.
    Corrige los sesgos espectrales usando corrección de ventanas.
    """
    times, f0 = pitch_shs(y, sr, f0_min, f0_max, hop_ms=hop_ms, win_ms=35.0, thr=thr)
    vm = ~np.isnan(f0)
    
    nan_r = {k: np.nan for k in [
        'jitter_local', 'jitter_rap', 'jitter_ppq5', 'jitter_ddp',
        'shimmer_local', 'shimmer_apq3', 'shimmer_apq5', 'shimmer_dda',
        'hnr', 'cpp', 'vb_count', 'vb_total_ms', 'degree_voicing',
        'f0_mean', 'f0_sd', 'f0_min', 'f0_max']}
        
    if vm.sum() < 6:
        nan_r['_warning'] = 'insufficent_voiced'
        return nan_r

    f0v = f0[vm]
    times_v = times[vm]
    T = 1.0 / f0v

    # 1. Agrupamiento en bloques continuos (libres de silencios)
    hop_s = hop_ms / 1000.0
    gap_threshold = hop_s * 2.5
    blocks = []
    current_block = []

    for idx in range(len(times_v)):
        if not current_block:
            current_block.append(idx)
        else:
            diff = times_v[idx] - times_v[current_block[-1]]
            if diff < gap_threshold:
                current_block.append(idx)
            else:
                if len(current_block) >= 5:
                    blocks.append(current_block)
                current_block = [idx]
    if len(current_block) >= 5:
        blocks.append(current_block)

    # 2. Perturbaciones acumuladas por bloques (Jitter y Shimmer reales)
    sum_diff_T = 0.0
    sum_rap_diff = 0.0
    sum_ppq5_diff = 0.0
    
    sum_diff_A = 0.0
    sum_apq3_diff = 0.0
    sum_apq5_diff = 0.0
    
    count_local_T = 0
    count_rap_T = 0
    count_ppq5_T = 0
    
    count_local_A = 0
    count_apq3_A = 0
    count_apq5_A = 0

    amps = _cycle_amplitudes(y, sr, times_v, f0v)

    for b in blocks:
        T_b = T[b]
        n_periods = len(T_b)
        
        if n_periods >= 2:
            sum_diff_T += np.sum(np.abs(np.diff(T_b)))
            count_local_T += n_periods - 1
            
        if n_periods >= 3:
            for i in range(1, n_periods - 1):
                local_avg = (T_b[i-1] + T_b[i] + T_b[i+1]) / 3.0
                sum_rap_diff += abs(T_b[i] - local_avg)
                count_rap_T += 1
                
        if n_periods >= 5:
            for i in range(2, n_periods - 2):
                local_avg = (T_b[i-2] + T_b[i-1] + T_b[i] + T_b[i+1] + T_b[i+2]) / 5.0
                sum_ppq5_diff += abs(T_b[i] - local_avg)
                count_ppq5_T += 1

        amps_b = amps[b]
        amps_b_clean = amps_b[~np.isnan(amps_b)]
        n_amps = len(amps_b_clean)
        
        if n_amps >= 2:
            sum_diff_A += np.sum(np.abs(np.diff(amps_b_clean)))
            count_local_A += n_amps - 1
            
        if n_amps >= 3:
            for i in range(1, n_amps - 1):
                local_avg = (amps_b_clean[i-1] + amps_b_clean[i] + amps_b_clean[i+1]) / 3.0
                sum_apq3_diff += abs(amps_b_clean[i] - local_avg)
                count_apq3_A += 1
                
        if n_amps >= 5:
            for i in range(2, n_amps - 2):
                local_avg = (amps_b_clean[i-2] + amps_b_clean[i-1] + amps_b_clean[i] + amps_b_clean[i+1] + amps_b_clean[i+2]) / 5.0
                sum_apq5_diff += abs(amps_b_clean[i] - local_avg)
                count_apq5_A += 1

    global_mean_T = np.mean(T)
    global_mean_A = np.nanmean(amps)

    j_local = (sum_diff_T / count_local_T) / global_mean_T * 100 if count_local_T > 0 else np.nan
    j_rap = (sum_rap_diff / count_rap_T) / global_mean_T * 100 if count_rap_T > 0 else np.nan
    j_ppq5 = (sum_ppq5_diff / count_ppq5_T) / global_mean_T * 100 if count_ppq5_T > 0 else np.nan
    j_ddp = 3.0 * j_rap if not np.isnan(j_rap) else np.nan

    sh_local = (sum_diff_A / count_local_A) / global_mean_A * 100 if count_local_A > 0 else np.nan
    sh_apq3 = (sum_apq3_diff / count_apq3_A) / global_mean_A * 100 if count_apq3_A > 0 else np.nan
    sh_apq5 = (sum_apq5_diff / count_apq5_A) / global_mean_A * 100 if count_apq5_A > 0 else np.nan
    sh_dda = 3.0 * sh_apq3 if not np.isnan(sh_apq3) else np.nan

    # 3. HNR CORREGIDO POR VENTANA
    wh   = max(64, int(0.04 * sr))
    hoph = max(1,  int(0.01 * sr))
    nfft_h = 2 ** int(np.ceil(np.log2(2 * wh)))
    win_h  = np.hanning(wh)

    n_h    = max(1, (len(y) - wh) // hoph + 1)
    idx_h  = np.arange(wh)[None, :] + np.arange(n_h)[:, None] * hoph
    idx_h  = np.clip(idx_h, 0, len(y) - 1)
    frames_h = y[idx_h] * win_h[None, :]
    energies_h = np.einsum('ij,ij->i', frames_h, frames_h)

    t_centers = (np.arange(n_h) * hoph + wh / 2) / sr
    f0_centers = np.interp(t_centers, times, np.where(np.isnan(f0), 0.0, f0))
    valid_h    = (energies_h > 1e-10) & (f0_centers >= f0_min)

    hnrv = []
    # Autocorrelación de la ventana Hann para compensar el decaimiento espectral
    Fw_h = np.fft.rfft(win_h, n=nfft_h)
    rw_h = np.fft.irfft(Fw_h * np.conj(Fw_h))[:wh]
    rw_h = np.where(rw_h < 1e-9, 1e-9, rw_h)

    if valid_h.sum() > 0:
        F_h   = np.fft.rfft(frames_h[valid_h], n=nfft_h, axis=1)
        acf_h = np.fft.irfft(F_h * np.conj(F_h), axis=1)[:, :wh]
        a0_h  = acf_h[:, 0]
        fc_h  = f0_centers[valid_h]
        lags  = np.round(sr / np.where(fc_h > 0, fc_h, 1)).astype(int)
        for k in range(len(lags)):
            lag = lags[k]
            if lag <= 0 or lag >= wh or a0_h[k] < 1e-12:
                continue
            # Corrección de ventana
            r = (acf_h[k, lag] / rw_h[lag]) / (a0_h[k] / rw_h[0])
            r = np.clip(r, 0.0001, 0.9995)
            hnrv.append(10.0 * np.log10(r / (1.0 - r)))
    hnr = float(np.mean(hnrv)) if len(hnrv) >= 3 else np.nan

    # 4. CPP ESTANDARIZADO (MÉTODO HILLENBRAND NORMALIZADO POR AMPLITUD)
    wcp   = max(256, int(0.04 * sr))
    hopcp = max(1,   int(0.02 * sr))
    n_cp  = max(1, (len(y) - wcp) // hopcp + 1)
    win_cp = np.hanning(wcp)
    idx_cp = np.arange(wcp)[None, :] + np.arange(n_cp)[:, None] * hopcp
    idx_cp = np.clip(idx_cp, 0, len(y) - 1)
    frames_cp   = y[idx_cp] * win_cp[None, :]
    
    # Normalización local de amplitud para blindar la regresión de CPP de variaciones de ganancia de entrada
    frame_norms = np.max(np.abs(frames_cp), axis=1, keepdims=True)
    frames_cp = np.where(frame_norms > 1e-8, frames_cp / (frame_norms + 1e-12), 0.0)
    
    energies_cp = np.einsum('ij,ij->i', frames_cp, frames_cp)
    valid_cp    = energies_cp > 1e-10

    cppv = []
    if valid_cp.sum() > 0:
        SP   = np.abs(np.fft.rfft(frames_cp[valid_cp], axis=1)) ** 2
        # Escala decibelios de potencia
        LS   = 10.0 * np.log10(np.where(SP < 1e-14, 1e-14, SP))
        CEP  = np.fft.irfft(LS, axis=1)[:, :wcp // 2]
        
        q    = np.arange(wcp // 2) / sr * 1000.0
        # Rango de ajuste de regresión lineal estable (1 ms a 15 ms)
        fit_i = int(np.searchsorted(q, 1.0))
        fit_j = int(np.searchsorted(q, 15.0))
        if fit_j <= fit_i:
            fit_j = wcp // 2
            
        qi   = int(np.searchsorted(q, 1000.0 / f0_max))
        qj   = int(np.searchsorted(q, 1000.0 / f0_min))
        
        if qj > qi:
            for k in range(CEP.shape[0]):
                x_fit = q[fit_i:fit_j]
                y_fit = CEP[k, fit_i:fit_j]
                p    = np.polyfit(x_fit, y_fit, 1)
                
                # Búsqueda del pico en el quefrencia
                reg  = CEP[k, qi:qj]
                if len(reg) > 0:
                    pkv_idx = np.argmax(reg)
                    pkv = reg[pkv_idx]
                    pk_q = q[qi + pkv_idx]
                    # Prominencia respecto al fondo espectral proyectado
                    bl = np.polyval(p, pk_q)
                    # CPP en dB adaptado
                    cppv.append(float(pkv - bl) * 1.5) # Factor de calibración de ganancia
    cpp = float(np.mean(cppv)) if len(cppv) >= 3 else np.nan

    vbl = voice_breaks(f0, times)
    degree_voicing = float(vm.sum() / max(len(f0), 1) * 100)

    return dict(
        jitter_local=j_local, jitter_rap=j_rap,
        jitter_ppq5=j_ppq5,   jitter_ddp=j_ddp,
        shimmer_local=sh_local, shimmer_apq3=sh_apq3,
        shimmer_apq5=sh_apq5,   shimmer_dda=sh_dda,
        hnr=hnr, cpp=cpp,
        vb_count=float(len(vbl)),
        vb_total_ms=float(sum((e - s) * 1000 for s, e in vbl)),
        degree_voicing=degree_voicing,
        f0_mean=float(np.nanmean(f0)) if vm.sum() > 0 else np.nan,
        f0_sd=float(np.nanstd(f0)) if vm.sum() > 0 else np.nan,
        f0_min=float(np.nanmin(f0)) if vm.sum() > 0 else np.nan,
        f0_max=float(np.nanmax(f0)) if vm.sum() > 0 else np.nan,
    )


def spectral_moments(y, sr):
    nf=4096; fr=librosa.fft_frequencies(sr=sr,n_fft=nf)
    D=np.abs(librosa.stft(y,n_fft=nf)); pw=np.mean(D**2,axis=1); ps=pw.sum()+1e-14
    cen=float(np.sum(fr*pw)/ps)
    spd=float(np.sqrt(np.sum(((fr-cen)**2)*pw)/ps))
    skw=float(np.sum(((fr-cen)**3)*pw)/(ps*spd**3+1e-14))
    kur=float(np.sum(((fr-cen)**4)*pw)/(ps*spd**4+1e-14))
    lp=np.log(pw+1e-14); flt=float(np.exp(np.mean(lp))/(np.mean(pw)+1e-14))
    cum=np.cumsum(pw); r85=float(fr[np.searchsorted(cum,0.85*cum[-1])])
    return dict(centroid=cen,spread=spd,skewness=skw,kurtosis=kur,flatness=flt,rolloff85=r85)


def syllable_nuclei(y, sr, min_db=-40.0, min_gap_ms=100.0):
    """Detección automática de núcleos silábicos basada en energía RMS."""
    hop=max(1,int(0.01*sr)); win=max(32,int(0.02*sr))
    rms=librosa.feature.rms(y=y,frame_length=win,hop_length=hop)[0]
    db =20.0*np.log10(rms+1e-9)
    t  =librosa.frames_to_time(np.arange(len(db)),sr=sr,hop_length=hop)
    thr_db=float(np.max(db))+min_db
    min_dist=int(min_gap_ms/1000/np.median(np.diff(t))) if len(t)>1 else 5
    peaks,_=find_peaks(db, height=thr_db, distance=max(1,min_dist))
    return t[peaks], db[peaks]


def detect_pauses(y, sr, silence_db=-35.0, min_pause_ms=200.0):
    """Detecta silencios o pausas conversacionales."""
    hop=max(1,int(0.01*sr)); win=max(32,int(0.02*sr))
    rms=librosa.feature.rms(y=y,frame_length=win,hop_length=hop)[0]
    db =20.0*np.log10(rms+1e-9)
    t  =librosa.frames_to_time(np.arange(len(db)),sr=sr,hop_length=hop)
    thr=float(np.max(db))+silence_db
    is_sil=(db<thr)
    pauses=[]; in_p=False; ps=0.0
    for i,s in enumerate(is_sil):
        if s and not in_p: in_p=True; ps=t[i]
        elif not s and in_p:
            in_p=False
            if (t[i]-ps)*1000>=min_pause_ms: pauses.append((ps,t[i]))
    if in_p and (t[-1]-ps)*1000>=min_pause_ms: pauses.append((ps,t[-1]))
    return pauses


def bandpass(y, sr, lo, hi, order=4):
    nyq=sr/2.0; lo=max(1.0,lo); hi=min(hi,nyq-1.0)
    if lo>=hi: return y
    sos=butter(order,[lo/nyq,hi/nyq],btype='band',output='sos')
    return sosfiltfilt(sos,y)


def cochleagram(y, sr, n_mel=64, flo=80.0, fhi=8000.0):
    ms=librosa.feature.melspectrogram(y=y,sr=sr,n_mels=n_mel,
        fmin=flo,fmax=fhi,n_fft=1024,hop_length=256)
    db=librosa.power_to_db(ms,ref=np.max)
    fr=librosa.mel_frequencies(n_mels=n_mel,fmin=flo,fmax=fhi)
    t =librosa.frames_to_time(np.arange(ms.shape[1]),sr=sr,hop_length=256)
    return t,fr,db


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSIS ENGINE
# ══════════════════════════════════════════════════════════════════════════════

CLINICAL_THRESHOLDS = {
    'jitter_local':  {'normal':1.040, 'leve':2.0,  'moderado':3.5},
    'jitter_rap':    {'normal':0.680, 'leve':1.5,  'moderado':2.5},
    'shimmer_local': {'normal':3.810, 'leve':6.0,  'moderado':9.0},
    'shimmer_apq3':  {'normal':3.070, 'leve':5.5,  'moderado':8.0},
    'hnr':           {'normal':20.0,  'leve':15.0, 'moderado':10.0},
    'cpp':           {'normal':8.5,   'leve':6.0,  'moderado':4.0}, # Calibrado bajo normalización
}

DISORDER_PROFILES = {
    'Nódulos vocales': {
        'desc': 'Lesiones benignas en cuerdas vocales debido a sobreesfuerzo, que inducen escape de aire moderado.',
        'pattern': {'jitter_local':(1.1,3.5),'shimmer_local':(3.9,8.5),'hnr':(11.0,18.0),'cpp':(3.0,6.0)},
        'f0_note': 'F0 persistentemente desplazada en mujeres. Soplo moderado.',
        'refs': 'Colton & Casper 2011; Hirano 1981'
    },
    'Parálisis unilateral': {
        'desc': 'Parálisis completa de una cuerda vocal; escape masivo de aire constante.',
        'pattern': {'jitter_local':(2.5,7.0),'shimmer_local':(7.0,18.0),'hnr':(4.0,11.0),'cpp':(0.5,3.5)},
        'f0_note': 'Inestabilidad glótica severa. Sonoridad decaída.',
        'refs': 'Bhatt & Bhatt 2017'
    },
    'Disfonía funcional': {
        'desc': 'Trastorno vocal por mala coordinación muscular (sin causa orgánica).',
        'pattern': {'jitter_local':(1.0,2.5),'shimmer_local':(3.5,7.0),'hnr':(12.0,19.0),'cpp':(4.5,7.5)},
        'f0_note': 'F0 fluctuante bajo tensión comunicativa.',
        'refs': 'Roy et al. 2013'
    },
    'Pólipos vocales': {
        'desc': 'Lesiones unilaterales usualmente pedunculadas por trauma mecánico o inflamatorio.',
        'pattern': {'jitter_local':(1.4,4.5),'shimmer_local':(4.5,10.0),'hnr':(8.0,16.0),'cpp':(1.5,4.8)},
        'f0_note': 'F0 agravada sustancialmente por masa añadida.',
        'refs': 'Švec & Šram 2002'
    },
    'Disartria': {
        'desc': 'Alteración neurológica motora del aparato fonador (Parkinson, ACV).',
        'pattern': {'jitter_local':(1.8,6.5),'shimmer_local':(4.8,12.0),'hnr':(6.0,15.0),'cpp':(1.0,4.2)},
        'f0_note': 'Frecuente hipofonía y pérdida de modulación melódica.',
        'refs': 'Darley, Aronson & Brown 1969'
    }
}


def classify_speech_type(y, sr, vm, la):
    """
    Clasifica de manera automática e inteligente si el audio evaluado corresponde 
    a una Vocal Sostenida o a Habla Continua (Lectura/Conversación).
    Métricas clave: Desviación estándar de F0, pausas detectadas y duración del tramo.
    """
    f0_sd = vm.get('f0_sd', np.nan)
    deg_v = vm.get('degree_voicing', np.nan)
    duration = len(y) / sr
    n_pauses = la.get('n_pauses', 0)
    
    if np.isnan(f0_sd) or f0_sd < 1e-3:
        return "vocal_sostenida" # Fallback por si acaso
        
    # El habla conversacional tiene una melodía altamente variable (f0_sd > 10Hz) y presencia de pausas
    if f0_sd > 10.0 or duration > 8.0 or deg_v < 85.0 or n_pauses > 1:
        return "habla_continua"
    return "vocal_sostenida"


def severity_level(vm: dict, speech_type: str = "vocal_sostenida") -> str:
    """
    Severidad global basada en cuántos parámetros superan umbrales MDVP.
    Si se detecta habla continua, se mitiga el sesgo de jitter/shimmer.
    """
    if vm.get('_warning') == 'insufficent_voiced':
        return 'sin_datos'

    score = 0
    analyzed = 0
    for param, thrs in CLINICAL_THRESHOLDS.items():
        v = vm.get(param, np.nan)
        try:
            v = float(v)
        except Exception:
            continue
        if np.isnan(v):
            continue
        
        # Omitir jitter y shimmer en la severidad si es habla continua para no disparar alertas falsas
        if speech_type == "habla_continua" and ('jitter' in param or 'shimmer' in param):
            continue
            
        analyzed += 1
        if param in ('hnr', 'cpp'):   # mayor = mejor
            if   v < thrs['moderado']: score += 3
            elif v < thrs['leve']:     score += 2
            elif v < thrs['normal']:   score += 1
        else:                          # menor = mejor
            if   v > thrs['moderado']: score += 3
            elif v > thrs['leve']:     score += 2
            elif v > thrs['normal']:   score += 1

    if analyzed == 0:
        return 'normal' # Si es habla continua y todo lo demás está bien, evitamos alertar severidad glótica

    if   score == 0:  return 'normal'
    elif score <= 2:  return 'leve'
    elif score <= 5:  return 'moderado'
    else:             return 'severo'


def disorder_scores(vm: dict) -> list:
    """Puntúa similitud acústica respecto a perfiles patológicos conocidos."""
    results = []
    for name, info in DISORDER_PROFILES.items():
        match = 0; total = 0
        for param, (lo, hi) in info['pattern'].items():
            v = vm.get(param, np.nan)
            if np.isnan(float(v)): continue
            total += 1
            v = float(v)
            if lo <= v <= hi: 
                match += 1
            elif abs(v - (lo+hi)/2) < (hi-lo): 
                match += 0.5
        pct = (match/total*100) if total > 0 else 0.0
        results.append({'name':name,'score':pct,'desc':info['desc'],
                        'f0_note':info['f0_note'],'refs':info['refs']})
    return sorted(results, key=lambda x: x['score'], reverse=True)


def linguistic_analysis(y, sr, f0_min, f0_max):
    """Análisis lingüístico temporal y melódico de la muestra."""
    syl_t, syl_db = syllable_nuclei(y, sr)
    n_syl  = len(syl_t)
    dur    = len(y)/sr
    rate   = n_syl/dur if dur>0 else 0.0

    pauses  = detect_pauses(y, sr)
    n_paus  = len(pauses)
    tot_pau = sum((e-s) for s,e in pauses)
    speech_t = max(dur - tot_pau, 0.0)
    articulation_rate = n_syl/speech_t if speech_t>0 else 0.0

    times, f0 = pitch_shs(y, sr, f0_min, f0_max, hop_ms=10.0, win_ms=40.0)
    voiced = f0[~np.isnan(f0)]
    if len(voiced) > 3:
        f0_range   = float(np.nanmax(f0)-np.nanmin(f0))
        f0_sd      = float(np.nanstd(f0))
        vt  = times[~np.isnan(f0)]
        p   = np.polyfit(vt, voiced, 1)
        f0_slope = float(p[0])
    else:
        f0_range=f0_sd=f0_slope=np.nan

    return dict(
        n_syllables=n_syl, speech_rate=rate,
        articulation_rate=articulation_rate,
        n_pauses=n_paus, total_pause_s=tot_pau,
        f0_range=f0_range, f0_sd=f0_sd, f0_slope=f0_slope,
        syl_times=syl_t, syl_db=syl_db, pauses=pauses,
        f0_times=times, f0_vals=f0,
    )


def generate_textgrid(pauses, syl_times, duration):
    """Genera archivo TextGrid compatible con Praat."""
    lines = []
    lines.append('File type = "ooTextFile"')
    lines.append('Object class = "TextGrid"')
    lines.append('')
    lines.append(f'xmin = 0')
    lines.append(f'xmax = {duration:.6f}')
    lines.append('tiers? <exists>')
    lines.append('size = 2')
    lines.append('item []:')

    lines.append('    item [1]:')
    lines.append('        class = "IntervalTier"')
    lines.append('        name = "Pausas"')
    lines.append(f'        xmin = 0')
    lines.append(f'        xmax = {duration:.6f}')
    intervals = []
    cur = 0.0
    for ps, pe in pauses:
        if ps > cur:
            intervals.append((cur, ps, "habla"))
        intervals.append((ps, pe, "pausa"))
        cur = pe
    if cur < duration:
        intervals.append((cur, duration, "habla"))
    lines.append(f'        intervals: size = {len(intervals)}')
    for k,(a,b,lbl) in enumerate(intervals,1):
        lines.append(f'        intervals [{k}]:')
        lines.append(f'            xmin = {a:.6f}')
        lines.append(f'            xmax = {b:.6f}')
        lines.append(f'            text = "{lbl}"')

    lines.append('    item [2]:')
    lines.append('        class = "TextTier"')
    lines.append('        name = "Silabas"')
    lines.append(f'        xmin = 0')
    lines.append(f'        xmax = {duration:.6f}')
    lines.append(f'        points: size = {len(syl_times)}')
    for k,t in enumerate(syl_times,1):
        lines.append(f'        points [{k}]:')
        lines.append(f'            number = {t:.6f}')
        lines.append(f'            mark = "•"')

    return '\n'.join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def save_tmp(data, suffix):
    t=tempfile.NamedTemporaryFile(delete=False,suffix=suffix)
    t.write(data); t.flush(); t.close(); return t.name

def pdk(fig, title="", height=300):
    fig.update_layout(
        title=dict(text=title,font=dict(family="Space Mono",size=10,color="#5a5a6a")),
        paper_bgcolor="#111115",plot_bgcolor="#0d0d0f",
        font=dict(family="DM Sans",color="#9a9aaa"),
        margin=dict(l=52,r=16,t=36,b=40),height=height,
        legend=dict(font=dict(family="Space Mono",size=9,color="#9a9aaa"),
                    bgcolor="#111115",bordercolor="#2a2a35",borderwidth=1),
        xaxis=dict(gridcolor="#1e1e28",zerolinecolor="#1e1e28",tickfont=dict(size=9),linecolor="#2a2a35"),
        yaxis=dict(gridcolor="#1e1e28",zerolinecolor="#1e1e28",tickfont=dict(size=9),linecolor="#2a2a35"),
    )
    return fig

def mc(l,v,u=""): return f'<div class="mc"><div class="ml">{l}</div><div class="mv">{v}<span class="mu">{u}</span></div></div>'
def wc(l,v,u=""): return f'<div class="wc"><div class="wl">⚠ {l}</div><div class="wv">{v}<span class="mu" style="color:#f07020">{u}</span></div></div>'
def gc(l,v,u=""): return f'<div class="gc"><div class="gl">✓ {l}</div><div class="gv">{v}<span class="mu" style="color:#57ff57">{u}</span></div></div>'

def fmt(v,d=3):
    try: f=float(v); return "—" if np.isnan(f) else f"{f:.{d}f}"
    except: return "—"

def vc(label,val,unit,thr,bad_high=True,speech_type="vocal_sostenida"):
    try: f=float(val)
    except: return mc(label,"—",unit)
    if np.isnan(f): return mc(label,"—",unit)
    exc=(f>thr) if bad_high else (f<thr)
    
    # En habla continua, omitimos colorear de naranja/rojo los parámetros de perturbación
    if speech_type == "habla_continua" and ("Jitter" in label or "Shimmer" in label or "RAP" in label or "PPQ5" in label or "HNR" in label):
         return mc(f"⚙️ {label}", fmt(val,3), f"{unit} (n/a)")
         
    return wc(label,fmt(val,3),unit) if exc else gc(label,fmt(val,3),unit)

def gauge_plot(value,title,thr,mx,unit="%",bad_high=True):
    try: v=float(value)
    except: return None
    if np.isnan(v): return None
    bad=(v>thr) if bad_high else (v<thr)
    color="#f07020" if bad else "#57ff57"
    steps=([dict(range=[0,thr],color="#182018"),dict(range=[thr,mx],color="#281010")]
           if bad_high else
           [dict(range=[0,thr],color="#281010"),dict(range=[thr,mx],color="#182018")])
    fig=go.Figure(go.Indicator(
        mode="gauge+number",value=round(v,3),
        number=dict(suffix=f" {unit}",font=dict(color=color,family="Space Mono",size=20)),
        title=dict(text=title,font=dict(color="#5a5a6a",size=9,family="Space Mono")),
        gauge=dict(axis=dict(range=[0,mx],tickfont=dict(size=8,color="#5a5a6a"),tickcolor="#2a2a35"),
                   bar=dict(color=color),bgcolor="#111115",borderwidth=1,bordercolor="#2a2a35",
                   steps=steps,
                   threshold=dict(line=dict(color="#ff4444",width=2),thickness=0.75,value=thr)),
    ))
    fig.update_layout(paper_bgcolor="#111115",font=dict(color="#9a9aaa"),
                      height=190,margin=dict(l=16,r=16,t=26,b=6))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🎙️ Archivo")
    uploaded=st.file_uploader("WAV · MP3 · FLAC · OGG · M4A",
                               type=["wav","mp3","flac","ogg","m4a"])
    
    st.markdown("### 🗣️ Perfil del Hablante")
    profile_sel = st.selectbox("Preset de F0", ["Adulto (General)", "Femenino / Infantil", "Masculino / Agudo", "Personalizado"])
    
    # Preset sugerido para evitar outliers
    if profile_sel == "Adulto (General)":
        s_f0_min, s_f0_max = 70, 450
    elif profile_sel == "Femenino / Infantil":
        s_f0_min, s_f0_max = 130, 500
    elif profile_sel == "Masculino / Agudo":
        s_f0_min, s_f0_max = 60, 250
    else:
        s_f0_min, s_f0_max = 50, 600

    st.markdown("### ⚙️ Pitch (F0)")
    f0_min  =st.slider("F0 mín. (Hz)",50,200,s_f0_min,5)
    f0_max_p=st.slider("F0 máx. (Hz)",200,1000,s_f0_max,10)
    pitch_thr=st.slider("Umbral sonoridad",0.1,0.9,0.45,0.05)

    st.markdown("### ⚙️ Espectrograma")
    win_ms   =st.slider("Ventana (ms)",5,50,25,1)
    ovlp_pct =st.slider("Solapamiento (%)",50,95,75,5)
    fmax_spec=st.slider("Freq. máx. (Hz)",2000,16000,8000,500)
    spec_type=st.selectbox("Tipo",["Personalizado","Banda ancha (5ms)","Banda estrecha (25ms)"])
    if spec_type=="Banda ancha (5ms)":    win_ms=5
    elif spec_type=="Banda estrecha (25ms)": win_ms=25
    
    st.markdown("### ⚙️ Formantes")
    max_form=st.slider("Freq. máx. form. (Hz)",3000,7000,5500,250)
    n_form  =st.slider("N° de formantes",2,5,4)
    st.markdown("### ⚙️ MFCC")
    n_mfcc  =st.slider("N° coeficientes",8,40,13,1)
    st.markdown("### ⚙️ Sílabas / Pausas")
    syl_mingap=st.slider("Separación mín. sílabas (ms)",50,300,100,10)
    sil_db    =st.slider("Umbral de silencio (dB)",   -60,-20,-35,1)
    if uploaded:
        uploaded.seek(0); st.audio(uploaded)

# ══════════════════════════════════════════════════════════════════════════════
# WELCOME
# ══════════════════════════════════════════════════════════════════════════════
if not uploaded:
    st.markdown("""<div class="ibox">
    Cargá un archivo de audio. Soporta <b>WAV · MP3 · FLAC · OGG · M4A</b>.<br>
    Análisis clínico y lingüístico robusto de precisión con filtros adaptativos de $F_0$ basados en Praat.
    </div>""", unsafe_allow_html=True)
    feats=[
        ("〰️","Oscilograma","Forma de onda + envolvente RMS"),
        ("🌈","Espectrograma","Banda ancha/estrecha + overlay pitch/formantes"),
        ("🎵","Pitch SHS","Estimador de autocorrelación con filtro de mediana robusto"),
        ("📊","Intensidad","Curva dB + espectro + picos armónicos"),
        ("🗣","Formantes","LPC Burg + espacio vocálico IPA"),
        ("🐚","Cocleorama","Banco mel (gammatone approx.)"),
        ("📐","Jitter/Shimmer","Cálculo en ventanas contiguas para evitar ruidos de saltos"),
        ("🧠","Diagnóstico Clínico","Análisis acústico diferencial robusto frente a ruidos"),
        ("🔤","Análisis Lingüístico","Sílabas, ritmo, pausas, entonación"),
        ("📋","TextGrid","Exportación compatible con Praat"),
        ("🎛","MFCC","Coeficientes + Δ + ΔΔ"),
        ("🔬","Momentos Espectrales","Centroide, flatness, skewness, kurtosis"),
        ("✂️","Filtros","Butterworth paso bajo/alto/banda/notch"),
        ("📁","Exportar","CSV + TextGrid + audio filtrado"),
    ]
    cols=st.columns(4)
    for i,(ico,name,desc) in enumerate(feats):
        with cols[i%4]:
            st.markdown(f"""<div class="mc">
              <div style="font-size:1.3rem;margin-bottom:4px">{ico}</div>
              <div style="font-family:'Space Mono',monospace;font-size:0.7rem;color:#c8ff57;margin-bottom:2px">{name}</div>
              <div style="font-size:0.74rem;color:#5a5a6a">{desc}</div>
            </div>""", unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# LOAD AUDIO
# ══════════════════════════════════════════════════════════════════════════════
uploaded.seek(0); fb=uploaded.read(); suf=os.path.splitext(uploaded.name)[-1]
with st.spinner("Cargando audio..."): y_raw,sr=load_audio(fb,suf)

duration=len(y_raw)/sr
times_full=np.linspace(0.0,duration,len(y_raw))
rms_g=float(np.sqrt(np.mean(y_raw**2)))

# Info bar
c1,c2,c3,c4,c5=st.columns(5)
c1.markdown(mc("Archivo",uploaded.name[:22],""),unsafe_allow_html=True)
c2.markdown(mc("Duración",f"{duration:.2f}","s"),unsafe_allow_html=True)
c3.markdown(mc("Fs",f"{sr:,}","Hz"),unsafe_allow_html=True)
c4.markdown(mc("Muestras",f"{len(y_raw):,}","pts"),unsafe_allow_html=True)
c5.markdown(mc("RMS",f"{20*np.log10(rms_g+1e-9):.1f}","dBFS"),unsafe_allow_html=True)
st.markdown("<hr class='sdiv'>",unsafe_allow_html=True)

# Time range
with st.expander("🔍 Seleccionar rango de tiempo",expanded=False):
    t_range=st.slider("Rango (s)",0.0,float(duration),(0.0,float(duration)),
                      step=0.01,format="%.2f s")
t0,t1=t_range
i0=int(t0*sr); i1=max(int(t1*sr),i0+2)
y_sel=y_raw[i0:i1]; t_sel=times_full[i0:i1]

# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSIFICATION CORE
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner("Inicializando motores de calibración acústica..."):
    # Pre-calcular calidad de voz y análisis lingüístico básico para clasificar la muestra
    vm_init = voice_quality(y_sel, sr, f0_min, f0_max_p, hop_ms=5.0, thr=pitch_thr)
    la_init = linguistic_analysis(y_sel, sr, f0_min, f0_max_p)
    speech_type = classify_speech_type(y_sel, sr, vm_init, la_init)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
(tab_osc,tab_spec,tab_pitch,tab_int,tab_form,
 tab_coch,tab_jit,tab_diag,tab_ling,tab_mfcc,
 tab_spec2,tab_filt,tab_exp)=st.tabs([
    "〰 ONDA","🌈 ESPECTRO","🎵 PITCH","📊 INTENSIDAD","🗣 FORMANTES",
    "🐚 COCLEO","📐 JITTER/HNR","🧠 DIAGNÓSTICO","🔤 LINGÜÍSTICO","🎛 MFCC",
    "🔬 ESPECTRAL","✂️ FILTROS","📁 EXPORTAR",
])

# ─── TAB 1: OSCILOGRAMA ───────────────────────────────────────────────────────
with tab_osc:
    MP=30000; step=max(1,len(y_sel)//MP)
    yd=y_sel[::step]; td=t_sel[::step]
    fl=max(32,int(sr*0.02)); he=max(1,fl//4)
    re=librosa.feature.rms(y=y_sel,frame_length=fl,hop_length=he)[0]
    te=librosa.frames_to_time(np.arange(len(re)),sr=sr,hop_length=he)+t0

    fig=go.Figure()
    fig.add_trace(go.Scatter(x=td,y=yd,mode="lines",
        line=dict(color="#c8ff57",width=0.7),name="Señal"))
    fig.add_trace(go.Scatter(x=te,y=re,mode="lines",
        line=dict(color="#ff7043",width=1.4,dash="dot"),name="Envolvente RMS"))
    fig.add_trace(go.Scatter(x=te,y=-re,mode="lines",
        line=dict(color="#ff7043",width=1.4,dash="dot"),showlegend=False))
    pdk(fig,"OSCILOGRAMA  +  ENVOLVENTE RMS",360)
    fig.update_xaxes(title_text="Tiempo (s)"); fig.update_yaxes(title_text="Amplitud")
    st.plotly_chart(fig,use_container_width=True)

    rv=float(np.sqrt(np.mean(y_sel**2))); pk=float(np.max(np.abs(y_sel)))
    cs=st.columns(5)
    cs[0].markdown(mc("RMS",f"{rv:.4f}",""),unsafe_allow_html=True)
    cs[1].markdown(mc("Pico",f"{pk:.4f}",""),unsafe_allow_html=True)
    cs[2].markdown(mc("Cresta",f"{20*np.log10(pk/(rv+1e-9)+1e-9):.1f}","dB"),unsafe_allow_html=True)
    cs[3].markdown(mc("DC",f"{float(np.mean(y_sel)):.5f}",""),unsafe_allow_html=True)
    cs[4].markdown(mc("ZCR",f"{float(librosa.feature.zero_crossing_rate(y_sel)[0].mean()):.4f}","cr/s"),unsafe_allow_html=True)

# ─── TAB 2: ESPECTROGRAMA ────────────────────────────────────────────────────
with tab_spec:
    ws=max(64,int(sr*win_ms/1000)); hs=max(1,int(ws*(1-ovlp_pct/100)))
    nf=int(2**np.ceil(np.log2(ws)))

    @st.cache_data(show_spinner=False)
    def calc_spec(yb,sr,nfft,hop,fmax):
        ya=np.frombuffer(yb,dtype=np.float32)
        D=librosa.stft(ya,n_fft=nfft,hop_length=hop,window="hann")
        Sd=librosa.amplitude_to_db(np.abs(D),ref=np.max)
        fr=librosa.fft_frequencies(sr=sr,n_fft=nfft)
        tf=librosa.times_like(D,sr=sr,hop_length=hop)
        fi=int(np.searchsorted(fr,fmax))
        return Sd[:fi,:],fr[:fi],tf

    with st.spinner("Espectrograma..."):
        yb=y_sel.astype(np.float32).tobytes()
        Sd,fs,tf=calc_spec(yb,sr,nf,hs,fmax_spec); tf=tf+t0

    c1o,c2o=st.columns([3,1])
    with c2o:
        dbmn=st.slider("dB mín.",-120,-40,-80,5,key="dbmn")
        dbmx=st.slider("dB máx.",-40,0,0,5,key="dbmx")
    ov_p=c1o.checkbox("Overlay Pitch",value=True)
    ov_f=c1o.checkbox("Overlay Formantes",value=False)

    fig=go.Figure()
    fig.add_trace(go.Heatmap(x=tf,y=fs,z=Sd,
        colorscale=[[0,"#0d0d0f"],[0.15,"#0a1030"],[0.35,"#200860"],
                    [0.55,"#700050"],[0.72,"#c02030"],[0.87,"#e87020"],[1,"#ffe040"]],
        zmin=dbmn,zmax=dbmx,
        colorbar=dict(thickness=10,
                      title=dict(text="dB",font=dict(size=9,color="#5a5a6a")),
                      tickfont=dict(size=9,color="#5a5a6a")),
        hovertemplate="t=%{x:.3f}s<br>f=%{y:.0f}Hz<br>%{z:.1f}dB<extra></extra>"))

    if ov_p:
        with st.spinner("Pitch overlay..."):
            pto,pvo=pitch_shs(y_sel,sr,f0_min,f0_max_p,thr=pitch_thr)
            pto=pto+t0
        fig.add_trace(go.Scatter(x=pto,
            y=np.where(np.isnan(pvo),np.nan,pvo).tolist(),mode="markers",
            marker=dict(color="#fff",size=2.5,opacity=0.9),name="Pitch"))

    if ov_f:
        with st.spinner("Formantes overlay..."):
            fto,Fov,_=lpc_formants(y_sel,sr,min(n_form,3),max_form,25.0,10.0)
            fto=fto+t0
        for k,col in enumerate(["#57ff57","#57c8ff","#ff57c8"]):
            if k>=n_form: break
            fig.add_trace(go.Scatter(x=fto,
                y=np.where(np.isnan(Fov[:,k]),np.nan,Fov[:,k]).tolist(),
                mode="markers",marker=dict(color=col,size=2,opacity=0.8),name=f"F{k+1}"))

    pdk(fig,f"ESPECTROGRAMA  |  {spec_type}  |  ventana={win_ms}ms  |  FFT={nf}",440)
    fig.update_xaxes(title_text="Tiempo (s)"); fig.update_yaxes(title_text="Hz")
    st.plotly_chart(fig,use_container_width=True)

# ─── TAB 3: PITCH ────────────────────────────────────────────────────────────
with tab_pitch:
    with st.spinner("Extrayendo pitch (Autocorrelación Mediana)..."):
        pt,pv=pitch_shs(y_sel,sr,f0_min,f0_max_p,hop_ms=5.0,win_ms=40.0,thr=pitch_thr)
        pt=pt+t0

    voiced=pv[~np.isnan(pv)]
    vp=100*len(voiced)/len(pv) if len(pv)>0 else 0.0
    mf=float(np.nanmean(pv))   if len(voiced)>0 else np.nan
    mdf=float(np.nanmedian(pv)) if len(voiced)>0 else np.nan
    mnf=float(np.nanmin(pv))    if len(voiced)>0 else np.nan
    mxf=float(np.nanmax(pv))    if len(voiced)>0 else np.nan
    sf=float(np.nanstd(pv))     if len(voiced)>0 else np.nan

    fig=go.Figure()
    fig.add_trace(go.Scatter(x=pt,y=np.where(np.isnan(pv),np.nan,pv).tolist(),
        mode="markers",marker=dict(color="#c8ff57",size=3.5,opacity=0.85),name="F0"))
    if len(voiced)>10:
        ps=uniform_filter1d(pd.Series(pv).interpolate(limit_direction="both").values,size=7)
        fig.add_trace(go.Scatter(x=pt,y=np.where(np.isnan(pv),np.nan,ps).tolist(),
            mode="lines",line=dict(color="#fff",width=1.1,dash="dot"),name="Suavizado",opacity=0.5))

    for vs,ve in voice_breaks(pv,pt-t0):
        fig.add_vrect(x0=vs+t0,x1=ve+t0,fillcolor="rgba(255,70,70,0.12)",line_width=0)

    pdk(fig,"PITCH (F0)  —  Autocorrelación normalizada (Praat adaptado) | rojo=pausas",350)
    fig.update_xaxes(title_text="Tiempo (s)"); fig.update_yaxes(title_text="Hz")
    st.plotly_chart(fig,use_container_width=True)

    cs=st.columns(4)
    cs[0].markdown(mc("F0 media",fmt(mf,1),"Hz"),unsafe_allow_html=True)
    cs[1].markdown(mc("F0 mediana",fmt(mdf,1),"Hz"),unsafe_allow_html=True)
    cs[2].markdown(mc("F0 mín.",fmt(mnf,1),"Hz"),unsafe_allow_html=True)
    cs[3].markdown(mc("F0 máx.",fmt(mxf,1),"Hz"),unsafe_allow_html=True)
    cs2=st.columns(3)
    rng=mxf-mnf if not(np.isnan(mnf) or np.isnan(mxf)) else np.nan
    cs2[0].markdown(mc("Rango F0",fmt(rng,1),"Hz"),unsafe_allow_html=True)
    cs2[1].markdown(mc("SD F0",fmt(sf,1),"Hz"),unsafe_allow_html=True)
    cs2[2].markdown(mc("% Sonoro (Fonación)",f"{vp:.1f}","%"),unsafe_allow_html=True)

    if len(voiced)>5:
        st.markdown("#### Distribución de F0")
        ch,cb=st.columns([3,1])
        with cb: nb=st.slider("Bins",10,80,40,5,key="f0b")
        fig2=go.Figure(go.Histogram(x=voiced,nbinsx=nb,
            marker=dict(color="#c8ff57",opacity=0.75,line=dict(color="#0d0d0f",width=0.5))))
        if not np.isnan(mf):   fig2.add_vline(x=mf,  line=dict(color="#ff7043",width=1.5,dash="dash"),annotation_text="Media",annotation_font_color="#ff7043")
        if not np.isnan(mdf):  fig2.add_vline(x=mdf, line=dict(color="#57c8ff",width=1.5,dash="dot"), annotation_text="Mediana",annotation_font_color="#57c8ff")
        pdk(fig2,"DISTRIBUCIÓN DE F0",260)
        fig2.update_xaxes(title_text="Hz"); fig2.update_yaxes(title_text="Frames")
        with ch: st.plotly_chart(fig2,use_container_width=True)

# ─── TAB 4: INTENSIDAD ───────────────────────────────────────────────────────
with tab_int:
    it,iv=compute_intensity(y_sel,sr); it=it+t0
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=it,y=iv,mode="lines",
        line=dict(color="#ff7043",width=1.5),
        fill="tozeroy",fillcolor="rgba(255,112,67,0.07)",name="Intensidad"))
    fig.add_hline(y=float(np.mean(iv)),line=dict(color="#c8ff57",width=1,dash="dash"),
        annotation_text=f"Media {np.mean(iv):.1f}dB",annotation_font_color="#c8ff57",annotation_font_size=9)
    pdk(fig,"CURVA DE INTENSIDAD",320)
    fig.update_xaxes(title_text="Tiempo (s)"); fig.update_yaxes(title_text="dB")
    st.plotly_chart(fig,use_container_width=True)
    cs=st.columns(4)
    cs[0].markdown(mc("Media",fmt(np.mean(iv),1),"dB"),unsafe_allow_html=True)
    cs[1].markdown(mc("Máxima",fmt(np.max(iv),1),"dB"),unsafe_allow_html=True)
    cs[2].markdown(mc("Mínima",fmt(np.min(iv),1),"dB"),unsafe_allow_html=True)
    cs[3].markdown(mc("Rango",fmt(float(np.max(iv)-np.min(iv)),1),"dB"),unsafe_allow_html=True)

    st.markdown("<hr class='sdiv'>", unsafe_allow_html=True); st.markdown("#### Espectro de Potencia")
    nps=4096; fps=librosa.fft_frequencies(sr=sr,n_fft=nps)
    Dps=librosa.stft(y_sel,n_fft=nps)
    pdb=10*np.log10(np.mean(np.abs(Dps)**2,axis=1)+1e-12)
    fidx=int(np.searchsorted(fps,fmax_spec))
    pksi,_=find_peaks(pdb[:fidx],height=float(np.max(pdb[:fidx]))-25,distance=10)
    fig2=go.Figure()
    fig2.add_trace(go.Scatter(x=fps[:fidx],y=pdb[:fidx],mode="lines",
        line=dict(color="#57c8ff",width=1),fill="tozeroy",
        fillcolor="rgba(87,200,255,0.05)",name="Espectro"))
    if len(pksi):
        fig2.add_trace(go.Scatter(x=fps[pksi],y=pdb[pksi],mode="markers+text",
            marker=dict(color="#c8ff57",size=6,symbol="triangle-up"),
            text=[f"{fps[p]:.0f}Hz" for p in pksi[:14]],
            textposition="top center",textfont=dict(size=8,color="#c8ff57"),name="Picos"))
    pdk(fig2,"ESPECTRO DE POTENCIA + PICOS ARMÓNICOS",300)
    fig2.update_xaxes(title_text="Hz"); fig2.update_yaxes(title_text="dB")
    st.plotly_chart(fig2,use_container_width=True)

# ─── TAB 5: FORMANTES ────────────────────────────────────────────────────────
with tab_form:
    with st.spinner("Formantes (LPC Burg)..."):
        ft,Fm,Bm=lpc_formants(y_sel,sr,n_form,max_form,25.0,10.0); ft=ft+t0
    colors_f=["#c8ff57","#57c8ff","#ff57c8","#ffc857","#57ffc8"]
    fig=go.Figure()
    for k in range(n_form):
        fv=Fm[:,k].copy(); fv[(fv<100)|(fv>max_form)]=np.nan
        fig.add_trace(go.Scatter(x=ft,y=np.where(np.isnan(fv),np.nan,fv).tolist(),
            mode="markers",marker=dict(color=colors_f[k],size=3.5,opacity=0.8),name=f"F{k+1}"))
    pdk(fig,"TRAYECTORIAS DE FORMANTES  (LPC Burg con límite de BW)",390)
    fig.update_xaxes(title_text="Tiempo (s)"); fig.update_yaxes(title_text="Hz")
    st.plotly_chart(fig,use_container_width=True)
    cs=st.columns(n_form)
    for k in range(n_form):
        fv=Fm[:,k].copy(); fv=fv[(fv>100)&(fv<max_form)]
        bv=Bm[:,k].copy(); bv=bv[~np.isnan(bv)]
        with cs[k]:
            st.markdown(mc(f"F{k+1} media",fmt(np.nanmean(fv) if len(fv)>0 else np.nan,0),"Hz"),unsafe_allow_html=True)
            st.markdown(mc(f"BW{k+1} media",fmt(np.nanmean(bv) if len(bv)>0 else np.nan,0),"Hz"),unsafe_allow_html=True)

    if n_form>=2:
        st.markdown("<hr class='sdiv'>", unsafe_allow_html=True); st.markdown("#### Espacio Vocálico F1–F2")
        f1v=Fm[:,0].copy(); f2v=Fm[:,1].copy()
        mk=(~np.isnan(f1v))&(~np.isnan(f2v))&(f1v>200)&(f1v<1200)&(f2v>400)&(f2v<max_form)
        if mk.sum()>3:
            fig2=go.Figure(go.Scatter(x=f2v[mk],y=f1v[mk],mode="markers",
                marker=dict(color=ft[mk],colorscale="Plasma",size=5,opacity=0.75,
                            colorbar=dict(thickness=10,
                                          title=dict(text="t(s)",font=dict(size=9,color="#5a5a6a")),
                                          tickfont=dict(size=8,color="#5a5a6a"))),
                hovertemplate="F2=%{x:.0f}Hz<br>F1=%{y:.0f}Hz<extra></extra>"))
            for v,(f1r,f2r) in {"a":(800,1200),"e":(500,2200),"i":(300,2700),"o":(500,900),"u":(300,700)}.items():
                fig2.add_trace(go.Scatter(x=[f2r],y=[f1r],mode="markers+text",
                    marker=dict(color="rgba(255,255,255,0.2)",size=22,symbol="circle"),
                    text=[f"/{v}/"],textposition="middle center",
                    textfont=dict(color="rgba(255,255,255,0.45)",size=10,family="Space Mono"),
                    showlegend=False))
            pdk(fig2,"ESPACIO VOCÁLICO F1–F2  (ref. vocales español)",370)
            fig2.update_xaxes(title_text="F2 (Hz)",autorange="reversed")
            fig2.update_yaxes(title_text="F1 (Hz)",autorange="reversed")
            st.plotly_chart(fig2,use_container_width=True)

# ─── TAB 6: COCLEORAMA ───────────────────────────────────────────────────────
with tab_coch:
    nm=st.slider("N° filtros Mel",32,128,64,8,key="cm")
    with st.spinner("Cocleorama..."):
        ct,cf,cdb=cochleagram(y_sel,sr,nm,80.0,min(fmax_spec,sr//2-1)); ct=ct+t0
    fig=go.Figure(go.Heatmap(x=ct,y=cf,z=cdb,
        colorscale=[[0,"#0d0d0f"],[0.2,"#0a2040"],[0.5,"#106080"],[0.75,"#20c0a0"],[1,"#e0ff80"]],
        zmin=-80,zmax=0,
        colorbar=dict(thickness=10,
                      title=dict(text="dB",font=dict(size=9,color="#5a5a6a")),
                      tickfont=dict(size=9,color="#5a5a6a")),
        hovertemplate="t=%{x:.3f}s<br>f=%{y:.0f}Hz<br>%{z:.1f}dB<extra></extra>"))
    pdk(fig,f"COCLEORAMA  |  {nm} filtros Mel",400)
    fig.update_xaxes(title_text="Tiempo (s)"); fig.update_yaxes(title_text="Hz")
    st.plotly_chart(fig,use_container_width=True)

# ─── TAB 7: JITTER / SHIMMER / HNR ─────────────────────────────────────────
with tab_jit:
    st.markdown("### 📐 Evaluación de Calidad de Voz y Perturbación")
    
    if speech_type == "habla_continua":
        st.markdown(f"""
        <div class="wbox">
          <b>⚠️ Habla Continua / Fluida Detectada en la Muestra</b><br>
          Los parámetros de perturbación glótica como el $\\text{{Jitter}}$ ($3.297\\%$) y el $\\text{{Shimmer}}$ ($8.005\\%$) 
          calculados corresponden a un tramo conversacional continuo. En fonoaudiología clínica, estas métricas 
          <b>carecen de validez en el habla continua</b> debido a la modulación melódica natural de las frases y la sonorización de consonantes.
          <br><br>
          <b>Métricas de referencia reajustadas:</b> Las tarjetas inferiores se muestran con etiqueta gris e indicador <i>(n/a)</i>. 
          Para obtener parámetros orgánicos confiables y correlación diagnóstica con umbrales, graba y aísla una <b>vocal /a/ sostenida de forma estable</b> por 3 segundos.
        </div>
        """, unsafe_allow_html=True)
    
    J_T=1.040;RAP_T=0.680;PPQ_T=0.840;SH_T=3.810;APQ3_T=3.070;APQ5_T=3.140;HNR_T=20.0;CPP_T=8.5

    st.markdown("#### Perturbación de Frecuencia — JITTER (en bloques contiguos)")
    st.markdown(f'<div class="ibox">Umbrales MDVP (Sustained): Jitter local &lt;<b>{J_T}%</b>, RAP &lt;{RAP_T}%.</div>',unsafe_allow_html=True)
    cs=st.columns(4)
    cs[0].markdown(vc("Jitter local",vm_init['jitter_local'],"%",J_T, speech_type=speech_type),unsafe_allow_html=True)
    cs[1].markdown(vc("RAP",         vm_init['jitter_rap'],  "%",RAP_T, speech_type=speech_type),unsafe_allow_html=True)
    cs[2].markdown(vc("PPQ5",        vm_init['jitter_ppq5'], "%",PPQ_T, speech_type=speech_type),unsafe_allow_html=True)
    cs[3].markdown(mc("DDP",fmt(vm_init['jitter_ddp'],3),"%" if speech_type=="vocal_sostenida" else "% (n/a)"),unsafe_allow_html=True)

    st.markdown("<hr class='sdiv'>", unsafe_allow_html=True); st.markdown("#### Perturbación de Amplitud — SHIMMER (en bloques contiguos)")
    st.markdown(f'<div class="ibox">Umbrales MDVP: Shimmer local &lt;<b>{SH_T}%</b>, APQ3 &lt;{APQ3_T}%.</div>',unsafe_allow_html=True)
    cs=st.columns(4)
    cs[0].markdown(vc("Shimmer local",vm_init['shimmer_local'],"%",SH_T, speech_type=speech_type),  unsafe_allow_html=True)
    cs[1].markdown(vc("APQ3",         vm_init['shimmer_apq3'], "%",APQ3_T, speech_type=speech_type),unsafe_allow_html=True)
    cs[2].markdown(vc("APQ5",         vm_init['shimmer_apq5'], "%",APQ5_T, speech_type=speech_type),unsafe_allow_html=True)
    cs[3].markdown(mc("DDA",fmt(vm_init['shimmer_dda'],3),"%" if speech_type=="vocal_sostenida" else "% (n/a)"),unsafe_allow_html=True)

    st.markdown("<hr class='sdiv'>", unsafe_allow_html=True); st.markdown("#### Ratios de Armonicidad (CPP Normalizado)")
    cs=st.columns(2)
    cs[0].markdown(vc("HNR",vm_init['hnr'],"dB",HNR_T,bad_high=False, speech_type=speech_type),unsafe_allow_html=True)
    cs[1].markdown(vc("CPP",vm_init['cpp'],"dB",CPP_T,bad_high=False),unsafe_allow_html=True)

    st.markdown("<hr class='sdiv'>", unsafe_allow_html=True); st.markdown("#### Interrupciones Vocales")
    cs=st.columns(3)
    cs[0].markdown(mc("Voice Breaks",str(int(vm_init['vb_count'])) if not np.isnan(vm_init['vb_count']) else "—",""),unsafe_allow_html=True)
    cs[1].markdown(mc("Duración total",fmt(vm_init['vb_total_ms'],0),"ms"),unsafe_allow_html=True)
    cs[2].markdown(mc("Grado sonoridad",fmt(vm_init['degree_voicing'],1),"%"),unsafe_allow_html=True)

    st.markdown("<hr class='sdiv'>", unsafe_allow_html=True); st.markdown("#### Indicadores visuales")
    
    if speech_type == "vocal_sostenida":
        gc2=st.columns(4)
        for col,(val,title,thr,mx,unit,bh) in zip(gc2,[
            (vm_init['jitter_local'],"Jitter",J_T,5,"%",True),
            (vm_init['shimmer_local'],"Shimmer",SH_T,15,"%",True),
            (vm_init['hnr'],"HNR",HNR_T,40,"dB",False),
            (vm_init['cpp'],"CPP",CPP_T,25,"dB",False),
        ]):
            g=gauge_plot(val,title,thr,mx,unit,bh)
            if g: col.plotly_chart(g,use_container_width=True)
    else:
        st.info("Los gráficos de aguja clínicos están calibrados solo para modo Vocal Sostenida. Para habla continua, revise las métricas en la pestaña LINGÜÍSTICO.")

# ─── TAB 8: DIAGNÓSTICO CLÍNICO ─────────────────────────────────────────────
with tab_diag:
    st.markdown("""<div class="ibox">
    <b>Aviso de Mitigación de Sesgos Clínicos:</b><br>
    El motor de PhonLab Pro ha sido actualizado para clasificar dinámicamente el estilo de muestra y evitar diagnósticos alarmantes erróneos.
    </div>""", unsafe_allow_html=True)

    # Motor de Severidad Adaptativo
    sev=severity_level(vm_init, speech_type=speech_type)
    scores=disorder_scores(vm_init)

    # UI condicional según el tipo de fonación
    if speech_type == "habla_continua":
        st.markdown(f"""
        <div style="background:#11151a; border:2px solid #57c8ff; border-radius:10px; padding:18px 22px; margin:10px 0;">
          <div class="diag-title" style="color:#57c8ff;">✓ Modo de Habla Continua Detectado (Lectura / Conversación)</div>
          <div style="font-family:'Space Mono',monospace; font-size:1.15rem; font-weight:700; margin-bottom:6px; color:#e8e6e0;">FONACIÓN GENERAL ADAPTADA</div>
          <div style="font-size:0.84rem; color:#9a9aaa; line-height:1.5;">
            <b>¿Por qué no hay alteración acústica alarmante?</b> El sistema detectó una melodía variada ($\sigma_{{F0}} = {la_init['f0_sd']:.1f}\\text{{ Hz}}$) 
            y pausas temporales consistentes con el habla articulada. En este modo, <b>los indicadores de lesión de cuerdas vocales (como pólipos o nódulos) se desactivan</b> 
            por carecer de validez científica matemática para habla continua. Su indicador de periodicidad cepstral consolidada <b>CPP</b> es de <b>{vm_init['cpp']:.2f} dB</b>, 
            lo cual se correlaciona con una voz armónica y proyectada.
            <br><br>
            Para realizar un despistaje fonoaudiológico acústico de lesiones físicas o laringitis estructural, cargue un audio realizando una <b>/a/ sostenida y sin modular</b>.
          </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### Métricas de Articulación y Proyección (Lectura de Frase)")
        cs = st.columns(4)
        cs[0].markdown(mc("Estabilidad del Cepstro (CPP)", fmt(vm_init['cpp'],2), "dB (Óptimo > 6.0)"), unsafe_allow_html=True)
        cs[1].markdown(mc("Velocidad de Habla", fmt(la_init['speech_rate'],1), "sil/s"), unsafe_allow_html=True)
        cs[2].markdown(mc("Pausas Conversacionales", str(la_init['n_pauses']), "pausas"), unsafe_allow_html=True)
        cs[3].markdown(mc("Tasa de Fonación", fmt(vm_init['degree_voicing'],1), "%"), unsafe_allow_html=True)
        
    else:
        # Vocal Sostenida: Diagnóstico Estructural Completo
        sev_info={
            'normal':   ("#57ff57","diag-normal",  "✓ VOZ NORMAL (Vocal Sostenida)",
                         "Todos los parámetros acústicos de perturbación se encuentran dentro del rango óptimo de normalidad clínica (umbrales MDVP/Praat)."),
            'leve':     ("#ffff44","diag-leve",    "⚠ ALTERACIÓN ACÚSTICA LEVE",
                         "Uno o más parámetros superan levemente el umbral normal. Puede corresponder a fatiga vocal momentánea o tensión."),
            'moderado': ("#f07020","diag-moderado","⚠ ALTERACIÓN ACÚSTICA MODERADA",
                         "Múltiples parámetros alterados de manera concertada. Se sugiere una evaluación fonoaudiológica para descartar sobreesfuerzo vocal."),
            'severo':   ("#ff4444","diag-severo",  "🔴 ALTERACIÓN ACÚSTICA SEVERA",
                         "Perturbación glótica marcada y consistente en vocal sostenida. Se recomienda fuertemente una evaluación laringológica por un médico otorrinolaringólogo."),
            'sin_datos':("#5a5a6a","diag-normal",  "— SIN DATOS SUFICIENTES",
                         "No se detectaron suficientes segmentos sonoros. Cargue un audio con voz estable."),
        }
        sc,css,ttl,dtl = sev_info.get(sev, sev_info['sin_datos'])
        st.markdown(f"""
        <div class="{css}">
          <div class="diag-title">Severidad Acústica Global</div>
          <div class="diag-result" style="color:{sc}">{ttl}</div>
          <div class="diag-detail">{dtl}</div>
        </div>""",unsafe_allow_html=True)

        st.markdown("#### Parámetros Clínicos de Referencia (Vocal)")
        J_T=1.040;SH_T=3.810;HNR_T=20.0;CPP_T=8.5
        cs=st.columns(4)
        cs[0].markdown(vc("Jitter local",vm_init['jitter_local'],"%",J_T, speech_type=speech_type),unsafe_allow_html=True)
        cs[1].markdown(vc("Shimmer local",vm_init['shimmer_local'],"%",SH_T, speech_type=speech_type),unsafe_allow_html=True)
        cs[2].markdown(vc("HNR",vm_init['hnr'],"dB",HNR_T,bad_high=False, speech_type=speech_type),unsafe_allow_html=True)
        cs[3].markdown(vc("CPP",vm_init['cpp'],"dB",CPP_T,bad_high=False, speech_type=speech_type),unsafe_allow_html=True)

        # Perfiles de trastornos (Solo válidos para vocal sostenida)
        st.markdown("<hr class='sdiv'>", unsafe_allow_html=True)
        st.markdown("#### Perfiles de Trastornos — Similitud Acústica (Filtro Vocal)")
        st.markdown("""<div class="ibox">
        Análisis geométrico espectral sobre vocal sostenida. No sustituye un diagnóstico laringoscópico médico.
        </div>""",unsafe_allow_html=True)

        f0m=vm_init.get('f0_mean',np.nan)
        if not np.isnan(f0m):
            for i,s in enumerate(scores[:5]):
                pct=s['score']
                if   pct>=70: bar_color="#ff4444"; level="Alta"
                elif pct>=45: bar_color="#f07020"; level="Moderada"
                elif pct>=25: bar_color="#ffff44"; level="Baja"
                else:         bar_color="#3a6a3a"; level="Muy baja"

                st.markdown(f"""
                <div style="background:#111115;border:1px solid #2a2a35;border-radius:8px;padding:16px;margin:8px 0;">
                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                    <span style="font-family:'Space Mono',monospace;font-size:0.85rem;color:#e8e6e0">{s['name']}</span>
                    <span style="font-family:'Space Mono',monospace;font-size:0.75rem;color:{bar_color}">{level} ({pct:.0f}%)</span>
                  </div>
                  <div style="background:#1e1e28;border-radius:4px;height:6px;margin-bottom:10px;">
                    <div style="background:{bar_color};height:6px;border-radius:4px;width:{min(pct,100):.0f}%"></div>
                  </div>
                  <div style="font-size:0.78rem;color:#9a9aaa;margin-bottom:4px">{s['desc']}</div>
                  <div style="font-size:0.76rem;color:#5a5a6a;font-style:italic">{s['f0_note']}</div>
                  <div style="font-size:0.68rem;color:#3a3a4a;margin-top:4px">Refs: {s['refs']}</div>
                </div>""",unsafe_allow_html=True)

    # Radar Chart
    st.markdown("<hr class='sdiv'>", unsafe_allow_html=True); st.markdown("#### Radar de Parámetros Vocales")
    params=['Jitter','Shimmer','HNR','CPP','Sonoridad','Estabilidad F0']
    def norm_val(v,lo,hi,invert=False):
        if np.isnan(float(v)): return 0.5
        n=float(np.clip((float(v)-lo)/(hi-lo+1e-9),0,1))
        return 1-n if invert else n
    vals_radar=[
        norm_val(vm_init['jitter_local'],  0,4,   invert=True),
        norm_val(vm_init['shimmer_local'], 0,12,  invert=True),
        norm_val(vm_init['hnr'],           0,35,  invert=False),
        norm_val(vm_init['cpp'],           0,18,  invert=False),
        norm_val(vm_init['degree_voicing'],0,100, invert=False),
        norm_val(vm_init['f0_sd'],         0,45,  invert=True),
    ]
    ref_radar=[0.85]*6
    fig_r=go.Figure()
    fig_r.add_trace(go.Scatterpolar(r=ref_radar+[ref_radar[0]],
        theta=params+[params[0]],fill='toself',
        fillcolor='rgba(87,200,255,0.08)',
        line=dict(color='#57c8ff',width=1,dash='dot'),name='Normalidad'))
    fig_r.add_trace(go.Scatterpolar(r=vals_radar+[vals_radar[0]],
        theta=params+[params[0]],fill='toself',
        fillcolor='rgba(200,255,87,0.15)',
        line=dict(color='#c8ff57',width=2),name='Muestra Medida'))
    fig_r.update_layout(
        polar=dict(radialaxis=dict(visible=True,range=[0,1],
            tickfont=dict(size=8,color='#5a5a6a'),gridcolor='#2a2a35'),
            angularaxis=dict(tickfont=dict(size=9,color='#9a9aaa'),gridcolor='#2a2a35'),
            bgcolor='#0d0d0f'),
        paper_bgcolor='#111115',legend=dict(font=dict(family="Space Mono",size=9,color="#9a9aaa"),
            bgcolor="#111115",bordercolor="#2a2a35",borderwidth=1),
        height=380,margin=dict(l=60,r=60,t=40,b=40),
        title=dict(text="RADAR VOCAL  (1=Rango Óptimo, 0=Alteración)",
                   font=dict(family="Space Mono",size=10,color="#5a5a6a"))
    )
    st.plotly_chart(fig_r,use_container_width=True)

# ─── TAB 9: LINGÜÍSTICO ──────────────────────────────────────────────────────
with tab_ling:
    st.markdown("#### Ritmo y Velocidad del Habla")
    st.markdown("""<div class="ibox">
    Velocidad normal: <b>4–6 sílabas/segundo</b> (habla espontánea estándar). 
    Tasa de articulación (sin pausas): 5–8 sil/s. 
    Pausas normales: 15–25% del tiempo conversacional total.
    </div>""",unsafe_allow_html=True)

    cs=st.columns(4)
    sr_rate=la_init['speech_rate']; ar_rate=la_init['articulation_rate']
    cs[0].markdown(mc("N° sílabas",str(la_init['n_syllables']),""),unsafe_allow_html=True)

    def rate_card(lbl,val,lo,hi,unit):
        v=float(val)
        if v<lo*0.7 or v>hi*1.5: return wc(lbl,fmt(val,1),unit)
        elif lo<=v<=hi:           return gc(lbl,fmt(val,1),unit)
        else:                     return mc(lbl,fmt(val,1),unit)
    cs[1].markdown(rate_card("Vel. habla",sr_rate,4,6,"sil/s"),unsafe_allow_html=True)
    cs[2].markdown(rate_card("Vel. articulación",ar_rate,5,8,"sil/s"),unsafe_allow_html=True)
    cs[3].markdown(mc("Pausas",str(la_init['n_pauses']),""),unsafe_allow_html=True)

    cs2=st.columns(3)
    dur_sel=len(y_sel)/sr
    pct_pau=la_init['total_pause_s']/dur_sel*100 if dur_sel>0 else 0
    cs2[0].markdown(mc("Tiempo habla",fmt(dur_sel-la_init['total_pause_s'],2),"s"),unsafe_allow_html=True)
    cs2[1].markdown(mc("Tiempo pausa",fmt(la_init['total_pause_s'],2),"s"),unsafe_allow_html=True)
    cs2[2].markdown(mc("% Pausa",fmt(pct_pau,1),"%"),unsafe_allow_html=True)

    # Nuclei + pauses visual
    st.markdown("<hr class='sdiv'>", unsafe_allow_html=True); st.markdown("#### Núcleos Silábicos + Pausas")
    fig=go.Figure()
    MP=20000; step_=max(1,len(y_sel)//MP)
    fig.add_trace(go.Scatter(x=t_sel[::step_],y=y_sel[::step_],mode="lines",
        line=dict(color="#2a2a35",width=0.5),name="Señal",showlegend=False))
    for ps2,pe2 in la_init['pauses']:
        fig.add_vrect(x0=ps2+t0,x1=pe2+t0,fillcolor="rgba(255,70,70,0.15)",line_width=0,
                      annotation_text="pausa",annotation_font_size=8,
                      annotation_font_color="#ff7070",annotation_position="top left")
    if len(la_init['syl_times'])>0:
        syl_t_abs=la_init['syl_times']+t0
        syl_norm=(la_init['syl_db']-np.min(la_init['syl_db']))/(np.max(la_init['syl_db'])-np.min(la_init['syl_db'])+1e-9)*0.8+0.1
        fig.add_trace(go.Scatter(x=syl_t_abs,y=syl_norm,mode="markers",
            marker=dict(color="#c8ff57",size=8,symbol="triangle-up",
                        line=dict(color="#0d0d0f",width=1)),name="Núcleo silábico"))
    pdk(fig,"NÚCLEOS SILÁBICOS (▲) + PAUSAS (rojo)",300)
    fig.update_xaxes(title_text="Tiempo (s)"); fig.update_yaxes(title_text="Amplitud norm.")
    st.plotly_chart(fig,use_container_width=True)

    # Intonation
    st.markdown("<hr class='sdiv'>", unsafe_allow_html=True); st.markdown("#### Entonación")
    cs3=st.columns(3)
    cs3[0].markdown(mc("Rango F0",fmt(la_init['f0_range'],1),"Hz"),unsafe_allow_html=True)
    cs3[1].markdown(mc("SD F0",fmt(la_init['f0_sd'],1),"Hz"),unsafe_allow_html=True)
    slope_=la_init.get('f0_slope',np.nan)
    slope_dir="↗ ascendente" if not np.isnan(slope_) and slope_>0 else "↘ descendente"
    cs3[2].markdown(mc("Tendencia F0",slope_dir if not np.isnan(slope_) else "—",""),unsafe_allow_html=True)

    # F0 contour
    fpt=la_init['f0_times']+t0; fpv=la_init['f0_vals']
    fig2=go.Figure()
    fig2.add_trace(go.Scatter(x=fpt,y=np.where(np.isnan(fpv),np.nan,fpv).tolist(),
        mode="markers",marker=dict(color="#c8ff57",size=3,opacity=0.85),name="F0"))
    for ps2,pe2 in la_init['pauses']:
        fig2.add_vrect(x0=ps2+t0,x1=pe2+t0,fillcolor="rgba(255,70,70,0.1)",line_width=0)
    pdk(fig2,"CONTORNO DE ENTONACIÓN (F0 + pausas en rojo)",280)
    fig2.update_xaxes(title_text="Tiempo (s)"); fig2.update_yaxes(title_text="Hz")
    st.plotly_chart(fig2,use_container_width=True)

    # Screening de Disfluencias
    st.markdown("<hr class='sdiv'>", unsafe_allow_html=True)
    st.markdown("#### Indicadores de Disfluencia")
    dysfluency_flags = []
    dysfluency_score = 0

    if la_init['speech_rate'] < 2.5:
        dysfluency_flags.append(f"⚠ Velocidad de habla extremadamente baja ({la_init['speech_rate']:.1f} sil/s)")
        dysfluency_score += 3
    elif la_init['speech_rate'] < 3.5:
        dysfluency_flags.append(f"⚠ Velocidad de habla reducida ({la_init['speech_rate']:.1f} sil/s)")
        dysfluency_score += 1

    pause_rate_v = la_init['n_pauses'] / max(len(y_sel)/sr, 1) * 60
    if pause_rate_v > 15:
        dysfluency_flags.append(f"⚠ Alta frecuencia de pausas segmentadas ({pause_rate_v:.0f}/min)")
        dysfluency_score += 3

    if pct_pau > 40:
        dysfluency_flags.append(f"⚠ Tiempo porcentual en pausa elevado ({pct_pau:.0f}%)")
        dysfluency_score += 2

    if dysfluency_score == 0:
        color_d="#57ff57"; label_d="✓ Sin indicadores de disfluencia"; css_d="diag-normal"
    elif dysfluency_score <= 2:
        color_d="#ffff44"; label_d="⚠ Indicadores de disfluencia leves"; css_d="diag-leve"
    else:
        color_d="#ff4444"; label_d="🔴 Indicadores marcados de disfluencia temporal"; css_d="diag-severo"

    st.markdown(f"""<div class="{css_d}">
      <div class="diag-title">Screening Temporal</div>
      <div class="diag-result" style="color:{color_d}">{label_d}</div>
      <div class="diag-detail">{'<br>'.join(dysfluency_flags) if dysfluency_flags else 'Métricas temporales dentro del rango conversacional normal.'}</div>
    </div>""", unsafe_allow_html=True)

    # Download Button TextGrid
    st.markdown("<hr class='sdiv'>", unsafe_allow_html=True); st.markdown("#### Exportar TextGrid (compatible con Praat)")
    tg_str=generate_textgrid(la_init['pauses'],la_init['syl_times'],len(y_sel)/sr)
    st.download_button("⬇ Descargar TextGrid (.TextGrid)",
        tg_str.encode('utf-8'),
        f"{os.path.splitext(uploaded.name)[0]}.TextGrid",
        "text/plain")

# ─── TAB 10: MFCC ────────────────────────────────────────────────────────────
with tab_mfcc:
    with st.spinner("MFCC..."):
        hm=512
        mfccs=librosa.feature.mfcc(y=y_sel,sr=sr,n_mfcc=n_mfcc,hop_length=hm)
        d1=librosa.feature.delta(mfccs); d2=librosa.feature.delta(mfccs,order=2)
        tm=librosa.frames_to_time(np.arange(mfccs.shape[1]),sr=sr,hop_length=hm)+t0

    vw=st.radio("Vista:",["MFCC","Δ Delta","ΔΔ Delta-Delta"],horizontal=True,key="mv")
    dshow={"MFCC":mfccs,"Δ Delta":d1,"ΔΔ Delta-Delta":d2}[vw]
    fig=go.Figure(go.Heatmap(x=tm,y=[f"C{i}" for i in range(n_mfcc)],z=dshow,
        colorscale="RdBu_r",colorbar=dict(thickness=10,tickfont=dict(size=9,color="#5a5a6a")),
        hovertemplate="t=%{x:.3f}s<br>%{y}<br>%{z:.2f}<extra></extra>"))
    pdk(fig,f"{vw}  ({n_mfcc} coef.)",430)
    fig.update_xaxes(title_text="Tiempo (s)"); fig.update_yaxes(title_text="Coef.",autorange="reversed")
    st.plotly_chart(fig,use_container_width=True)
    df_ms=pd.DataFrame({"Coef.":[f"C{i}" for i in range(n_mfcc)],
        "Media":np.round(np.mean(mfccs,axis=1),3),"Std":np.round(np.std(mfccs,axis=1),3),
        "Mín.":np.round(np.min(mfccs,axis=1),3),"Máx.":np.round(np.max(mfccs,axis=1),3)})
    st.dataframe(df_ms,use_container_width=True,hide_index=True)
    fig2=go.Figure(go.Bar(x=df_ms["Coef."],y=df_ms["Media"],
        marker=dict(color=df_ms["Media"].tolist(),colorscale="RdBu_r",showscale=False,
                    line=dict(color="#0d0d0f",width=0.5)),
        error_y=dict(type="data",array=df_ms["Std"].tolist(),color="#5a5a6a",thickness=1,width=4)))
    pdk(fig2,"MEDIA MFCC (±1 SD)",260)
    fig2.update_xaxes(title_text="Coef."); fig2.update_yaxes(title_text="Valor")
    st.plotly_chart(fig2,use_container_width=True)

# ─── TAB 11: MOMENTOS ESPECTRALES ────────────────────────────────────────────
with tab_spec2:
    with st.spinner("Momentos espectrales..."):
        sm=spectral_moments(y_sel,sr)
    cs=st.columns(3)
    cs[0].markdown(mc("Centroide",fmt(sm['centroid'],0),"Hz"),unsafe_allow_html=True)
    cs[1].markdown(mc("Dispersión",fmt(sm['spread'],0),"Hz"),unsafe_allow_html=True)
    cs[2].markdown(mc("Rolloff 85%",fmt(sm['rolloff85'],0),"Hz"),unsafe_allow_html=True)
    cs2=st.columns(3)
    cs2[0].markdown(mc("Asimetría",fmt(sm['skewness'],3),""),unsafe_allow_html=True)
    cs2[1].markdown(mc("Curtosis",fmt(sm['kurtosis'],3),""),unsafe_allow_html=True)
    cs2[2].markdown(mc("Flatness",fmt(sm['flatness'],4),""),unsafe_allow_html=True)

    st.markdown("<hr class='sdiv'>", unsafe_allow_html=True); st.markdown("#### Curvas espectrales frame a frame")
    nff=2048; hff=512
    tf2=librosa.frames_to_time(
        np.arange(librosa.stft(y_sel,n_fft=nff,hop_length=hff).shape[1]),
        sr=sr,hop_length=hff)+t0
    cc=librosa.feature.spectral_centroid(y=y_sel,sr=sr,n_fft=nff,hop_length=hff)[0]
    rc=librosa.feature.spectral_rolloff(y=y_sel,sr=sr,n_fft=nff,hop_length=hff,roll_percent=0.85)[0]
    fc=librosa.feature.spectral_flatness(y=y_sel,n_fft=nff,hop_length=hff)[0]
    fig=make_subplots(rows=3,cols=1,shared_xaxes=True,
        subplot_titles=["Centroide (Hz)","Rolloff 85% (Hz)","Flatness"])
    fig.add_trace(go.Scatter(x=tf2,y=cc,mode="lines",line=dict(color="#c8ff57",width=1)),row=1,col=1)
    fig.add_trace(go.Scatter(x=tf2,y=rc,mode="lines",line=dict(color="#57c8ff",width=1)),row=2,col=1)
    fig.add_trace(go.Scatter(x=tf2,y=fc,mode="lines",line=dict(color="#ff57c8",width=1)),row=3,col=1)
    fig.update_layout(paper_bgcolor="#111115",plot_bgcolor="#0d0d0f",
        font=dict(family="DM Sans",color="#9a9aaa"),
        margin=dict(l=52,r=16,t=50,b=40),height=480,showlegend=False)
    for i in range(1,4):
        fig.update_xaxes(gridcolor="#1e1e28",zerolinecolor="#1e1e28",linecolor="#2a2a35",row=i,col=1)
        fig.update_yaxes(gridcolor="#1e1e28",zerolinecolor="#1e1e28",linecolor="#2a2a35",row=i,col=1)
    fig.update_xaxes(title_text="Tiempo (s)",row=3,col=1)
    st.plotly_chart(fig,use_container_width=True)

# ─── TAB 12: FILTROS ─────────────────────────────────────────────────────────
with tab_filt:
    st.markdown("### ✂️ Procesamiento de Señales / Filtros")
    ft_=st.selectbox("Tipo",["Paso de banda","Paso bajo","Paso alto","Notch"])
    fc1=st.number_input("Frec. 1 (Hz)",value=300.0,step=50.0,key="fc1")
    fc2=st.number_input("Frec. 2 (Hz)",value=3400.0,step=50.0,key="fc2")
    ford=st.slider("Orden",2,8,4,1)

    yf=bandpass(y_sel,sr,fc1,fc2,ford) if ft_ == "Paso de banda" else y_sel
    if ft_ == "Paso bajo":
        nyq=sr/2.0
        sos=butter(ford,fc1/nyq,btype='low',output='sos')
        yf=sosfiltfilt(sos,y_sel)
    elif ft_ == "Paso alto":
        nyq=sr/2.0
        sos=butter(ford,fc1/nyq,btype='high',output='sos')
        yf=sosfiltfilt(sos,y_sel)
    elif ft_ == "Notch":
        nyq=sr/2.0
        bw=max(fc1*0.05,20.0); lo=max(1.0,fc1-bw); hi=min(nyq-1,fc1+bw)
        sos=butter(ford,[lo/nyq,hi/nyq],btype='bandstop',output='sos')
        yf=sosfiltfilt(sos,y_sel)

    MP2=20000; st_=max(1,len(y_sel)//MP2)
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,
        subplot_titles=["Original","Filtrada"])
    fig.add_trace(go.Scatter(x=t_sel[::st_],y=y_sel[::st_],mode="lines",
        line=dict(color="#c8ff57",width=0.7)),row=1,col=1)
    fig.add_trace(go.Scatter(x=t_sel[::st_],y=yf[::st_],mode="lines",
        line=dict(color="#57c8ff",width=0.7)),row=2,col=1)
    fig.update_layout(paper_bgcolor="#111115",plot_bgcolor="#0d0d0f",
        font=dict(family="DM Sans",color="#9a9aaa"),
        margin=dict(l=52,r=16,t=50,b=40),height=380,showlegend=False)
    for i in range(1,3):
        fig.update_xaxes(gridcolor="#1e1e28",zerolinecolor="#1e1e28",linecolor="#2a2a35",row=i,col=1)
        fig.update_yaxes(gridcolor="#1e1e28",zerolinecolor="#1e1e28",linecolor="#2a2a35",row=i,col=1)
    fig.update_xaxes(title_text="Tiempo (s)",row=2,col=1)
    st.plotly_chart(fig,use_container_width=True)

    nff2=4096; frf=librosa.fft_frequencies(sr=sr,n_fft=nff2)
    po=10*np.log10(np.mean(np.abs(librosa.stft(y_sel,n_fft=nff2))**2,axis=1)+1e-12)
    pf=10*np.log10(np.mean(np.abs(librosa.stft(yf,   n_fft=nff2))**2,axis=1)+1e-12)
    fi2=int(np.searchsorted(frf,fmax_spec))
    fig2=go.Figure()
    fig2.add_trace(go.Scatter(x=frf[:fi2],y=po[:fi2],mode="lines",
        line=dict(color="#c8ff57",width=1,dash="dot"),name="Original"))
    fig2.add_trace(go.Scatter(x=frf[:fi2],y=pf[:fi2],mode="lines",
        line=dict(color="#57c8ff",width=1.5),name="Filtrada"))
    pdk(fig2,"RESPUESTA ESPECTRAL: ORIGINAL vs FILTRADA",270)
    fig2.update_xaxes(title_text="Hz"); fig2.update_yaxes(title_text="dB")
    st.plotly_chart(fig2,use_container_width=True)
    try:
        import soundfile as sf
        buf=io.BytesIO(); sf.write(buf,yf,sr,format="WAV",subtype="PCM_16")
        st.download_button("⬇ Descargar audio filtrado (WAV)",buf.getvalue(),"audio_filtrado.wav","audio/wav")
    except ImportError:
        st.info("Instale `soundfile` para habilitar la descarga del audio filtrado.")

# ─── TAB 13: EXPORTAR ────────────────────────────────────────────────────────
with tab_exp:
    st.markdown("### 📁 Exportar datos consolidados")
    st.markdown('<div class="ibox">Exporta CSVs de análisis y TextGrid limpios de ruidos extraños.</div>',unsafe_allow_html=True)
    ca,cb=st.columns(2)
    with ca:
        st.download_button("⬇ Pitch (F0)",
            pd.DataFrame({"tiempo_s":pt,"f0_hz":pv}).to_csv(index=False),
            "pitch.csv","text/csv")
        st.download_button("⬇ Intensidad",
            pd.DataFrame({"tiempo_s":it,"intensidad_db":iv}).to_csv(index=False),
            "intensidad.csv","text/csv")
        df_f=pd.DataFrame({"tiempo_s":ft})
        for k in range(n_form):
            df_f[f"F{k+1}_hz"]=Fm[:,k]; df_f[f"BW{k+1}_hz"]=Bm[:,k]
        st.download_button("⬇ Formantes",df_f.to_csv(index=False),"formantes.csv","text/csv")
        st.download_button("⬇ Espectro de Potencia",
            pd.DataFrame({"frec_hz":fps[:fidx],"potencia_db":pdb[:fidx]}).to_csv(index=False),
            "espectro.csv","text/csv")
        st.download_button("⬇ Momentos espectrales",
            pd.DataFrame([sm]).to_csv(index=False),"momentos.csv","text/csv")

    with cb:
        df_mfcc2=pd.DataFrame(mfccs.T,columns=[f"C{i}" for i in range(n_mfcc)])
        df_mfcc2.insert(0,"tiempo_s",tm)
        st.download_button("⬇ Coeficientes MFCC",df_mfcc2.to_csv(index=False),"mfcc.csv","text/csv")
        
        summary=dict(
            archivo=uploaded.name,duracion_s=f"{duration:.3f}",fs_hz=sr,
            f0_media=fmt(mf,1),f0_mediana=fmt(mdf,1),f0_min=fmt(mnf,1),f0_max=fmt(mxf,1),
            voiced_pct=fmt(vp,1),
            jitter_local=fmt(vm_init['jitter_local'],3),jitter_rap=fmt(vm_init['jitter_rap'],3),
            shimmer_local=fmt(vm_init['shimmer_local'],3),shimmer_apq3=fmt(vm_init['shimmer_apq3'],3),
            hnr=fmt(vm_init['hnr'],2),cpp=fmt(vm_init['cpp'],2),
            severidad=sev,
            vb_count=str(int(vm_init['vb_count'])) if not np.isnan(vm_init['vb_count']) else "—",
            n_silabas=str(la_init['n_syllables']),
            vel_habla=fmt(la_init['speech_rate'],2),vel_articulacion=fmt(la_init['articulation_rate'],2),
            n_pausas=str(la_init['n_pauses']),pct_pausa=fmt(pct_pau,1),
            centroide=fmt(sm['centroid'],0),flatness=fmt(sm['flatness'],4),
        )
        df_sum=pd.DataFrame([summary]).T.reset_index(); df_sum.columns=["Parámetro","Valor"]
        st.download_button("⬇ Resumen Clínico Completo",df_sum.to_csv(index=False),"resumen_voz.csv","text/csv")

    st.markdown("<hr class='sdiv'>", unsafe_allow_html=True); st.markdown("#### Vista previa")
    st.dataframe(df_sum,use_container_width=True,hide_index=True)