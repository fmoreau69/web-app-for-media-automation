"""
Détection de freinage brusque — portage SALSA (`ComputeAccClean` + `identifyBrakeSignal`
+ `extractBrake` + `calculatePlage`), capability-first.

Chaîne : nettoyage de l'accéléro longitudinal → détection des plages de décélération
sous seuil, graduées en 3 niveaux de sévérité (modéré → urgence) → events.

⚠️ Les seuils de SALSA n'étaient PAS dans les .m (dans le modèle compilé) : les défauts
ci-dessous (m/s²) sont des a-priori RAISONNABLES à RECALIBRER sur données Navya réelles,
et dépendent de l'orientation physique de l'accéléromètre (X = longitudinal, décél. < 0).
"""
from __future__ import annotations

import numpy as np

from ..data_types import DataType, TypedFrame
from ..function_catalog import (FunctionSpec, PortSpec, ParamSpec,
                                FunctionCategory, register)


def _moving_average(x, n):
    """Moyenne glissante n points (comme lissageSig), même longueur (bords rognés)."""
    if n <= 1:
        return np.asarray(x, dtype=float)
    k = np.ones(n) / n
    return np.convolve(np.asarray(x, dtype=float), k, mode='same')


def _highpass_dc(x, alpha=0.01):
    """Retrait de biais/dérive DC : x − EMA_lente(x) (équivalent highpass 1er ordre)."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    ema = np.empty_like(x)
    ema[0] = x[0]
    for i in range(1, x.size):
        ema[i] = ema[i - 1] + alpha * (x[i] - ema[i - 1])
    return x - ema


def clean_accel(a, *, smooth_pts=3, flat_var_eps=1e-4, flat_min_run=5, dc_alpha=0.001):
    # dc_alpha bas = baseline LENTE (retire le biais capteur sans atténuer un freinage
    # de ~1 s ; un alpha trop haut, ex. 0.01, écrasait l'événement — mesuré 2026-07-20).
    """Nettoie un signal accéléro : met à zéro les plages « capteur figé » (variance
    glissante < eps sur un run assez long), retire le biais DC, puis lisse."""
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return a
    # variance glissante 20 pts (⟨x²⟩ − ⟨x⟩²)
    w = min(20, a.size)
    m1 = _moving_average(a, w)
    m2 = _moving_average(a * a, w)
    var = np.maximum(m2 - m1 * m1, 0.0)
    flat = var < flat_var_eps
    # runs de "flat" >= flat_min_run → zéro
    out = a.copy()
    i = 0
    n = a.size
    while i < n:
        if flat[i]:
            j = i
            while j < n and flat[j]:
                j += 1
            if j - i >= flat_min_run:
                out[i:j] = 0.0
            i = j
        else:
            i += 1
    out = _highpass_dc(out, dc_alpha)
    return _moving_average(out, smooth_pts)


def _plages(binary, time, min_dur_s):
    """Plages continues où binary==1 de durée >= min_dur_s (comme calculatePlage).
    Retourne [(i0, i1)] indices inclusifs."""
    b = np.asarray(binary).astype(int)
    if b.size == 0:
        return []
    d = np.diff(np.concatenate(([0], b, [0])))
    starts = np.where(d > 0)[0]
    ends = np.where(d < 0)[0] - 1
    out = []
    for i0, i1 in zip(starts, ends):
        if time[i1] - time[i0] >= min_dur_s:
            out.append((i0, i1))
    return out


# Niveaux (borne_max < min(décél sur la plage) <= borne_min), en m/s² (décél. négatives).
# brake2 modéré, brake3 fort, brake4 urgence. À RECALIBRER.
DEFAULT_LEVELS = [
    {'level': 2, 'borne_min': -2.0, 'borne_max': -3.5},
    {'level': 3, 'borne_min': -3.5, 'borne_max': -5.0},
    {'level': 4, 'borne_min': -5.0, 'borne_max': -50.0},
]


def detect_braking(signal: TypedFrame, *, value_field='value', trigger=-1.5,
                   min_dur_s=0.3, smooth_pts=3, recalibrate_levels=None) -> TypedFrame:
    """Détecte les freinages brusques → events (time, duration, type='brake', level,
    peak_decel). `signal` = accéléro longitudinal (time, value). Detector."""
    df = signal.df
    if value_field not in df.columns or 'time' not in df.columns or len(df) == 0:
        import pandas as pd
        return TypedFrame(pd.DataFrame(columns=['time', 'duration', 'type', 'level',
                                                'peak_decel']), DataType.EVENTS)
    time = df['time'].to_numpy(dtype=float)
    a = clean_accel(df[value_field].to_numpy(dtype=float), smooth_pts=smooth_pts)
    levels = recalibrate_levels or DEFAULT_LEVELS
    triggered = (a <= trigger).astype(int)
    events = []
    for (i0, i1) in _plages(triggered, time, min_dur_s):
        peak = float(a[i0:i1 + 1].min())      # décélération la plus forte (la plus négative)
        lvl = None
        for L in levels:
            if L['borne_max'] < peak <= L['borne_min']:
                lvl = L['level']
                break
        if lvl is None:
            continue
        events.append({'time': float(time[i0]), 'duration': float(time[i1] - time[i0]),
                       'type': 'brake', 'level': lvl, 'peak_decel': round(peak, 2)})
    import pandas as pd
    return TypedFrame(pd.DataFrame(events, columns=['time', 'duration', 'type', 'level',
                                                    'peak_decel']),
                      DataType.EVENTS, meta={'trigger': trigger})


SPEC = register(FunctionSpec(
    key='brake_detection',
    name='Freinage brusque',
    description="Détecte les freinages brusques depuis l'accéléromètre longitudinal, "
                "gradués en 3 niveaux de sévérité (modéré→urgence). Seuils à recalibrer.",
    category=FunctionCategory.DETECTOR,
    tags=['timeseries', 'requires-accel', 'needs-calibration'],
    inputs=[
        PortSpec('signal', DataType.SIGNAL, required_fields=['time', 'value'],
                 description='Accélération longitudinale (m/s², décélération < 0).'),
    ],
    outputs=[
        PortSpec('events', DataType.EVENTS,
                 produced_fields=['time', 'duration', 'type', 'level', 'peak_decel']),
    ],
    params=[
        ParamSpec('trigger', 'float', -1.5, -10.0, 0.0, unit='m/s²',
                  description='Seuil de déclenchement (décélération sous laquelle on regarde).'),
        ParamSpec('min_dur_s', 'float', 0.3, 0.0, 5.0, unit='s',
                  description='Durée minimale d\'une plage de freinage.'),
        ParamSpec('smooth_pts', 'int', 3, 1, 15,
                  description='Lissage du signal accéléro (points).'),
    ],
    cost={'cpu_bound': True},
    projects=['ENA'],
    fn=detect_braking,
))
