"""Benchmark di accuratezza, latenza, memoria ed energia di reti GDSR su TOFDSR (e Redwood)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # la cartella benchmark_gdsr
EXTERNAL = ROOT / 'external'                          # repository scaricati
WEIGHTS = ROOT / 'weights'                            # pesi extra (Depth Anything V2)
RESULTS = ROOT / 'results'

# Reti incluse: RMSE pubblicata sul TOFDSR reale (x4) + codice ufficiale disponibile.
# I pesi addestrati su TOFDSR NON sono un requisito: latenza, memoria ed energia non
# dipendono dal valore dei pesi; per le reti senza pesi si riporta la RMSE pubblicata.
REPO_DIRS = {
    'dkn':      EXTERNAL / 'DKN',      # github.com/cvlab-yonsei/dkn
    'fdkn':     EXTERNAL / 'DKN',
    'fdsr':     EXTERNAL / 'FDSR',     # github.com/lingzhi96/RGB-D-D-Dataset (codice in codes/)
    'dctnet':   EXTERNAL / 'DCTNet',   # github.com/Zhaozixiang1228/GDSR-DCTNet
    'suft':     EXTERNAL / 'SUFT',     # github.com/ShiWuxuan/SUFT
    'sgnet':    EXTERNAL / 'SGNet',    # github.com/yanzq95/SGNet
    'dornet_t': EXTERNAL / 'DORNet',   # github.com/yanzq95/DORNet
    'dornet':   EXTERNAL / 'DORNet',
    'ducos':    EXTERNAL / 'DuCos',    # github.com/yanzq95/DuCos
}

MODEL_NAMES = {
    'bicubic': 'Bicubic', 'dkn': 'DKN', 'fdkn': 'FDKN', 'fdsr': 'FDSR', 'dctnet': 'DCTNet',
    'suft': 'SUFT', 'sgnet': 'SGNet', 'dornet_t': 'DORNet-T', 'dornet': 'DORNet', 'ducos': 'DuCos',
}
ALL_MODELS = list(MODEL_NAMES)

# Valori pubblicati su TOFDSR reale x4 (RMSE in cm, parametri in milioni).
#   [DORNet] Wang et al., CVPR 2025, Tab. 1 (metodi riaddestrati sul TOFDSR reale)
#   [DuCos]  Yan et al., ICCV 2025, tabella real-world: modelli addestrati su Hypersim
#            (sintetico) e testati sul TOFDSR reale; 34.38 M include Depth Anything V2.
PUBLISHED = {
    'dkn':      {'rmse_cm': 5.50, 'params_M': 1.16, 'source': 'DORNet (CVPR 2025), Tab. 1'},
    'fdkn':     {'rmse_cm': 5.77, 'params_M': 0.69, 'source': 'DORNet (CVPR 2025), Tab. 1'},
    'fdsr':     {'rmse_cm': 5.03, 'params_M': 0.60, 'source': 'DORNet (CVPR 2025), Tab. 1'},
    'dctnet':   {'rmse_cm': 5.16, 'params_M': 0.48, 'source': 'DORNet (CVPR 2025), Tab. 1'},
    'suft':     {'rmse_cm': 4.37, 'params_M': 22.01, 'source': 'DORNet (CVPR 2025), Tab. 1'},
    'sgnet':    {'rmse_cm': 4.33, 'params_M': 8.97, 'source': 'DORNet (CVPR 2025), Tab. 1'},
    'dornet_t': {'rmse_cm': 4.87, 'params_M': 0.46, 'source': 'DORNet (CVPR 2025), Tab. 1'},
    'dornet':   {'rmse_cm': 4.21, 'params_M': 3.05, 'source': 'DORNet (CVPR 2025), Tab. 1'},
    'ducos':    {'rmse_cm': 4.29, 'params_M': 34.38, 'source': 'DuCos (ICCV 2025), addestrata su Hypersim'},
}


def resolve(p):
    """Percorso dal config: assoluto cosi' com'e', relativo rispetto a benchmark_gdsr."""
    if p is None or str(p).strip() == '':
        return None
    p = Path(str(p)).expanduser()
    return p if p.is_absolute() else (ROOT / p).resolve()
