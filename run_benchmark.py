#!/usr/bin/env python
"""Benchmark GDSR su TOFDSR reale x4: RMSE, latenza, memoria, energia, parametri, FLOPs.

Reti: Bicubic (riferimento), DKN, FDKN, FDSR, DCTNet, SUFT, SGNet, DORNet-T, DORNet, DuCos.

Uso tipico (oppure run_tofdsr.bat / run_tofdsr.sh):
    python run_benchmark.py                                   # protocollo completo, FP32
    python run_benchmark.py --precision fp16                  # seconda configurazione del paper
    python run_benchmark.py --models dornet ducos --limit 50 --energy-seconds 10   # prova veloce
    python run_benchmark.py --dataset redwood                 # Redwood (per la tesi)

Ogni rete gira in un PROCESSO SEPARATO: il picco di memoria di una non sporca la misura
della successiva, e se una fallisce le altre vengono misurate lo stesso. A fine corsa stampa
le tabelle e salva results/<dataset>_<precisione>_<gpu>_<data>.json e .csv.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

try:                                     # console Windows: niente crash sui caratteri accentati
    sys.stdout.reconfigure(errors='replace')
    sys.stderr.reconfigure(errors='replace')
except Exception:
    pass

from gdsrbench import ALL_MODELS  # noqa: E402


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', choices=['tofdsr', 'redwood'], default='tofdsr')
    ap.add_argument('--models', nargs='+', choices=ALL_MODELS, default=ALL_MODELS)
    ap.add_argument('--config', default=str(HERE / 'config.json'))
    ap.add_argument('--device', default='cuda', help='cuda | cuda:1 | cpu')
    ap.add_argument('--limit', type=int, default=0, help='usa solo le prime N immagini (0 = tutte)')
    # protocollo
    ap.add_argument('--precision', choices=['fp32', 'tf32', 'fp16'], default='fp32',
                    help='fp32 = riferimento (TF32 spento); fp16 = autocast; tf32 = default di PyTorch')
    ap.add_argument('--amp', action='store_true', help='alias di --precision fp16')
    ap.add_argument('--warmup', type=int, default=10, help='immagini di warm-up non misurate')
    ap.add_argument('--repeats', type=int, default=3, help='ripetizioni del test set per la latenza')
    ap.add_argument('--energy-seconds', type=float, default=60, help='durata minima del ciclo di energia (0 = niente energia)')
    ap.add_argument('--energy-repeats', type=int, default=3)
    ap.add_argument('--idle-seconds', type=float, default=10, help='durata della misura della potenza a riposo')
    ap.add_argument('--settle-seconds', type=float, default=5, help='attesa prima della misura a riposo')
    ap.add_argument('--energy-frames', type=int, default=0, help='immagini tenute in GPU nel ciclo (0 = tutte)')
    ap.add_argument('--no-cudnn-benchmark', action='store_true', help='disattiva la scelta automatica dei kernel cuDNN')
    ap.add_argument('--no-flops', action='store_true')
    ap.add_argument('--no-subprocess', action='store_true', help='tutte le reti nello stesso processo (debug)')
    # percorsi e pesi: sovrascrivono config.json
    ap.add_argument('--tofdsr-root')
    ap.add_argument('--redwood-rgbd')
    ap.add_argument('--redwood-target')
    ap.add_argument('--scenes', nargs='+')
    ap.add_argument('--max-frames-per-scene', type=int)
    ap.add_argument('--ckpt', action='append', default=[], metavar='RETE=PERCORSO',
                    help='checkpoint per una rete, es. --ckpt fdsr=C:/pesi/fdsr_tofdsr.pth (ripetibile)')
    # interni
    ap.add_argument('--worker', choices=ALL_MODELS, help=argparse.SUPPRESS)
    ap.add_argument('--out', help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.amp:
        a.precision = 'fp16'
    return a


def load_cfg(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def overrides(a):
    o = {}
    for k in ('tofdsr_root', 'redwood_rgbd', 'redwood_target', 'scenes'):
        if getattr(a, k):
            o[k] = getattr(a, k)
    if a.max_frames_per_scene is not None:
        o['max_frames_per_scene'] = a.max_frames_per_scene
    return o


def ckpt_for(a, cfg, model):
    from gdsrbench import resolve
    cli = dict(x.split('=', 1) for x in a.ckpt if '=' in x)
    p = cli.get(model) or cfg['weights'].get(a.dataset, {}).get(model) or ''
    p = resolve(p)
    if p is not None and not p.is_file():
        print(f'  [ATTENZIONE] checkpoint non trovato: {p}  -> pesi casuali (RMSE non misurata)')
        return None
    return p


# =========================================================================== worker
def run_one(a, model):
    import torch
    from gdsrbench.datasets import build_dataset
    from gdsrbench.measure import run, set_precision
    from gdsrbench.wrappers import build_wrapper

    cfg = load_cfg(a.config)
    dev = torch.device(a.device if (a.device == 'cpu' or torch.cuda.is_available()) else 'cpu')
    if dev.type == 'cpu' and a.device != 'cpu':
        print('  [ATTENZIONE] CUDA non disponibile: misuro su CPU (niente energia). Controlla di '
              'aver installato PyTorch con supporto CUDA (vedi README).')
    if dev.type == 'cuda':
        torch.cuda.set_device(dev)                  # alcune reti usano .cuda() senza indice
        torch.backends.cudnn.benchmark = not a.no_cudnn_benchmark
    prec = set_precision(a.precision)
    torch.manual_seed(0)

    ds = build_dataset(a.dataset, cfg, limit=a.limit, overrides=overrides(a))
    print(f'  {ds.describe()}')
    wcfg = {'dav2_weights': cfg['weights'].get('depth_anything_v2_vits'),
            'sgnet_num_feats': cfg.get('sgnet_num_feats', 24)}
    norm = cfg.get('normalization', {}).get(a.dataset, {}).get(model)
    base = torch.cuda.memory_allocated(dev) if dev.type == 'cuda' else None
    w = build_wrapper(model, norm, dev, wcfg)
    ck = ckpt_for(a, cfg, model)
    if model == 'bicubic':
        print('  riferimento non appreso')
    else:
        print(f'  checkpoint: {ck if ck else "(nessuno: pesi casuali; si riporta la RMSE pubblicata)"}'
              f'   normalizzazione depth: {w.norm_mode}')
    energy = None
    if a.energy_seconds and a.energy_seconds > 0:
        energy = dict(seconds=a.energy_seconds, repeats=a.energy_repeats, idle_s=a.idle_seconds,
                      settle_s=a.settle_seconds, max_frames=a.energy_frames)
    res = run(w, ds, ck, warmup=a.warmup, repeats=a.repeats, precision=a.precision,
              flops=not a.no_flops, base_alloc=base, energy=energy)
    res['dataset'] = a.dataset
    res['normalization'] = w.norm_mode
    res['precision_flags'] = prec
    return res


def worker_main(a):
    try:
        res = run_one(a, a.worker)
        out = {'ok': True, 'result': res}
    except Exception as e:
        traceback.print_exc()
        msg = f'{type(e).__name__}: {e}'
        if 'out of memory' in str(e).lower():
            msg += '  (memoria GPU insufficiente: prova --precision fp16 o una GPU piu\' grande)'
        out = {'ok': False, 'error': msg}
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(out, f)
    return 0 if out['ok'] else 1


# =========================================================================== orchestratore
def main():
    a = parse()
    if a.worker:
        sys.exit(worker_main(a))

    from gdsrbench import MODEL_NAMES
    from gdsrbench.report import print_table, save

    print(f'Benchmark su {a.dataset.upper()}  -  precisione {a.precision}  -  reti: '
          f'{", ".join(MODEL_NAMES[m] for m in a.models)}')
    results, failures = [], {}
    for m in a.models:
        print(f'\n---------------- {MODEL_NAMES[m]} ----------------', flush=True)
        if a.no_subprocess:
            try:
                results.append(run_one(a, m))
            except Exception as e:
                traceback.print_exc()
                failures[m] = f'{type(e).__name__}: {e}'
            continue
        fd, tmp = tempfile.mkstemp(suffix='.json', prefix=f'bench_{m}_')
        os.close(fd)
        cmd = [sys.executable, str(Path(__file__).resolve()), '--worker', m, '--out', tmp]
        cmd += [x for x in sys.argv[1:]]
        env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
        rc = subprocess.call(cmd, env=env)
        try:
            with open(tmp, encoding='utf-8') as f:
                out = json.load(f)
        except Exception:
            out = {'ok': False, 'error': f'il processo e\' terminato senza risultati (codice {rc})'}
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
        if out['ok']:
            results.append(out['result'])
        else:
            failures[m] = out['error']

    print_table(a.dataset, results, failures)
    if results:
        base = save(a.dataset, results, failures, vars(a))
        print(f'  salvati: {base}.json  e  {base.name}.csv')
    sys.exit(0 if not failures else 1)


if __name__ == '__main__':
    main()
