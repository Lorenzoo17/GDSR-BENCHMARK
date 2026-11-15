#!/usr/bin/env python
"""Scarica le reti del benchmark (codice ufficiale + pesi disponibili) in external/ e weights/.

    python download_models.py                    # tutte le reti
    python download_models.py --pip              # prima installa anche requirements.txt
    python download_models.py --models dornet ducos
    python download_models.py --force            # riscarica tutto da capo

Rilanciarlo e' sicuro: salta cio' che e' gia' presente e completo.

  DKN, FDKN  external/DKN     github.com/cvlab-yonsei/dkn
  FDSR       external/FDSR    github.com/lingzhi96/RGB-D-D-Dataset
  DCTNet     external/DCTNet  github.com/Zhaozixiang1228/GDSR-DCTNet   (pesi NYU / RGB-D-D nel repo)
  SUFT       external/SUFT    github.com/ShiWuxuan/SUFT
  SGNet      external/SGNet   github.com/yanzq95/SGNet
  DORNet(-T) external/DORNet  github.com/yanzq95/DORNet                (pesi TOFDSR nel repo)
  DuCos      external/DuCos   github.com/yanzq95/DuCos                 (pesi da Hugging Face, ~215 MB l'uno)
             weights/depth_anything_v2_vits.pth                        (~100 MB)

Usa git se c'e', altrimenti lo zip di GitHub. Funziona uguale su Windows e Linux.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXTERNAL = HERE / 'external'
WEIGHTS = HERE / 'weights'

# chiave rete -> (cartella, owner, repo, file che deve esistere)
REPOS = {
    'dkn':    ('DKN', 'cvlab-yonsei', 'dkn', 'models.py'),
    'fdsr':   ('FDSR', 'lingzhi96', 'RGB-D-D-Dataset', 'codes/models.py'),
    'dctnet': ('DCTNet', 'Zhaozixiang1228', 'GDSR-DCTNet', 'model.py'),
    'suft':   ('SUFT', 'ShiWuxuan', 'SUFT', 'models/SUFT.py'),
    'sgnet':  ('SGNet', 'yanzq95', 'SGNet', 'models/SGNet.py'),
    'dornet': ('DORNet', 'yanzq95', 'DORNet', 'net/dornet_ddp.py'),
    'ducos':  ('DuCos', 'yanzq95', 'DuCos', 'model/ours/ducos.py'),
}
ALIASES = {'fdkn': 'dkn', 'dornet_t': 'dornet', 'bicubic': None}

HF = os.environ.get('HF_ENDPOINT', 'https://huggingface.co').rstrip('/')
# (rete, destinazione, url, dimensione attesa in byte o None, dataset per cui serve)
FILES = [
    ('ducos', EXTERNAL / 'DuCos' / 'ckpts' / 'RealTOFDSR.pth.tar',
     f'{HF}/RaynWu2002/DuCos/resolve/main/DuCos/ckpts/RealTOFDSR.pth.tar', 215327246, 'tofdsr'),
    ('ducos', EXTERNAL / 'DuCos' / 'ckpts' / 'RealRGBDD.pth.tar',
     f'{HF}/RaynWu2002/DuCos/resolve/main/DuCos/ckpts/RealRGBDD.pth.tar', 215327182, 'redwood'),
    ('ducos', WEIGHTS / 'depth_anything_v2_vits.pth',
     f'{HF}/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth', None, 'all'),
]

DORNET_CKPTS = ['TOFDSR.pth', 'RGBDD.pth', 'TOFDSR_T.pth', 'RGBDD_T.pth']


def say(msg=''):
    print(msg, flush=True)


# --------------------------------------------------------------------------- repository
def clone(name, owner, repo, check, force):
    dest = EXTERNAL / name
    if dest.exists() and (dest / check).is_file() and not force:
        say(f'  [ok] {name}: gia\' presente in {dest}')
        return True
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    url = f'https://github.com/{owner}/{repo}.git'
    if shutil.which('git'):
        say(f'  [git] clono {url}')
        r = subprocess.call(['git', 'clone', '--depth', '1', url, str(dest)])
        if r == 0 and (dest / check).is_file():
            return True
        say(f'  [git] fallito (codice {r}), provo con lo zip')
        shutil.rmtree(dest, ignore_errors=True)
    for branch in ('main', 'master'):
        zurl = f'https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{branch}'
        tmp = Path(tempfile.mkdtemp())
        try:
            zpath = tmp / 'repo.zip'
            download(zurl, zpath, None, label=f'{name} (zip)')
            with zipfile.ZipFile(zpath) as z:
                z.extractall(tmp / 'x')
            inner = next((tmp / 'x').iterdir())
            shutil.move(str(inner), str(dest))
            if (dest / check).is_file():
                return True
        except Exception as e:
            say(f'  [zip] {branch}: {type(e).__name__}: {e}')
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    say(f'  [ERRORE] impossibile scaricare {name}')
    return False


# --------------------------------------------------------------------------- file
def download(url, dest, expected=None, label=None):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + '.part')
    label = label or dest.name
    req = urllib.request.Request(url, headers={'User-Agent': 'gdsr-benchmark/1.0'})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as r, open(part, 'wb') as f:
        total = int(r.headers.get('Content-Length') or 0) or expected or 0
        done, last = 0, 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if time.time() - last > 0.5:
                last = time.time()
                pct = f'{100 * done / total:5.1f}%' if total else ''
                sys.stdout.write(f'\r  [download] {label}: {done / 2**20:7.1f} MB {pct}')
                sys.stdout.flush()
    sys.stdout.write(f'\r  [download] {label}: {done / 2**20:7.1f} MB in {time.time() - t0:.0f} s\n')
    if expected and done != expected:
        raise IOError(f'dimensione inattesa: {done} byte invece di {expected}')
    if total and done != total:
        raise IOError(f'download incompleto: {done} byte su {total}')
    os.replace(part, dest)


def fetch(dest, url, expected, force):
    if dest.is_file() and not force and (expected is None or dest.stat().st_size == expected):
        say(f'  [ok] {dest.relative_to(HERE)}: gia\' presente')
        return True
    try:
        download(url, dest, expected)
        return True
    except Exception as e:
        say(f'\n  [ERRORE] {dest.name}: {type(e).__name__}: {e}')
        say(f'           scaricalo a mano da {url}\n           e mettilo in {dest}')
        return False


def check_dornet():
    ok = True
    for c in DORNET_CKPTS:
        p = EXTERNAL / 'DORNet' / 'checkpoints' / c
        if not p.is_file():
            say(f'  [ERRORE] manca {p}')
            ok = False
        elif p.stat().st_size < 10_000:            # puntatore Git LFS invece del file vero
            say(f'  [ERRORE] {p} e\' un puntatore Git LFS ({p.stat().st_size} byte). '
                f'Esegui "git lfs pull" in external/DORNet oppure riscarica con --force senza git.')
            ok = False
    if ok:
        say(f'  [ok] pesi DORNet: {", ".join(DORNET_CKPTS)}')
    return ok


def pip_install():
    req = HERE / 'requirements.txt'
    say(f'  [pip] installo {req.name}')
    return subprocess.call([sys.executable, '-m', 'pip', 'install', '-r', str(req)]) == 0


def check_env(models):
    try:
        import torch
        cuda = torch.cuda.is_available()
        say(f'  [ok] PyTorch {torch.__version__}  CUDA disponibile: {cuda}'
            + (f'  ({torch.cuda.get_device_name(0)})' if cuda else ''))
        if not cuda:
            say('  [ATTENZIONE] PyTorch senza CUDA: i tempi sarebbero quelli della CPU.\n'
                '               Installa la versione CUDA da https://pytorch.org/get-started/locally/')
    except ImportError:
        say('  [ATTENZIONE] PyTorch non installato. Installalo da https://pytorch.org/get-started/locally/\n'
            '               (scegli la versione con CUDA), poi riesegui questo script.')
        return
    if torch.cuda.is_available():
        try:
            import pynvml as nv
            nv.nvmlInit()
            h = nv.nvmlDeviceGetHandleByIndex(0)
            try:
                nv.nvmlDeviceGetTotalEnergyConsumption(h)
                say('  [ok] NVML: contatore di energia disponibile')
            except Exception:
                say('  [info] NVML: contatore di energia NON disponibile su questa GPU: si integrera\' la potenza')
        except ImportError:
            say('  [ATTENZIONE] manca nvidia-ml-py (pip install nvidia-ml-py): niente misure di energia')
        except Exception as e:
            say(f'  [ATTENZIONE] NVML non inizializzabile ({e}): niente misure di energia')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pip', action='store_true', help='installa prima requirements.txt (PyTorch escluso)')
    ap.add_argument('--force', action='store_true', help='riscarica anche cio\' che c\'e\' gia\'')
    ap.add_argument('--no-weights', action='store_true', help='solo i repository, niente pesi')
    ap.add_argument('--models', nargs='+', default=list(REPOS),
                    choices=list(REPOS) + [k for k in ALIASES], help='quali reti scaricare (default: tutte)')
    ap.add_argument('--only', choices=['tofdsr', 'redwood'], help='pesi DuCos solo per questo dataset')
    a = ap.parse_args()
    models = sorted({ALIASES.get(m, m) for m in a.models} - {None}, key=list(REPOS).index)

    EXTERNAL.mkdir(exist_ok=True)
    WEIGHTS.mkdir(exist_ok=True)
    ok = True
    if a.pip:
        ok &= pip_install()
    say('\n== Repository')
    for m in models:
        name, owner, repo, check = REPOS[m]
        ok &= clone(name, owner, repo, check, a.force)
    say('\n== Pesi')
    if 'dornet' in models and (EXTERNAL / 'DORNet').is_dir():
        ok &= check_dornet()
    say('  [info] Pesi TOFDSR pubblici solo per DORNet, DORNet-T e DuCos. Le altre reti girano con\n'
        '         pesi casuali: latenza, memoria ed energia sono valide, la RMSE si prende dal paper.')
    if not a.no_weights:
        for m, dest, url, size, use in FILES:
            if m not in models or (a.only and use not in (a.only, 'all')):
                continue
            ok &= fetch(dest, url, size, a.force)
    say('\n== Ambiente')
    check_env(models)
    say('\nFatto.' if ok else '\nFinito con errori: leggi i messaggi sopra (le altre reti sono comunque pronte).')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
