#!/usr/bin/env python
"""Scarica il dataset TOFDSR, lo estrae e lo collega al benchmark.

    python download_tofdsr.py                 # scarica tutto in datasets/TOFDSR
    python download_tofdsr.py --solo-test     # estrae solo il test (560 immagini): basta per il benchmark
    python download_tofdsr.py --da-archivio C:/Download/TOFDSR.zip   # usa un archivio gia' scaricato
    python download_tofdsr.py --dest /dati/TOFDSR                      # altra cartella

Cosa fa
  1. se config.json punta gia' a un TOFDSR valido, si ferma (usa --force per rifare);
  2. scarica l'archivio ufficiale da Google Drive (con gdown, installato se manca);
  3. lo estrae e cerca da solo la cartella che contiene Test/RGB, Test/GT, Test/LR_Filled,
     qualunque sia la struttura interna dell'archivio;
  4. la sposta in <dest> (default: datasets/TOFDSR dentro questa cartella), aggiunge la lista
     TOFDSR_Test.txt se manca, controlla che ci siano le 560 immagini di test;
  5. aggiorna "tofdsr.root" in config.json, cosi' run_tofdsr / run_paper la trovano subito.

Fonte: pagina ufficiale TOFDC (https://yanzq95.github.io/projectpage/TOFDC/index.html).
Licenza: solo uso accademico, non commerciale (PCA Lab, Nanjing University of Science and
Technology). Citare: Yan et al., "Tri-Perspective View Decomposition for Geometry-Aware
Depth Completion", CVPR 2024.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = HERE / 'config.json'
DEFAULT_DEST = HERE / 'datasets' / 'TOFDSR'

GDRIVE_ID = '1iQEtPBVIA8pmYeSh6YKH93CsJWik1EJz'
GDRIVE_URL = f'https://drive.google.com/file/d/{GDRIVE_ID}/view'
BAIDU_URL = 'https://pan.baidu.com/s/105z6ppzzxH7aC81AUjfwIA?pwd=nust'
N_TEST = 560
SUBDIRS = ('RGB', 'GT', 'LR_Filled')

try:                                     # console Windows: niente crash sui caratteri accentati
    sys.stdout.reconfigure(errors='replace')
except Exception:
    pass


def say(msg=''):
    print(msg, flush=True)


# --------------------------------------------------------------------------- controlli
def _child(d, name):
    """Sottocartella `name` di d, senza badare a maiuscole/minuscole."""
    if not d.is_dir():
        return None
    p = d / name
    if p.is_dir():
        return p
    for c in d.iterdir():
        if c.is_dir() and c.name.lower() == name.lower():
            return c
    return None


def check_dataset(root):
    """(ok, messaggio) per una cartella che dovrebbe contenere Test/RGB, Test/GT, Test/LR_Filled."""
    root = Path(root)
    test = _child(root, 'Test')
    if test is None:
        return False, f'manca la cartella Test in {root}'
    counts = {}
    for s in SUBDIRS:
        d = _child(test, s)
        if d is None:
            return False, f'manca Test/{s} in {root}'
        counts[s] = sum(1 for f in d.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg'))
    bad = {k: v for k, v in counts.items() if v < N_TEST}
    if bad:
        return False, 'immagini di test mancanti: ' + ', '.join(f'{k} {v}/{N_TEST}' for k, v in bad.items())
    return True, 'test completo: ' + ', '.join(f'{k} {v}' for k, v in counts.items())


def find_dataset_root(base):
    """Cerca, dentro l'archivio estratto, la cartella che contiene Test/{RGB,GT,LR_Filled}."""
    base = Path(base)
    cands = [base] + [p for p in base.rglob('*') if p.is_dir()]
    for d in sorted(cands, key=lambda p: len(p.parts)):
        t = _child(d, 'Test')
        if t is not None and all(_child(t, s) is not None for s in SUBDIRS):
            return d
    return None


def config_root():
    try:
        cfg = json.load(open(CONFIG, encoding='utf-8'))
        p = Path(str(cfg['tofdsr']['root'])).expanduser()
        return p if p.is_absolute() else (HERE / p).resolve()
    except Exception:
        return None


def update_config(dest):
    cfg = json.load(open(CONFIG, encoding='utf-8'))
    try:
        val = os.path.relpath(dest, HERE)
    except ValueError:                    # Windows: dischi diversi
        val = str(dest)
    val = val.replace('\\', '/')
    old = cfg['tofdsr'].get('root')
    cfg['tofdsr']['root'] = val
    shutil.copy2(CONFIG, CONFIG.with_suffix('.json.bak'))
    with open(CONFIG, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write('\n')
    say(f'  [ok] config.json: tofdsr.root  "{old}"  ->  "{val}"   (copia del vecchio in config.json.bak)')


# --------------------------------------------------------------------------- download
def ensure_gdown():
    try:
        import gdown  # noqa: F401
        return True
    except ImportError:
        say('  [pip] installo gdown (serve per scaricare da Google Drive)')
        r = subprocess.call([sys.executable, '-m', 'pip', 'install', 'gdown'])
        if r != 0:
            return False
        try:
            import importlib
            importlib.invalidate_caches()
            import gdown  # noqa: F401
            return True
        except ImportError:
            return False


def download_archive(workdir):
    if not ensure_gdown():
        return None
    import gdown
    out = Path(workdir) / 'TOFDSR_archivio'
    say(f'  [download] TOFDSR da Google Drive (alcuni GB: puo\' richiedere un po\')')
    try:
        try:
            path = gdown.download(id=GDRIVE_ID, output=str(out), quiet=False, resume=True)
        except TypeError:                 # gdown vecchio
            path = gdown.download(f'https://drive.google.com/uc?id={GDRIVE_ID}', str(out), quiet=False)
    except Exception as e:
        say(f'  [ERRORE] download: {type(e).__name__}: {e}')
        path = None
    if not path or not Path(path).is_file() or Path(path).stat().st_size < 1_000_000:
        say('  [ERRORE] Google Drive non ha restituito l\'archivio (a volte limita i download dei file\n'
            '           molto richiesti: riprova piu\' tardi). In alternativa scaricalo a mano da:\n'
            f'             {GDRIVE_URL}\n'
            f'             {BAIDU_URL}   (codice: nust)\n'
            '           e poi:  download_tofdsr --da-archivio <file scaricato>')
        return None
    return Path(path)


# --------------------------------------------------------------------------- estrazione
def extract(archive, into, solo_test):
    archive = Path(archive)
    into = Path(into)
    into.mkdir(parents=True, exist_ok=True)

    def wanted(name):
        if not solo_test:
            return True
        parts = [p.lower() for p in name.replace('\\', '/').split('/')]
        return 'test' in parts or parts[-1].endswith('.txt')

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            names = [n for n in z.namelist() if wanted(n)]
            say(f'  [estrazione] zip: {len(names)} file')
            for i, n in enumerate(names, 1):
                target = (into / n).resolve()
                if not str(target).startswith(str(into.resolve())):   # percorsi fuori cartella
                    continue
                z.extract(n, into)
                if i % 2000 == 0:
                    say(f'    {i}/{len(names)}')
        return True
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            members = [m for m in t.getmembers() if wanted(m.name)
                       and not m.name.startswith(('/', '..')) and '..' not in Path(m.name).parts]
            say(f'  [estrazione] tar: {len(members)} file')
            t.extractall(into, members=members)
        return True
    seven = shutil.which('7z') or shutil.which('7za') or shutil.which('7zz')
    if seven:
        say('  [estrazione] archivio non zip/tar: uso 7-Zip')
        return subprocess.call([seven, 'x', '-y', f'-o{into}', str(archive)]) == 0
    say('  [ERRORE] formato dell\'archivio non riconosciuto (non e\' zip o tar) e 7-Zip non e\' installato.\n'
        '           Estrailo a mano e poi:  download_tofdsr --da-cartella <cartella estratta>')
    return False


def place(found, dest):
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(found), str(dest))
    # le liste stanno a volte accanto alla cartella del dataset
    for name in ('TOFDSR_Test.txt', 'TOFDSR_Train.txt'):
        if not (dest / name).is_file():
            for c in (found.parent / name, found.parent.parent / name):
                if c.is_file():
                    shutil.copy2(c, dest / name)
                    break
    if not (dest / 'TOFDSR_Test.txt').is_file():
        lst = HERE / 'external' / 'DORNet' / 'data' / 'TOFDSR_Test.txt'
        if lst.is_file():
            shutil.copy2(lst, dest / 'TOFDSR_Test.txt')
            say('  [info] TOFDSR_Test.txt preso da external/DORNet/data')
        else:
            say('  [info] TOFDSR_Test.txt non trovato: il benchmark usera\' quella di DORNet dopo il setup')


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dest', default=str(DEFAULT_DEST), help='dove mettere il dataset (default: datasets/TOFDSR)')
    ap.add_argument('--solo-test', action='store_true', help='estrae solo il test set (risparmia molto spazio)')
    ap.add_argument('--da-archivio', help='usa un archivio gia\' scaricato invece di scaricarlo')
    ap.add_argument('--da-cartella', help='usa una cartella gia\' estratta (la sposta in --dest)')
    ap.add_argument('--tieni-archivio', action='store_true', help='non cancellare l\'archivio scaricato')
    ap.add_argument('--no-config', action='store_true', help='non modificare config.json')
    ap.add_argument('--force', action='store_true', help='rifai anche se il dataset c\'e\' gia\'')
    a = ap.parse_args()
    dest = Path(a.dest).expanduser().resolve()

    say('== TOFDSR')
    say('  Licenza: solo uso accademico e non commerciale (PCA Lab, NJUST). Va citato:\n'
        '  Yan et al., "Tri-Perspective View Decomposition for Geometry-Aware Depth Completion", CVPR 2024.\n')

    # ------------------------------------------------------------ gia' presente?
    if not a.force:
        for cand in [config_root(), dest]:
            if cand is not None and cand.is_dir():
                ok, msg = check_dataset(cand)
                if ok:
                    say(f'  [ok] TOFDSR gia\' presente in {cand}  ({msg})')
                    if cand != config_root() and not a.no_config:
                        update_config(cand)
                    say('\nNiente da fare (usa --force per riscaricarlo).')
                    return 0

    # ------------------------------------------------------------ sorgente
    dest.parent.mkdir(parents=True, exist_ok=True)       # temporanei sullo stesso disco: spostamento immediato
    work = Path(tempfile.mkdtemp(prefix='tofdsr_tmp_', dir=str(dest.parent)))
    try:
        if a.da_cartella:
            src = Path(a.da_cartella).expanduser().resolve()
            found = find_dataset_root(src)
            if found is None:
                say(f'  [ERRORE] in {src} non trovo Test/RGB, Test/GT, Test/LR_Filled')
                return 1
            if found == dest:
                pass
            else:
                say(f'  [sposto] {found} -> {dest}')
                place(found, dest)
        else:
            if a.da_archivio:
                archive = Path(a.da_archivio).expanduser().resolve()
                if not archive.is_file():
                    say(f'  [ERRORE] archivio non trovato: {archive}')
                    return 1
            else:
                archive = download_archive(work)
                if archive is None:
                    return 1
            say(f'  archivio: {archive}  ({archive.stat().st_size / 2**30:.2f} GB)')
            ex = work / 'estratto'
            if not extract(archive, ex, a.solo_test):
                return 1
            found = find_dataset_root(ex)
            if found is None:
                say('  [ERRORE] nell\'archivio non trovo la struttura Test/RGB, Test/GT, Test/LR_Filled.\n'
                    f'           Contenuto estratto lasciato in {ex} per controllarlo.')
                a.tieni_archivio = True
                return 1
            place(found, dest)
            if not a.da_archivio and a.tieni_archivio:
                keep = dest.parent / archive.name
                shutil.move(str(archive), str(keep))
                say(f'  archivio conservato in {keep}')

        ok, msg = check_dataset(dest)
        say(f'  [{"ok" if ok else "ATTENZIONE"}] {dest}: {msg}')
        if not a.no_config:
            update_config(dest)
        say('\nFatto. Ora puoi lanciare run_tofdsr / run_paper.' if ok else
            '\nIl dataset e\' stato messo al suo posto ma il test non e\' completo: controlla i messaggi sopra.')
        return 0 if ok else 1
    finally:
        if not a.tieni_archivio:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
