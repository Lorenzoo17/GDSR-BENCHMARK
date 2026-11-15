"""Caricamento dei due test set in un formato unico, indipendente dalla rete.

Ogni campione e' un dict:
    rgb     uint8  HxWx3, ordine RGB
    lr_mm   float32 HxW, depth d'ingresso GIA' riportata alla risoluzione del GT, in mm
    gt_mm   float32 HxW, riferimento in mm
    mask    bool   HxW, pixel su cui si calcola la RMSE
    name    str
    lr_hw   (h, w) della depth a bassa risoluzione originale
    lr_raw_mm float32 lr_hw, la depth a bassa risoluzione NON ingrandita, in mm (serve alle
            reti che vogliono l'ingresso LR: SGNet, PromptDA, PriorDA)

La normalizzazione specifica di ogni rete NON si fa qui ma nel wrapper della rete
(gdsrbench/wrappers.py), cosi' i tre modelli vedono esattamente gli stessi dati.

Tutte le immagini si leggono con PIL e non con cv2.imread, che su Windows non apre i
percorsi con caratteri accentati (es. C:\\Universita' -> "Universita`").
"""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    BICUBIC = Image.Resampling.BICUBIC
except AttributeError:                       # Pillow < 9.1
    BICUBIC = Image.BICUBIC

CROP = 6          # bordo escluso dalla RMSE, come in tutti i codici ufficiali


# =========================================================================== TOFDSR
class TOFDSRTest:
    """Test set di TOFDSR (560 immagini), stesso protocollo di DORNet/DuCos.

    - lr = LR_Filled ingrandita bicubica (PIL) alla taglia del GT
    - maschera: GT fra 100 e 5000 mm, bordo di 6 px escluso
    Le liste ufficiali contengono percorsi tipo 'TOFDC_split/Test/RGB/...'. Se nella
    cartella del dataset la radice 'TOFDC_split' non c'e' (come in TOFDSR/TOFDSR/Test/...)
    il primo componente viene tolto automaticamente.
    """

    def __init__(self, root, test_list=None, limit=0):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f'Cartella TOFDSR non trovata: {self.root}\n'
                                    f'  -> imposta "tofdsr.root" in config.json o usa --tofdsr-root')
        if test_list is None or not Path(test_list).is_file():
            cands = [self.root / 'TOFDSR_Test.txt',
                     self.root.parent / 'TOFDSR_Test.txt']
            from . import REPO_DIRS
            cands += [REPO_DIRS['dornet'] / 'data' / 'TOFDSR_Test.txt']
            test_list = next((c for c in cands if c.is_file()), None)
            if test_list is None:
                raise FileNotFoundError('TOFDSR_Test.txt non trovato (ne\' nel dataset ne\' in '
                                        'external/DORNet/data). Esegui prima setup.')
        self.test_list = Path(test_list)
        lines = [l.strip() for l in open(self.test_list, encoding='utf-8') if l.strip()]
        self.items = [l.split(',')[:3] for l in lines]
        if limit and limit > 0:
            self.items = self.items[:limit]
        # controllo subito il primo file, cosi' un percorso sbagliato fallisce qui
        self._path(self.items[0][0])

    def _path(self, rel):
        rel = rel.replace('\\', '/')
        cands = [self.root / rel]
        parts = rel.split('/')
        if len(parts) > 1:
            cands.append(self.root.joinpath(*parts[1:]))
        for c in cands:
            if c.is_file():
                return c
        raise FileNotFoundError(f'File TOFDSR non trovato. Provati:\n  ' +
                                '\n  '.join(str(c) for c in cands))

    def describe(self):
        return f'TOFDSR test: {len(self.items)} immagini  (lista: {self.test_list})'

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rgb_p, gt_p, lr_p = (self._path(p) for p in self.items[i])
        rgb = np.array(Image.open(rgb_p).convert('RGB'))
        gt = np.array(Image.open(gt_p)).astype(np.float32)
        h, w = gt.shape
        lr_img = Image.open(lr_p)
        lr_hw = (lr_img.size[1], lr_img.size[0])
        # identico a data/tofdc_dataloader.py di DORNet (e al loader di DuCos)
        lr = np.array(lr_img.resize((w, h), BICUBIC)).astype(np.float32)
        mask = (gt >= 100) & (gt <= 5000)
        lr_raw = np.array(lr_img).astype(np.float32)
        return {'rgb': rgb, 'lr_mm': lr, 'gt_mm': gt, 'mask': mask,
                'name': Path(gt_p).stem, 'lr_hw': lr_hw, 'lr_raw_mm': lr_raw}


# =========================================================================== Redwood
DEPTH_MIN_M, DEPTH_MAX_M = 0.1, 10.0


def _read_depth_m(path):
    d = np.array(Image.open(path))
    if d.ndim == 3:
        d = d[:, :, 0]
    return d.astype(np.float32) / 1000.0


def downsample_masked(depth, valid, s):
    """Media su blocchi sxs contando solo i pixel validi (come redwood_dataloader.py)."""
    h, w = depth.shape
    h, w = (h // s) * s, (w // s) * s
    d = depth[:h, :w].reshape(h // s, s, w // s, s)
    v = valid[:h, :w].reshape(h // s, s, w // s, s).astype(np.float32)
    tot = (d * v).sum(axis=(1, 3))
    cnt = v.sum(axis=(1, 3))
    lr = np.where(cnt > 0, tot / np.maximum(cnt, 1e-6), 0.0).astype(np.float32)
    return lr, cnt > 0


def fill_holes(depth, max_iter=12):
    out = depth.copy()
    for _ in range(max_iter):
        holes = out <= 0
        if not holes.any():
            break
        vm = (out > 0).astype(np.float32)
        n = cv2.blur(vm, (3, 3))
        s = cv2.blur(out * vm, (3, 3))
        take = holes & (n > 0.01)
        if not take.any():
            break
        out[take] = (s / np.maximum(n, 1e-6))[take]
    return out


class RedwoodTest:
    """Redwood con target LiDAR, formulazione B (sensore ridotto x4 e riportato a HR).

    Replica la logica di redwood_dataloader.py usato in tesi:
      - ingresso: depth del sensore -> media mascherata su blocchi 4x4 -> riempimento buchi
        -> bicubica alla risoluzione piena
      - riferimento: <stem>_dense.png, valutato solo dove <stem>_mask.png e' attiva e la
        profondita' e' fra 0.1 e 10 m
    Struttura attesa:
      rgbd_root/<scena>/image/<stem>.jpg|png     rgbd_root/<scena>/depth/<stem>.png
      target_root/<scena>/<stem>_dense.png       target_root/<scena>/<stem>_mask.png
    """

    def __init__(self, rgbd_root, target_root, scenes=None, scale=4, min_coverage=30.0,
                 max_frames_per_scene=100, limit=0):
        self.rgbd_root, self.target_root = Path(rgbd_root), Path(target_root)
        for p, k in ((self.rgbd_root, 'redwood.rgbd_root'), (self.target_root, 'redwood.target_root')):
            if not p.is_dir():
                raise FileNotFoundError(f'Cartella Redwood non trovata: {p}\n  -> imposta "{k}" in config.json')
        self.scale = int(scale)
        if not scenes:
            scenes = sorted(d.name for d in self.target_root.iterdir()
                            if d.is_dir() and (self.rgbd_root / d.name).is_dir())
        self.items, self.per_scene = [], {}
        for scene in scenes:
            tdir, sdir = self.target_root / scene, self.rgbd_root / scene
            if not tdir.is_dir() or not sdir.is_dir():
                print(f'[attenzione] scena saltata, cartella mancante: {scene}')
                continue
            cov = {}
            man = tdir / 'targets_manifest.json'
            if man.exists():
                dfs = sorted((sdir / 'depth').glob('*.png'))
                for rec in json.load(open(man, encoding='utf-8')):
                    i = rec.get('frame', -1)
                    if 0 <= i < len(dfs):
                        cov[dfs[i].stem] = rec.get('coverage_pct', 100.0)
            stems = sorted(f.name[:-len('_dense.png')] for f in tdir.glob('*_dense.png'))
            stems = [s for s in stems if cov.get(s, 100.0) >= min_coverage]
            if max_frames_per_scene and len(stems) > max_frames_per_scene:
                idx = np.linspace(0, len(stems) - 1, max_frames_per_scene).round().astype(int)
                stems = [stems[i] for i in sorted(set(idx.tolist()))]
            self.per_scene[scene] = len(stems)
            self.items += [(scene, s) for s in stems]
        if limit and limit > 0:
            self.items = self.items[:limit]
        if not self.items:
            raise RuntimeError('Redwood: nessun frame trovato. Controlla percorsi, scene e min_coverage.')

    def describe(self):
        sc = '  '.join(f'{k}:{v}' for k, v in self.per_scene.items())
        return f'Redwood (formulazione B, x{self.scale}): {len(self.items)} frame  [{sc}]'

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        scene, stem = self.items[i]
        sdir, tdir = self.rgbd_root / scene, self.target_root / scene
        rgb = None
        for ext in ('.jpg', '.png', '.jpeg'):
            p = sdir / 'image' / f'{stem}{ext}'
            if p.exists():
                rgb = np.array(Image.open(p).convert('RGB'))
                break
        if rgb is None:
            raise FileNotFoundError(sdir / 'image' / f'{stem}.jpg')

        gt = _read_depth_m(tdir / f'{stem}_dense.png')
        mask = np.array(Image.open(tdir / f'{stem}_mask.png'))
        mask = (mask[..., 0] if mask.ndim == 3 else mask) > 0
        mask &= (gt > DEPTH_MIN_M) & (gt < DEPTH_MAX_M)

        cons = _read_depth_m(sdir / 'depth' / f'{stem}.png')
        valid = (cons > DEPTH_MIN_M) & (cons < DEPTH_MAX_M)
        cons = np.where(valid, cons, 0.0).astype(np.float32)
        lr, _ = downsample_masked(cons, valid, self.scale)
        lr_hw = lr.shape
        lr = fill_holes(lr)
        inp = cv2.resize(lr, (cons.shape[1], cons.shape[0]), interpolation=cv2.INTER_CUBIC)
        inp = np.clip(inp, 0.0, DEPTH_MAX_M)

        h, w = gt.shape
        if rgb.shape[:2] != (h, w):
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        if inp.shape != (h, w):
            inp = cv2.resize(inp, (w, h), interpolation=cv2.INTER_CUBIC)
        return {'rgb': rgb, 'lr_mm': inp * 1000.0, 'gt_mm': gt * 1000.0, 'mask': mask,
                'name': f'{scene}/{stem}', 'lr_hw': lr_hw, 'lr_raw_mm': lr * 1000.0}


# =========================================================================== metrica
def rmse_cm(pred_mm, gt_mm, mask):
    """RMSE in cm sui pixel della maschera, bordo di 6 px escluso. Accetta torch o numpy.

    Su TOFDSR coincide con tofdsr_calc_rmse di DORNet/DuCos (crop 6, gt in [100,5000] mm).
    """
    import torch
    c = CROP
    if isinstance(pred_mm, torch.Tensor):
        gt = torch.as_tensor(gt_mm, device=pred_mm.device)
        m = torch.as_tensor(mask, device=pred_mm.device)
        p, g, m = pred_mm[c:-c, c:-c], gt[c:-c, c:-c], m[c:-c, c:-c]
        if m.sum() == 0:
            return float('nan')
        d = (p[m] - g[m]) / 10.0
        return float(torch.sqrt(torch.mean(d * d)))
    p, g, m = pred_mm[c:-c, c:-c], gt_mm[c:-c, c:-c], mask[c:-c, c:-c]
    if m.sum() == 0:
        return float('nan')
    d = (p[m] - g[m]) / 10.0
    return float(np.sqrt(np.mean(d * d)))


def build_dataset(name, cfg, limit=0, overrides=None):
    from . import resolve
    o = overrides or {}
    if name == 'tofdsr':
        c = cfg['tofdsr']
        return TOFDSRTest(resolve(o.get('tofdsr_root') or c['root']),
                          resolve(c.get('test_list')), limit=limit)
    if name == 'redwood':
        c = cfg['redwood']
        return RedwoodTest(resolve(o.get('redwood_rgbd') or c['rgbd_root']),
                           resolve(o.get('redwood_target') or c['target_root']),
                           scenes=o.get('scenes') or c.get('scenes'),
                           scale=c.get('scale', 4), min_coverage=c.get('min_coverage', 30.0),
                           max_frames_per_scene=o.get('max_frames_per_scene',
                                                      c.get('max_frames_per_scene', 100)),
                           limit=limit)
    raise ValueError(name)
