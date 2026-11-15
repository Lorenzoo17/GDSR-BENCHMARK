"""Le reti del benchmark dietro un'interfaccia unica.

    w = build_wrapper('dornet', norm_mode, device, cfg)
    info = w.load(ckpt_path)                 # dict con lo stato del caricamento
    inp, ctx = w.prepare(sample)             # tensori gia' sul device (fuori dalle misure)
    out = w.forward(inp)                     # SOLO la rete: e' la parte misurata
    pred_mm = w.postprocess(out, ctx)        # HxW torch, in millimetri
    w.prior_module()                         # sotto-modulo del prior congelato (o None)

Il codice delle reti viene importato dai repository ufficiali in external/ (scaricati da
download_models.py), senza modificarli. Le toppe applicate a runtime sono solo di
compatibilita' e sono elencate qui:
  - DORNet: se mmcv non c'e' (su Windows quasi mai), la convoluzione deformabile modulata
    passa a torchvision.ops.deform_conv2d (stessa DCNv2; verificato 4.2117 cm su TOFDSR).
  - DuCos: il costruttore carica Depth Anything V2 da un percorso assoluto del server degli
    autori: lo si redirige a weights/depth_anything_v2_vits.pth.
  - DCTNet: usa torch.rfft/torch.irfft, rimossi in PyTorch 1.8; sono ridefiniti con
    torch.fft (stessa trasformata).
  - DKN/FDKN e SGNet: chiamano .cuda() fisso; su CPU (solo per prove) .cuda() diventa nullo.

Preprocessing = quello del test ufficiale di ciascuna rete (vedi README). Per le reti senza
pesi addestrati su TOFDSR la RMSE non si misura (si riporta quella pubblicata): il
preprocessing conta solo perche' le dimensioni dei tensori siano quelle vere.
"""

import importlib.util
import sys
import types
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import REPO_DIRS, resolve

try:
    from PIL import Image
    _BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    _BICUBIC = Image.BICUBIC


# --------------------------------------------------------------------------- utility
def _add_path(p):
    p = str(p)
    if p not in sys.path:
        sys.path.insert(0, p)


def _import_file(name, path):
    """Importa un file .py con un nome univoco (evita conflitti fra 'models.py' diversi)."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_REAL_TORCH_LOAD = torch.load


def _torch_load(path):
    """torch.load compatibile con torch vecchi e nuovi (>=2.6 ha weights_only=True)."""
    try:
        return _REAL_TORCH_LOAD(path, map_location='cpu', weights_only=False)
    except TypeError:
        return _REAL_TORCH_LOAD(path, map_location='cpu')


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ('state_dict', 'model_state_dict', 'model', 'net', 'params'):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def _strip(sd, prefixes=('module.',)):
    out = OrderedDict()
    for k, v in sd.items():
        for p in prefixes:
            if k.startswith(p):
                k = k[len(p):]
        out[k] = v
    return out


def _load_into(model, sd, allow_missing_prefix=None):
    """Carica i pesi e verifica che sia andato tutto a posto (un caricamento parziale non
    da' errori ma produce RMSE senza senso)."""
    res = model.load_state_dict(sd, strict=False)
    missing = list(res.missing_keys)
    unexpected = list(res.unexpected_keys)
    ignorable = [k for k in missing if allow_missing_prefix and k.startswith(allow_missing_prefix)]
    real_missing = [k for k in missing if k not in ignorable]
    ok = not real_missing and not unexpected
    if real_missing or unexpected:
        print(f'  [ATTENZIONE] caricamento parziale: {len(real_missing)} chiavi mancanti, '
              f'{len(unexpected)} inattese')
        for k in (real_missing + unexpected)[:8]:
            print('     ', k)
    return {'ok': ok, 'missing': len(real_missing), 'unexpected': len(unexpected),
            'missing_ignored': len(ignorable)}


def _rgb_minmax(rgb_u8):
    x = rgb_u8.astype(np.float32)
    mn, mx = x.min(), x.max()
    return (x - mn) / max(mx - mn, 1e-6)


def _norm_depth(lr_mm, mode):
    """Ritorna (lr_norm, scale, offset) con  mm = norm * scale + offset."""
    if isinstance(mode, str) and mode.lower() == 'minmax':
        mn, mx = float(lr_mm.min()), float(lr_mm.max())
        sc = max(mx - mn, 1e-6)
        return (lr_mm - mn) / sc, sc, mn
    sc = float(mode)
    return lr_mm / sc, sc, 0.0


def _to_t(a, device):
    a = np.ascontiguousarray(a)
    t = torch.from_numpy(a).float()
    if t.ndim == 2:
        t = t[None, None]
    elif t.ndim == 3:
        t = t.permute(2, 0, 1)[None]
    return t.to(device)


def _pad_to(t, mult):
    if mult <= 1:
        return t, (0, 0)
    h, w = t.shape[-2:]
    ph, pw = (-h) % mult, (-w) % mult
    if ph == 0 and pw == 0:
        return t, (0, 0)
    return F.pad(t, (0, pw, 0, ph), mode='replicate'), (ph, pw)


def _lr_lowres(s, scale):
    """Depth LR alla taglia (H/s, W/s), come nei loader RGB-D-D 'real' di SGNet e SUFT:
    il file LR viene ridimensionato bicubico (PIL) a W/s x H/s."""
    h, w = s['gt_mm'].shape
    lr = s['lr_raw_mm']
    if lr.shape != (h // scale, w // scale):
        lr = np.array(Image.fromarray(lr.astype(np.float32), mode='F')
                      .resize((w // scale, h // scale), _BICUBIC)).astype(np.float32)
    return lr


class _cuda_noop:
    """Rende nullo Tensor.cuda() (solo su CPU, per le reti che lo chiamano fisso)."""
    def __init__(self, active):
        self.active = active

    def __enter__(self):
        if self.active:
            self.real = torch.Tensor.cuda
            torch.Tensor.cuda = lambda t, *a, **k: t

    def __exit__(self, *a):
        if self.active:
            torch.Tensor.cuda = self.real


# --------------------------------------------------------------------------- base
class Wrapper:
    key = ''
    pad_mult = 1
    default_norm = 'minmax'

    def __init__(self, norm_mode, device, cfg):
        self.norm_mode = norm_mode if norm_mode is not None else self.default_norm
        self.device = device
        self.cfg = cfg
        self.net = None
        self.extra = {}

    def prior_module(self):
        """Modulo pre-addestrato congelato eseguito a inferenza (es. Depth Anything)."""
        return None

    def prior_calls(self):
        """Chiamate (oggetto, nome_metodo) che costituiscono il prior, per misurarne il tempo."""
        return []

    def n_params(self):
        """(parametri della rete di ricostruzione, parametri del prior congelato)."""
        tot = sum(p.numel() for p in self.net.parameters())
        pm = self.prior_module()
        pri = sum(p.numel() for p in pm.parameters()) if pm is not None else 0
        return tot - pri, pri

    def load_generic(self, ckpt, prefixes=('module.',)):
        if not ckpt:
            return {'ok': False, 'reason': 'nessun checkpoint TOFDSR (si riporta la RMSE pubblicata)'}
        return _load_into(self.net, _strip(_extract_state_dict(_torch_load(ckpt)), prefixes))

    load = load_generic

    def postprocess(self, out, ctx):
        ph, pw = ctx['pad']
        o = out[0, 0].float()
        if ph:
            o = o[:-ph]
        if pw:
            o = o[:, :-pw]
        if ctx.get('clamp01'):
            o = o.clamp(0, 1)
        return o * ctx['scale'] + ctx['offset']


# --------------------------------------------------------------------------- Bicubic
class _Bicubic(nn.Module):
    def forward(self, lr, size):
        return F.interpolate(lr, size=size, mode='bicubic', align_corners=False)


class BicubicWrapper(Wrapper):
    """Riferimento non appreso: ingrandimento bicubico della LR (sulla GPU)."""
    key = 'bicubic'

    def build(self):
        self.net = _Bicubic()

    def load(self, ckpt):
        return {'ok': True, 'note': 'non appreso'}

    def prepare(self, s):
        return (_to_t(s['lr_raw_mm'], self.device), tuple(s['gt_mm'].shape)), \
            {'scale': 1.0, 'offset': 0.0, 'pad': (0, 0)}

    def forward(self, inp):
        return self.net(*inp)


# --------------------------------------------------------------------------- DKN / FDKN
class DKNWrapper(Wrapper):
    """DKN (IJCV 2021). Come inference.py: RGB in BGR /255 (cv2), depth gia' ingrandita a HR.
    filter_size d=15, kernel k=3 (configurazione del repository)."""
    key = 'dkn'
    cls_name = 'DKN'
    pad_mult = 4

    def build(self):
        f = REPO_DIRS[self.key] / 'models.py'
        if not f.is_file():
            raise FileNotFoundError(f'DKN non trovato in {f.parent}. Esegui setup.')
        m = _import_file('dkn_models', f)
        self.net = getattr(m, self.cls_name)(kernel_size=3, filter_size=15, residual=True)

    def prepare(self, s):
        bgr = s['rgb'][..., ::-1].astype(np.float32) / 255.0
        lr_n, sc, off = _norm_depth(s['lr_mm'], self.norm_mode)
        img, pad = _pad_to(_to_t(bgr, self.device), self.pad_mult)
        dep, _ = _pad_to(_to_t(lr_n, self.device), self.pad_mult)
        return (img, dep), {'scale': sc, 'offset': off, 'pad': pad}

    def forward(self, inp):
        with _cuda_noop(self.device.type != 'cuda'):     # grid_generator chiama .cuda()
            return self.net(inp)


class FDKNWrapper(DKNWrapper):
    key = 'fdkn'
    cls_name = 'FDKN'


# --------------------------------------------------------------------------- FDSR
class FDSRWrapper(Wrapper):
    """FDSR (CVPR 2021). RGB in BGR /255 (cv2 nel loro test.py), depth HR min-max."""
    key = 'fdsr'
    pad_mult = 4          # resample_data(x, 4) vuole H e W multipli di 4

    def build(self):
        f = REPO_DIRS['fdsr'] / 'codes' / 'models.py'
        if not f.is_file():
            raise FileNotFoundError(f'FDSR non trovato in {f.parent}. Esegui setup.')
        _add_path(f.parent)                              # models.py importa octconv
        m = _import_file('fdsr_models', f)
        self.net = m.Net(num_feats=32, depth_chanels=1, color_channel=3, kernel_size=3)

    def prepare(self, s):
        bgr = s['rgb'][..., ::-1].astype(np.float32) / 255.0
        lr_n, sc, off = _norm_depth(s['lr_mm'], self.norm_mode)
        img, pad = _pad_to(_to_t(bgr, self.device), self.pad_mult)
        dep, _ = _pad_to(_to_t(lr_n, self.device), self.pad_mult)
        return (img, dep), {'scale': sc, 'offset': off, 'pad': pad}

    def forward(self, inp):
        return self.net(inp)


# --------------------------------------------------------------------------- DCTNet
def _ensure_old_fft():
    """torch.rfft / torch.irfft (API rimossa in PyTorch 1.8) riscritte con torch.fft."""
    if hasattr(torch, 'rfft') and hasattr(torch, 'irfft'):
        return

    def rfft(x, signal_ndim, normalized=False, onesided=True):
        assert signal_ndim == 1
        norm = 'ortho' if normalized else 'backward'
        X = torch.fft.rfft(x, dim=-1, norm=norm) if onesided else torch.fft.fft(x, dim=-1, norm=norm)
        return torch.view_as_real(X)

    def irfft(X, signal_ndim, normalized=False, onesided=True, signal_sizes=None):
        assert signal_ndim == 1
        norm = 'ortho' if normalized else 'backward'
        c = torch.view_as_complex(X.contiguous())
        if onesided:
            n = signal_sizes[-1] if signal_sizes else None
            return torch.fft.irfft(c, n=n, dim=-1, norm=norm)
        return torch.fft.ifft(c, dim=-1, norm=norm).real

    torch.rfft = rfft
    torch.irfft = irfft


class DCTNetWrapper(Wrapper):
    """DCTNet (CVPR 2022). Depth gia' ingrandita a HR e normalizzata in [0,1], RGB in [0,1];
    l'uscita e' tagliata in [0,1] come nel loro test.py."""
    key = 'dctnet'

    def build(self):
        repo = REPO_DIRS['dctnet']
        if not (repo / 'model.py').is_file():
            raise FileNotFoundError(f'DCTNet non trovato in {repo}. Esegui setup.')
        _ensure_old_fft()
        _add_path(repo)                                  # model.py importa dct
        m = _import_file('dctnet_model', repo / 'model.py')
        self.net = m.DCTNet()

    def prepare(self, s):
        lr_n, sc, off = _norm_depth(s['lr_mm'], self.norm_mode)
        return (_to_t(lr_n, self.device), _to_t(_rgb_minmax(s['rgb']), self.device)), \
            {'scale': sc, 'offset': off, 'pad': (0, 0), 'clamp01': True}

    def forward(self, inp):
        return self.net(*inp)


# --------------------------------------------------------------------------- SUFT / SGNet
class _LowResInput(Wrapper):
    """Reti che prendono la depth a BASSA risoluzione (H/4 x W/4): LR e RGB min-max per
    immagine, come nei loro rgbdd_dataloader ('real')."""
    scale = 4

    def prepare(self, s):
        h, w = s['gt_mm'].shape
        lr = _lr_lowres(s, self.scale)
        mn, mx = float(lr.min()), float(lr.max())
        sc = max(mx - mn, 1e-6)
        H, W = lr.shape[0] * self.scale, lr.shape[1] * self.scale
        rgb = _to_t(_rgb_minmax(s['rgb'][:H, :W]), self.device)
        return (rgb, _to_t((lr - mn) / sc, self.device)), {'scale': sc, 'offset': mn, 'pad': (0, 0),
                                                           'hw': (h, w)}

    def postprocess(self, out, ctx):
        o = out[0, 0].float() * ctx['scale'] + ctx['offset']
        h, w = ctx['hw']
        if o.shape != (h, w):                       # solo se H o W non sono multipli di 4
            o = F.pad(o[None, None], (0, w - o.shape[1], 0, h - o.shape[0]), mode='replicate')[0, 0]
        return o


class SUFTWrapper(_LowResInput):
    """SUFT (ACM MM 2022), num_feats=32 come nel loro test.py (22.01 M parametri)."""
    key = 'suft'

    def build(self):
        repo = REPO_DIRS['suft']
        if not (repo / 'models' / 'SUFT.py').is_file():
            raise FileNotFoundError(f'SUFT non trovato in {repo}. Esegui setup.')
        _add_path(repo)
        from models.SUFT import SUFT_network
        self.net = SUFT_network(num_feats=32, kernel_size=3, scale=self.scale)

    def forward(self, inp):
        return self.net(inp)


class SGNetWrapper(_LowResInput):
    """SGNet (AAAI 2024). num_feats=24: la configurazione real-world, che da' gli 8.97 M
    parametri riportati da DORNet su TOFDSR (num_feats=48 e' quella di NYU, ~36 M)."""
    key = 'sgnet'

    def build(self):
        repo = REPO_DIRS['sgnet']
        if not (repo / 'models' / 'SGNet.py').is_file():
            raise FileNotFoundError(f'SGNet non trovato in {repo}. Esegui setup.')
        _add_path(repo)
        from models.SGNet import SGNet
        nf = int(self.cfg.get('sgnet_num_feats', 24))
        with _cuda_noop(self.device.type != 'cuda'):   # models/common.py chiama .cuda()
            self.net = SGNet(num_feats=nf, kernel_size=3, scale=self.scale)
        self.extra['num_feats'] = nf

    def forward(self, inp):
        out, _ = self.net(inp)
        return out


# --------------------------------------------------------------------------- DORNet
def _ensure_deform_conv():
    """Rende importabile 'mmcv.ops.modulated_deform_conv2d', con torchvision se serve."""
    try:
        from mmcv.ops import modulated_deform_conv2d  # noqa: F401
        return 'mmcv'
    except Exception:
        pass
    from torch.nn.modules.utils import _pair
    from torchvision.ops import deform_conv2d as _tv

    def modulated_deform_conv2d(input, offset, mask, weight, bias, stride, padding,
                                dilation, groups, deform_groups):
        # DORNet passa input (1, b*Cin, H, W), pesi (b*Cout, Cin, k, k), groups=b,
        # offset/mask con il batch vero (b, ...): si riportano a (1, b*..., H, W).
        n, cin = input.shape[0], input.shape[1]
        kh, kw = weight.shape[-2:]
        if offset.shape[0] != n:
            offset = offset.reshape(n, -1, *offset.shape[-2:])
            mask = mask.reshape(n, -1, *mask.shape[-2:])
        assert cin // weight.shape[1] == groups
        assert offset.shape[1] // (2 * kh * kw) == deform_groups
        return _tv(input, offset, weight, bias, stride=_pair(stride), padding=_pair(padding),
                   dilation=_pair(dilation), mask=mask)

    mmcv = types.ModuleType('mmcv')
    ops = types.ModuleType('mmcv.ops')
    ops.modulated_deform_conv2d = modulated_deform_conv2d
    mmcv.ops = ops
    sys.modules['mmcv'] = mmcv
    sys.modules['mmcv.ops'] = ops
    return 'torchvision'


class DORNetWrapper(Wrapper):
    """DORNet (CVPR 2025): RGB min-max, depth / 5000. A test serve solo Net.srn (= SRN)."""
    key = 'dornet'
    pad_mult = 8
    default_norm = 5000
    tiny = False

    def build(self):
        repo = REPO_DIRS[self.key]
        if not (repo / 'net' / 'dornet_ddp.py').is_file():
            raise FileNotFoundError(f'DORNet non trovato in {repo}. Esegui setup.')
        self.extra['dcn_backend'] = _ensure_deform_conv()
        _add_path(repo)
        from net.dornet_ddp import SRN
        self.net = SRN(nfeats=24, reduction=4) if self.tiny else SRN(nfeats=64, reduction=16)

    def load(self, ckpt):
        if not ckpt:
            return {'ok': False, 'reason': 'checkpoint non indicato'}
        sd = _strip(_extract_state_dict(_torch_load(ckpt)), ('module.', 'srn.'))
        # i checkpoint RGB-D-D/NYU contengono anche 'Dab.*', usato SOLO in addestramento
        dab = [k for k in sd if k.startswith('Dab.')]
        for k in dab:
            del sd[k]
        info = _load_into(self.net, sd)
        info['dropped_training_only'] = len(dab)
        return info

    def prepare(self, s):
        rgb, pad = _pad_to(_to_t(_rgb_minmax(s['rgb']), self.device), self.pad_mult)
        lr_n, sc, off = _norm_depth(s['lr_mm'], self.norm_mode)
        lr, _ = _pad_to(_to_t(lr_n, self.device), self.pad_mult)
        return (lr, rgb), {'scale': sc, 'offset': off, 'pad': pad}

    def forward(self, inp):
        lr, rgb = inp
        out, _ = self.net(x_query=lr, rgb=rgb)
        return out


class DORNetTWrapper(DORNetWrapper):
    key = 'dornet_t'
    tiny = True


# --------------------------------------------------------------------------- DuCos
class DuCosWrapper(Wrapper):
    """DuCos (ICCV 2025): RGB min-max, depth / 5000 su TOFDSR. Il prior e' Depth Anything V2
    ViT-S congelato (net.depth_anything): i suoi tempi e parametri si riportano a parte."""
    key = 'ducos'
    default_norm = 5000

    def build(self):
        repo = REPO_DIRS['ducos']
        if not (repo / 'model' / 'ours' / 'ducos.py').is_file():
            raise FileNotFoundError(f'DuCos non trovato in {repo}. Esegui setup.')
        _add_path(repo)
        from model.ours.Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2

        dav2 = resolve(self.cfg.get('dav2_weights'))
        state = {'dav2': 'random'}

        def patched_load(f, *a, **k):
            if isinstance(f, str) and f.endswith('depth_anything_v2_vits.pth'):
                if dav2 is not None and dav2.is_file():
                    state['dav2'] = str(dav2)
                    return _torch_load(str(dav2))
                print('  [info] Depth Anything V2 ViT-S non trovato: inizializzo a caso '
                      '(lo sovrascrive il checkpoint di DuCos, se lo contiene)')
                return DepthAnythingV2(encoder='vits', features=64,
                                       out_channels=[48, 96, 192, 384]).state_dict()
            return _REAL_TORCH_LOAD(f, *a, **k)

        torch.load = patched_load
        try:
            from model.ours.ducos import depthprompting
            args = types.SimpleNamespace(prop_kernel=9, prop_time=18, conf_prop=True, loss='L1L2')
            self.net = depthprompting(args)          # come model_list.import_model
        finally:
            torch.load = _REAL_TORCH_LOAD
        self.extra['dav2_init'] = state['dav2']

    def prior_module(self):
        return self.net.depth_anything

    def prior_calls(self):
        # nel forward di DuCos il prior e' encoder ViT-S (get_intermediate_layers) + testa DPT
        da = self.net.depth_anything
        return [(da.pretrained, 'get_intermediate_layers'), (da.depth_head, 'forward')]

    def load(self, ckpt):
        if not ckpt:
            return {'ok': False, 'reason': 'checkpoint non indicato'}
        sd = _strip(_extract_state_dict(_torch_load(ckpt)))
        info = _load_into(self.net, sd, allow_missing_prefix='depth_anything.')
        if info.get('missing_ignored') and self.extra.get('dav2_init') == 'random':
            print('  [ATTENZIONE] il checkpoint non contiene Depth Anything V2 e il file DAv2 '
                  'manca: RMSE non valida. Esegui setup per scaricarlo.')
            info['ok'] = False
        return info

    def prepare(self, s):
        lr_n, sc, off = _norm_depth(s['lr_mm'], self.norm_mode)
        return {'rgb': _to_t(_rgb_minmax(s['rgb']), self.device), 'dep': _to_t(lr_n, self.device)}, \
            {'scale': sc, 'offset': off, 'pad': (0, 0)}

    def forward(self, inp):
        return self.net(inp)['pred']


WRAPPERS = {'bicubic': BicubicWrapper, 'dkn': DKNWrapper, 'fdkn': FDKNWrapper,
            'fdsr': FDSRWrapper, 'dctnet': DCTNetWrapper, 'suft': SUFTWrapper,
            'sgnet': SGNetWrapper, 'dornet_t': DORNetTWrapper, 'dornet': DORNetWrapper,
            'ducos': DuCosWrapper}


def build_wrapper(key, norm_mode, device, cfg):
    w = WRAPPERS[key](norm_mode, device, cfg)
    w.build()
    return w
