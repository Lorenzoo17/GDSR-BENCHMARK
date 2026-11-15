"""Protocollo di misura: accuratezza, latenza, memoria, energia (paper, Sez. IV-D/E/F).

LATENZA (IV-D)
  - batch 1, torch.no_grad(), modello in eval();
  - gli ingressi sono preparati e trasferiti sulla GPU PRIMA della misura: lettura dei file,
    preprocessing e copia host->device sono esclusi;
  - warm-up: le prime `warmup` immagini del test passano nella rete senza essere misurate
    (allocazioni, scelta dei kernel cuDNN, inizializzazione delle librerie);
  - ogni immagine e' misurata con CUDA events (start/end attorno al forward) e
    sincronizzazione esplicita; l'intero test set e' ripetuto `repeats` volte;
  - si riportano mediana e IQR su tutte le misure (immagini x ripetizioni), piu' media e
    deviazione standard, e la mediana di ogni ripetizione;
  - per le reti con un prior pre-addestrato (DuCos: Depth Anything V2) il tempo del prior e'
    misurato a parte con CUDA events attorno alle sue chiamate (encoder + testa);
    ricostruzione = totale - prior.
MEMORIA
  - torch.cuda.max_memory_allocated durante le passate misurate (pesi + attivazioni +
    buffer temporanei). Il contesto CUDA non e' incluso.
ENERGIA (IV-E)
  - contatore di energia cumulativo della scheda via NVML, letto prima e dopo, con
    sincronizzazione;
  - un sottoinsieme del test set (di default tutto) e' gia' sulla GPU e viene processato in
    ciclo finche' sono passati almeno `energy_s` secondi;
  - E_img = (C1 - C0) / N,   P_media = (C1 - C0) / dt;
  - prima di ogni ripetizione si misura la potenza a riposo P_idle per `idle_s` secondi
    (dopo `settle_s` secondi di attesa perche' la GPU scenda di clock);
    E_dyn = E_img - P_idle * dt / N;
  - `energy_repeats` ripetizioni, si riportano media e deviazione standard.
PRECISIONE (IV-F)
  - fp32 (riferimento): TF32 disattivato per matmul e convoluzioni cuDNN;
  - tf32: TF32 attivo (default di PyTorch per cuDNN);
  - fp16: autocast FP16 (TF32 disattivato per le operazioni rimaste in FP32).
  La RMSE e' ricalcolata con le uscite di quel device e di quella precisione.
FLOPs
  - torch.utils.flop_counter su un'immagine. Non conta operatori non standard (convoluzione
    deformabile di DORNet via torchvision, grid_sample): per quelle reti e' un limite inferiore.
"""

import os
import platform
import time

import numpy as np
import torch

from .datasets import rmse_cm


# =========================================================================== precisione
def set_precision(precision):
    tf32 = precision == 'tf32'
    try:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        torch.set_float32_matmul_precision('high' if tf32 else 'highest')
    except Exception:
        pass
    return {'precision': precision, 'allow_tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
            'allow_tf32_cudnn': torch.backends.cudnn.allow_tf32,
            'cudnn_benchmark': torch.backends.cudnn.benchmark}


class _autocast:
    def __init__(self, device, precision):
        self.ctx = (torch.autocast('cuda', dtype=torch.float16)
                    if precision == 'fp16' and device.type == 'cuda' else None)

    def __enter__(self):
        if self.ctx:
            self.ctx.__enter__()

    def __exit__(self, *a):
        if self.ctx:
            self.ctx.__exit__(*a)


# =========================================================================== timer
class _Timer:
    """CUDA events su GPU, perf_counter su CPU."""
    def __init__(self, device):
        self.cuda = device.type == 'cuda'
        self.device = device

    def mark(self):
        if self.cuda:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            return e
        return time.perf_counter()

    def ms(self, a, b):
        if self.cuda:
            return a.elapsed_time(b)
        return 1e3 * (b - a)

    def sync(self):
        if self.cuda:
            torch.cuda.synchronize(self.device)


class _PriorHook:
    """Misura il tempo del prior dentro il forward: avvolge i metodi indicati dal wrapper
    (es. encoder e testa di Depth Anything) con due CUDA events e somma le chiamate."""
    def __init__(self, calls, timer):
        self.t = timer
        self.events = []
        self.saved = []
        for obj, name in calls:
            orig = getattr(obj, name)
            had_own = name in obj.__dict__

            def wrapped(*a, _orig=orig, **k):
                s = self.t.mark()
                out = _orig(*a, **k)
                self.events.append((s, self.t.mark()))
                return out
            setattr(obj, name, wrapped)
            self.saved.append((obj, name, orig, had_own))

    def take_ms(self):
        v = sum(self.t.ms(a, b) for a, b in self.events)
        self.events = []
        return v

    def remove(self):
        for obj, name, orig, had_own in self.saved:
            if had_own:
                setattr(obj, name, orig)
            else:
                delattr(obj, name)


def _rss_mb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 2**20
    except Exception:
        return float('nan')


def _stats(a):
    a = np.asarray(a, dtype=np.float64)
    q25, q50, q75 = np.percentile(a, [25, 50, 75])
    return {'median': float(q50), 'q25': float(q25), 'q75': float(q75), 'iqr': float(q75 - q25),
            'mean': float(a.mean()), 'std': float(a.std()), 'min': float(a.min()), 'max': float(a.max()),
            'p95': float(np.percentile(a, 95))}


def count_flops(wrapper, inp, precision):
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except Exception:
        return None
    try:
        with torch.no_grad(), _autocast(wrapper.device, precision):
            fc = FlopCounterMode(display=False)
            with fc:
                wrapper.forward(inp)
        return fc.get_total_flops() / 1e9
    except Exception as e:
        print(f'  [info] conteggio FLOPs non riuscito: {type(e).__name__}: {e}')
        return None


def env_info(device):
    info = {'python': platform.python_version(), 'torch': torch.__version__,
            'os': f'{platform.system()} {platform.release()}', 'device': str(device),
            'host_cpu': platform.processor() or platform.machine()}
    try:
        import torchvision
        info['torchvision'] = torchvision.__version__
    except Exception:
        pass
    if device.type == 'cuda':
        p = torch.cuda.get_device_properties(device)
        info.update({'gpu': p.name, 'gpu_mem_GB': round(p.total_memory / 2**30, 1),
                     'sm_count': p.multi_processor_count, 'cuda': torch.version.cuda,
                     'cudnn': torch.backends.cudnn.version()})
    else:
        info['threads'] = torch.get_num_threads()
    return info


# =========================================================================== energia
def measure_energy(wrapper, dataset, precision, nvml, seconds, repeats, idle_s, settle_s, max_frames):
    dev = wrapper.device
    n_avail = len(dataset) if not max_frames else min(max_frames, len(dataset))
    inputs = []
    try:
        for i in range(n_avail):
            inputs.append(wrapper.prepare(dataset[i])[0])
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f'  [info] energia: memoria piena dopo {len(inputs)} immagini precaricate, uso quelle')
    if not inputs:
        return {'error': 'nessuna immagine precaricabile in memoria'}
    torch.cuda.synchronize(dev)
    nvml.start_sampler()
    runs = []
    for r in range(repeats):
        # ---- potenza a riposo
        time.sleep(settle_s)
        torch.cuda.synchronize(dev)
        e0, t0 = nvml.energy_mj(), time.perf_counter()
        time.sleep(idle_s)
        e1, t1 = nvml.energy_mj(), time.perf_counter()
        p_idle = (e1 - e0) / 1e3 / (t1 - t0)
        # ---- ciclo di inferenza
        n = 0
        torch.cuda.synchronize(dev)
        c0, t0 = nvml.energy_mj(), time.perf_counter()
        with torch.no_grad(), _autocast(dev, precision):
            done = False
            while not done:
                for inp in inputs:
                    wrapper.forward(inp)
                    n += 1
                    if time.perf_counter() - t0 >= seconds:
                        done = True
                        break
        torch.cuda.synchronize(dev)
        c1, t1 = nvml.energy_mj(), time.perf_counter()
        dt = t1 - t0
        e_tot = (c1 - c0) / 1e3                                  # J
        e_img = e_tot / n
        run = {'n_images': n, 'seconds': dt, 'P_idle_W': p_idle, 'P_mean_W': e_tot / dt,
               'E_img_J': e_img, 'E_dyn_J': e_img - p_idle * dt / n, 'throughput_img_s': n / dt}
        runs.append(run)
        print(f'  [energia] ripetizione {r + 1}/{repeats}: {n} immagini in {dt:.1f} s, '
              f'P media {run["P_mean_W"]:.1f} W (riposo {p_idle:.1f} W), '
              f'E/img {1e3 * e_img:.1f} mJ, E_dyn/img {1e3 * run["E_dyn_J"]:.1f} mJ', flush=True)
    nvml.stop_sampler()
    del inputs
    torch.cuda.empty_cache()
    out = {'frames_in_loop': n_avail, 'repeats': repeats, 'target_seconds': seconds,
           'idle_seconds': idle_s, 'settle_seconds': settle_s, 'method': nvml.energy_method, 'runs': runs}
    for k in ('E_img_J', 'E_dyn_J', 'P_mean_W', 'P_idle_W', 'throughput_img_s'):
        v = np.array([x[k] for x in runs])
        out[k] = {'mean': float(v.mean()), 'std': float(v.std())}
    return out


# =========================================================================== run
def run(wrapper, dataset, ckpt, *, warmup=10, repeats=3, precision='fp32', flops=True,
        base_alloc=None, energy=None, verbose_every=100):
    """energy: None oppure dict(seconds, repeats, idle_s, settle_s, max_frames)."""
    dev = wrapper.device
    timer = _Timer(dev)
    res = {'model': wrapper.key, 'checkpoint': str(ckpt) if ckpt else None, 'env': env_info(dev)}

    nvml = None
    if dev.type == 'cuda':
        try:
            from .gpu import NVML
            nvml = NVML(dev)
            res['env'].update(nvml.info())
        except Exception as e:
            print(f'  [info] NVML non disponibile ({type(e).__name__}: {e}): niente energia '
                  f'(pip install nvidia-ml-py)')

    # ---------------------------------------------------------------- pesi e parametri
    t0 = time.perf_counter()
    res['load'] = wrapper.load(str(ckpt) if ckpt else None)
    res['weights_loaded'] = bool(res['load'].get('ok'))
    wrapper.net.eval()
    rec, pri = wrapper.n_params()
    res['params_M'] = rec / 1e6                 # rete di ricostruzione (come nella Tabella I)
    res['prior_params_M'] = pri / 1e6           # prior congelato (es. Depth Anything V2)
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
        base = base_alloc if base_alloc is not None else torch.cuda.memory_allocated(dev)
    wrapper.net.to(dev)
    timer.sync()
    res['load_s'] = time.perf_counter() - t0
    if dev.type == 'cuda':
        res['vram_weights_MB'] = (torch.cuda.memory_allocated(dev) - base) / 2**20
    res.update(wrapper.extra)

    n = len(dataset)
    s0 = dataset[0]
    res['input_hw'] = list(s0['gt_mm'].shape)
    res['lr_hw'] = list(s0['lr_hw'])
    measure_rmse = res['weights_loaded']

    # ---------------------------------------------------------------- warm-up
    with torch.no_grad(), _autocast(dev, precision):
        for i in range(max(0, warmup)):
            inp, _ = wrapper.prepare(dataset[i % n])
            wrapper.forward(inp)
            del inp
    timer.sync()

    # ---------------------------------------------------------------- latenza + memoria + RMSE
    calls = wrapper.prior_calls()
    hook = _PriorHook(calls, timer) if calls else None
    if dev.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(dev)
    lat = np.zeros((repeats, n))
    lat_prior = np.zeros((repeats, n)) if hook else None
    rmses, prep_ms, load_ms = [], [], []
    rss_peak = _rss_mb()
    t_start = time.perf_counter()
    with torch.no_grad():
        for r in range(repeats):
            for i in range(n):
                a = time.perf_counter()
                s = dataset[i]
                b = time.perf_counter()
                inp, ctx = wrapper.prepare(s)
                timer.sync()
                c = time.perf_counter()
                with _autocast(dev, precision):
                    e0 = timer.mark()
                    out = wrapper.forward(inp)
                    e1 = timer.mark()
                timer.sync()
                lat[r, i] = timer.ms(e0, e1)
                if hook:
                    lat_prior[r, i] = hook.take_ms()
                if r == 0:
                    load_ms.append(1e3 * (b - a))
                    prep_ms.append(1e3 * (c - b))
                    if measure_rmse:
                        rmses.append(rmse_cm(wrapper.postprocess(out, ctx), s['gt_mm'], s['mask']))
                rss_peak = max(rss_peak, _rss_mb())
                del inp, out
                k = r * n + i + 1
                if verbose_every and (k % verbose_every == 0 or k == repeats * n):
                    msg = (f'  [{wrapper.key}] rip. {r + 1}/{repeats}  {i + 1:4d}/{n}  '
                           f'mediana {np.median(lat[:r + 1].ravel()[:k]):8.2f} ms')
                    if rmses:
                        msg += f'  RMSE {np.nanmean(rmses):.4f} cm'
                    print(msg, flush=True)
    res['latency_loop_s'] = time.perf_counter() - t_start
    if hook:
        hook.remove()

    res['n_frames'] = n
    res['repeats'] = repeats
    res['latency_ms'] = _stats(lat)
    res['latency_ms']['per_repeat_median'] = [float(np.median(x)) for x in lat]
    res['latency_ms_per_frame'] = lat.tolist()
    if hook:
        res['latency_prior_ms'] = _stats(lat_prior)
        res['latency_recon_ms'] = _stats(lat - lat_prior)
    res['fps'] = float(1e3 / res['latency_ms']['median'])
    res['preprocess_ms'] = float(np.mean(prep_ms))
    res['data_load_ms'] = float(np.mean(load_ms))
    res['rss_peak_MB'] = float(rss_peak)
    if dev.type == 'cuda':
        res['vram_peak_alloc_MB'] = torch.cuda.max_memory_allocated(dev) / 2**20
        res['vram_peak_reserved_MB'] = torch.cuda.max_memory_reserved(dev) / 2**20
    if rmses:
        res['rmse_cm'] = float(np.nanmean(rmses))
        res['rmse_per_frame'] = [float(x) for x in rmses]
    res['precision'] = precision

    # ---------------------------------------------------------------- energia
    if energy and nvml is not None:
        print(f'  [energia] ciclo di almeno {energy["seconds"]} s x {energy["repeats"]} ripetizioni '
              f'({nvml.energy_method})', flush=True)
        res['energy'] = measure_energy(wrapper, dataset, precision, nvml, **energy)
        res['env_after'] = nvml.info()
    elif energy and dev.type != 'cuda':
        res['energy'] = {'error': 'energia misurabile solo su GPU NVIDIA'}

    # ---------------------------------------------------------------- FLOPs
    if flops:
        inp, _ = wrapper.prepare(s0)
        res['gflops'] = count_flops(wrapper, inp, precision)
    return res
