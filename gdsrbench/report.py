"""Stampa e salvataggio dei risultati."""

import csv
import json
import math
import re
from datetime import datetime

from . import MODEL_NAMES, PUBLISHED, RESULTS


def _f(x, fmt):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return '-'
    return format(x, fmt)


def _g(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def print_table(dataset_name, results, failures):
    W = 118
    print()
    print('=' * W)
    print(f'  RISULTATI  -  {dataset_name.upper()}')
    print('=' * W)
    if results:
        r0 = results[0]
        env = r0['env']
        pf = r0.get('precision_flags', {})
        print(f"  GPU      : {env.get('gpu', env.get('host_cpu'))}   driver {env.get('driver', '-')}"
              f"   limite potenza {_f(env.get('power_limit_W'), '.0f')} W"
              f"   persistence mode {env.get('persistence_mode', '-')}")
        print(f"  software : torch {env['torch']}   CUDA {env.get('cuda', '-')}   cuDNN {env.get('cudnn', '-')}"
              f"   {env['os']}   host {env.get('host_cpu', '-')}")
        print(f"  protocollo: {r0['n_frames']} immagini x {r0['repeats']} ripetizioni, batch 1, "
              f"precisione {r0['precision']} (TF32 matmul {pf.get('allow_tf32_matmul')}, "
              f"cuDNN {pf.get('allow_tf32_cudnn')}), cudnn.benchmark {pf.get('cudnn_benchmark')}")
        print(f"  risoluzione: HR {r0['input_hw'][1]}x{r0['input_hw'][0]}, LR {r0['lr_hw'][1]}x{r0['lr_hw'][0]}")
        em = _g(r0, 'energy', 'method')
        if em:
            print(f"  energia  : {em}; ciclo >= {_g(r0, 'energy', 'target_seconds')} s x "
                  f"{_g(r0, 'energy', 'repeats')}, riposo {_g(r0, 'energy', 'idle_seconds')} s")

        # ---------------------------------------------------------- accuratezza e dimensioni
        print('-' * W)
        print('  ACCURATEZZA E DIMENSIONI')
        print(f"  {'rete':<10}{'pesi':>6}{'param M':>10}{'pubbl.':>8}{'prior M':>9}{'GFLOPs':>9}"
              f"{'RMSE cm':>10}{'pubbl.':>8}")
        for r in results:
            # i valori pubblicati sono su TOFDSR: su Redwood non si confrontano
            pub = PUBLISHED.get(r['model'], {}) if dataset_name == 'tofdsr' else {}
            loaded = r['weights_loaded'] and r['model'] != 'bicubic'
            print(f"  {MODEL_NAMES[r['model']]:<10}{('si' if loaded else ('-' if r['model'] == 'bicubic' else 'NO')):>6}"
                  f"{r['params_M']:>10.2f}{_f(pub.get('params_M'), '8.2f'):>8}"
                  f"{_f(r['prior_params_M'] or None, '9.2f'):>9}{_f(r.get('gflops'), '9.1f'):>9}"
                  f"{_f(r.get('rmse_cm'), '10.3f'):>10}{_f(pub.get('rmse_cm'), '8.2f'):>8}")

        # ---------------------------------------------------------- costo
        print('-' * W)
        print('  COSTO PER IMMAGINE')
        print(f"  {'rete':<10}{'lat. med ms':>12}{'IQR ms':>9}{'FPS':>8}{'VRAM pk MB':>12}{'VRAM pesi':>10}"
              f"{'RAM pk MB':>10}{'E/img mJ':>10}{'+/-':>7}{'E_dyn mJ':>10}{'P media W':>10}{'P riposo':>9}")
        for r in results:
            lat = r['latency_ms']
            e = r.get('energy') or {}

            def emj(k, st):
                v = _g(e, k, st)
                return 1e3 * v if isinstance(v, (int, float)) else None
            print(f"  {MODEL_NAMES[r['model']]:<10}{lat['median']:>12.3f}{lat['iqr']:>9.3f}{r['fps']:>8.1f}"
                  f"{_f(r.get('vram_peak_alloc_MB'), '12.0f'):>12}{_f(r.get('vram_weights_MB'), '10.0f'):>10}"
                  f"{r['rss_peak_MB']:>10.0f}{_f(emj('E_img_J', 'mean'), '10.1f'):>10}"
                  f"{_f(emj('E_img_J', 'std'), '7.1f'):>7}{_f(emj('E_dyn_J', 'mean'), '10.1f'):>10}"
                  f"{_f(_g(e, 'P_mean_W', 'mean'), '10.1f'):>10}{_f(_g(e, 'P_idle_W', 'mean'), '9.1f'):>9}")
        print('-' * W)
        print('  param M = rete di ricostruzione (prior congelato escluso, come nella Tabella I); prior M a parte')
        print('  lat. = mediana su immagini x ripetizioni, solo forward (CUDA events); IQR = q75 - q25')
        print('  VRAM pk = max_memory_allocated (contesto CUDA escluso); E/img = energia della scheda per immagine;')
        print('  E_dyn = E/img - P_riposo * dt / N.  RMSE "-" = pesi TOFDSR non disponibili: vale la colonna pubbl.')
        for r in results:
            notes = []
            if r.get('latency_prior_ms'):
                notes.append(f"prior {r['latency_prior_ms']['median']:.2f} ms + ricostruzione "
                             f"{r['latency_recon_ms']['median']:.2f} ms (mediane)")
            if r['model'] in ('dornet', 'dornet_t'):
                notes.append(f"conv. deformabile: {r.get('dcn_backend')} (non contata nei GFLOPs)")
            if r['model'] in ('dkn', 'fdkn'):
                notes.append('grid_sample non contato nei GFLOPs')
            if r['model'] == 'ducos':
                notes.append('RMSE pubblicata: modello addestrato su Hypersim; il checkpoint scaricato e\' RealTOFDSR')
            reps = r['latency_ms'].get('per_repeat_median')
            if reps and len(reps) > 1:
                notes.append('mediane per ripetizione ' + ', '.join(f'{x:.2f}' for x in reps) + ' ms')
            if notes:
                print(f"  {MODEL_NAMES[r['model']]}: " + ' | '.join(notes))
    for k, msg in failures.items():
        print(f'  {MODEL_NAMES.get(k, k)}: FALLITO -> {msg}')
    print('=' * W)


def save(dataset_name, results, failures, args_dict):
    RESULTS.mkdir(exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    gpu = results[0]['env'].get('gpu', 'cpu') if results else 'na'
    gpu = re.sub(r'[^A-Za-z0-9]+', '-', gpu).strip('-')
    prec = results[0].get('precision', 'fp32') if results else 'na'
    base = RESULTS / f'{dataset_name}_{prec}_{gpu}_{ts}'
    with open(base.with_suffix('.json'), 'w', encoding='utf-8') as f:
        json.dump({'dataset': dataset_name, 'args': args_dict, 'published': PUBLISHED,
                   'results': results, 'failures': failures}, f, indent=1)
    cols = ['model', 'gpu', 'precision', 'weights_loaded', 'n_frames', 'repeats',
            'params_M', 'params_published_M', 'prior_params_M', 'gflops',
            'rmse_cm', 'rmse_published_cm',
            'lat_median_ms', 'lat_q25_ms', 'lat_q75_ms', 'lat_iqr_ms', 'lat_mean_ms', 'lat_std_ms',
            'lat_prior_median_ms', 'lat_recon_median_ms', 'fps',
            'vram_peak_alloc_MB', 'vram_peak_reserved_MB', 'vram_weights_MB', 'rss_peak_MB',
            'E_img_mJ', 'E_img_std_mJ', 'E_dyn_mJ', 'E_dyn_std_mJ', 'P_mean_W', 'P_idle_W',
            'energy_throughput_img_s', 'energy_method', 'driver', 'power_limit_W', 'torch', 'cuda', 'cudnn',
            'checkpoint']
    with open(base.with_suffix('.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            lat, e, env = r['latency_ms'], r.get('energy') or {}, r['env']
            pub = PUBLISHED.get(r['model'], {})

            def mj(*k):
                v = _g(e, *k)
                return round(1e3 * v, 3) if isinstance(v, (int, float)) else ''
            w.writerow([MODEL_NAMES[r['model']], env.get('gpu', env['device']), r['precision'],
                        r['weights_loaded'], r['n_frames'], r['repeats'],
                        round(r['params_M'], 4), pub.get('params_M', ''), round(r['prior_params_M'], 4),
                        r.get('gflops'), r.get('rmse_cm', ''), pub.get('rmse_cm', ''),
                        lat['median'], lat['q25'], lat['q75'], lat['iqr'], lat['mean'], lat['std'],
                        _g(r, 'latency_prior_ms', 'median') or '', _g(r, 'latency_recon_ms', 'median') or '',
                        round(r['fps'], 2), r.get('vram_peak_alloc_MB'), r.get('vram_peak_reserved_MB'),
                        r.get('vram_weights_MB'), round(r['rss_peak_MB'], 1),
                        mj('E_img_J', 'mean'), mj('E_img_J', 'std'), mj('E_dyn_J', 'mean'), mj('E_dyn_J', 'std'),
                        _g(e, 'P_mean_W', 'mean') or '', _g(e, 'P_idle_W', 'mean') or '',
                        _g(e, 'throughput_img_s', 'mean') or '', e.get('method', ''),
                        env.get('driver', ''), env.get('power_limit_W', ''), env['torch'],
                        env.get('cuda', ''), env.get('cudnn', ''), r.get('checkpoint') or ''])
    return base
