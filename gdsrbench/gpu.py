"""Lettura dei contatori della GPU tramite NVML (pacchetto nvidia-ml-py, modulo pynvml).

Energia: nvmlDeviceGetTotalEnergyConsumption, il contatore cumulativo della scheda in
millijoule dall'avvio del driver (GPU Volta e successive: 3090, 4090, H100, 4060 si').
Se il contatore non e' disponibile si ripiega sull'integrazione della potenza istantanea
(nvmlDeviceGetPowerUsage) campionata da un thread ogni 10 ms, e lo si dichiara nei risultati.

L'abbinamento fra il device di PyTorch e quello NVML avviene per UUID, perche' l'ordine
degli indici puo' differire (CUDA_VISIBLE_DEVICES, ordine PCI).
"""

import threading
import time


class NVML:
    def __init__(self, torch_device):
        import pynvml as nv
        import torch
        self.nv = nv
        nv.nvmlInit()
        self.handle = None
        self.matched_by = 'uuid'
        try:
            want = str(torch.cuda.get_device_properties(torch_device).uuid).lower()
            for i in range(nv.nvmlDeviceGetCount()):
                h = nv.nvmlDeviceGetHandleByIndex(i)
                u = nv.nvmlDeviceGetUUID(h)
                u = (u.decode() if isinstance(u, bytes) else u).lower()
                if u.replace('gpu-', '').replace('mig-', '') == want.replace('gpu-', ''):
                    self.handle = h
                    break
        except Exception:
            pass
        if self.handle is None:
            self.handle = nv.nvmlDeviceGetHandleByIndex(torch_device.index or 0)
            self.matched_by = 'index (verifica che sia la GPU giusta)'
        try:
            nv.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            self.energy_method = 'contatore di energia NVML'
        except Exception:
            self.energy_method = 'integrazione della potenza NVML (contatore non disponibile)'
        self._acc_mj = 0.0
        self._thr = None

    # ----------------------------------------------------------------- energia
    def energy_mj(self):
        """Energia cumulativa in mJ (contatore o integrale della potenza)."""
        if self.energy_method.startswith('contatore'):
            return float(self.nv.nvmlDeviceGetTotalEnergyConsumption(self.handle))
        return self._acc_mj

    def start_sampler(self):
        """Solo se manca il contatore: integra la potenza a 10 ms."""
        if self.energy_method.startswith('contatore') or self._thr is not None:
            return
        self._stop = False

        def loop():
            last = time.perf_counter()
            while not self._stop:
                time.sleep(0.01)
                now = time.perf_counter()
                p_mw = self.nv.nvmlDeviceGetPowerUsage(self.handle)
                self._acc_mj += p_mw * (now - last)       # mW * s = mJ
                last = now
        self._thr = threading.Thread(target=loop, daemon=True)
        self._thr.start()

    def stop_sampler(self):
        if self._thr is not None:
            self._stop = True
            self._thr.join()
            self._thr = None

    def power_w(self):
        return self.nv.nvmlDeviceGetPowerUsage(self.handle) / 1000.0

    # ----------------------------------------------------------------- stato
    def _safe(self, f, *a):
        try:
            return f(self.handle, *a)
        except Exception:
            return None

    def info(self):
        nv = self.nv
        d = {'nvml_match': self.matched_by, 'energy_method': self.energy_method}
        try:
            v = nv.nvmlSystemGetDriverVersion()
            d['driver'] = v.decode() if isinstance(v, bytes) else v
        except Exception:
            pass
        lim = self._safe(nv.nvmlDeviceGetEnforcedPowerLimit)
        d['power_limit_W'] = lim / 1000.0 if lim else None
        dl = self._safe(nv.nvmlDeviceGetPowerManagementDefaultLimit)
        d['power_limit_default_W'] = dl / 1000.0 if dl else None
        pm = self._safe(nv.nvmlDeviceGetPersistenceMode)
        d['persistence_mode'] = None if pm is None else bool(pm)
        d['clock_sm_max_MHz'] = self._safe(nv.nvmlDeviceGetMaxClockInfo, nv.NVML_CLOCK_SM)
        d['clock_mem_max_MHz'] = self._safe(nv.nvmlDeviceGetMaxClockInfo, nv.NVML_CLOCK_MEM)
        d['clock_sm_now_MHz'] = self._safe(nv.nvmlDeviceGetClockInfo, nv.NVML_CLOCK_SM)
        d['temperature_C'] = self._safe(nv.nvmlDeviceGetTemperature, nv.NVML_TEMPERATURE_GPU)
        d['pstate'] = self._safe(nv.nvmlDeviceGetPerformanceState)
        try:
            procs = nv.nvmlDeviceGetComputeRunningProcesses(self.handle)
            d['other_compute_processes'] = max(0, len(procs) - 1)
        except Exception:
            pass
        return d
