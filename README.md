# Benchmark GDSR su TOFDSR: accuratezza, latenza, memoria, energia

Codice di misura per il paper *Accuracy, Latency, and Energy of Guided Depth
Super-Resolution Across GPU Architectures*. Implementa il protocollo della Sezione IV
(latenza, memoria, energia, precisione numerica) sul test reale di TOFDSR ×4 (560 immagini).
Funziona uguale su Windows e Ubuntu.

## Guida passo per passo

I comandi sono uguali su Ubuntu e Windows; dove cambiano ci sono entrambe le versioni.
Su Windows usa l'**Anaconda Prompt**, non il prompt normale.

### 0. Cosa serve prima

- **Driver NVIDIA** recente. Controllo: `nvidia-smi` deve mostrare la GPU e, in alto a
  destra, "CUDA Version" **12.1 o superiore** (driver ≥ 525). Se il comando non esiste, il
  driver va installato (sui server, dall'amministratore).
- **Conda** (Anaconda o Miniconda) e **git**.
- **La cartella del benchmark**, da GitHub (`git clone <url>`) o copiata a mano.
  Tutti i comandi seguenti si lanciano **dentro questa cartella**.

Il CUDA Toolkit **non** serve: PyTorch include già le sue librerie CUDA.

### 1. Crea l'ambiente conda

```bash
conda create -n gdsrbench python=3.10 -y
conda activate gdsrbench
```

D'ora in poi, **ogni volta** che apri un nuovo terminale: `conda activate gdsrbench`.

### 2. Installa PyTorch con CUDA

Sempre le stesse versioni su tutte le macchine (il paper dichiara software identico):

```bash
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
```

Controllo:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Deve stampare `2.3.1+cu121 True` e il nome della GPU. Se stampa `+cpu` o `False`, vedi
[Problemi frequenti](#problemi-frequenti).

### 3. Installa gli altri pacchetti

**Dopo** PyTorch (se lo fai prima, `kornia` si porta dietro un PyTorch solo-CPU):

```bash
pip install -r requirements.txt
```

### 4. Scarica il dataset TOFDSR

| Windows | Ubuntu |
|---|---|
| `download_tofdsr.bat --solo-test` | `bash download_tofdsr.sh --solo-test` |

Lo mette in `datasets/TOFDSR` e aggiorna da solo `config.json`. `--solo-test` estrae solo le
560 immagini di test, che sono tutto ciò che serve. Se il dataset c'è già e `config.json`
punta lì, non fa niente.

Se Google Drive rifiuta il download, lo script stampa i link ufficiali (Google Drive e
Baidu, codice `nust`): scarica l'archivio a mano e poi
`download_tofdsr.bat --da-archivio <file>` (oppure `--da-cartella <cartella estratta>`).

Licenza del dataset: solo uso accademico e non commerciale; va citato Yan et al., CVPR 2024
(Tri-Perspective View Decomposition).

### 5. Scarica le reti

| | Windows | Ubuntu |
|---|---|---|
| reti + pesi disponibili | `setup.bat` | `bash setup.sh` |
| reti senza pesi (tutte a pesi casuali) | `setup.bat --no-weights` | `bash setup.sh --no-weights` |

Clona i repository ufficiali in `external/` e, senza `--no-weights`, scarica i pesi di DuCos
e Depth Anything V2 in `weights/` e `external/DuCos/ckpts/`. Alla fine controlla PyTorch,
CUDA e NVML: per misurare l'energia deve comparire `[ok] NVML: contatore di energia
disponibile`. Si può rilanciare quando vuoi: scarica solo ciò che manca.

### 6. Prova veloce (pochi minuti)

Prima della corsa lunga, controlla che tutto funzioni su 20 immagini:

| Windows | Ubuntu |
|---|---|
| `run_tofdsr.bat --device cuda:0 --limit 20 --repeats 1 --energy-seconds 10` | `bash run_tofdsr.sh --device cuda:0 --limit 20 --repeats 1 --energy-seconds 10` |

Alla fine deve stampare le due tabelle con tutte e dieci le reti e nessuna riga `FALLITO`
(l'unica eccezione attesa è SGNet in FP16, vedi sotto).

### 7. Corsa completa per il paper

| Windows | Ubuntu |
|---|---|
| `run_paper.bat --device cuda:0` | `bash run_paper.sh --device cuda:0` |

Esegue tutto il protocollo prima in FP32 (riferimento) e poi in FP16. Per far girare **tutte**
le reti con pesi casuali, come nella campagna sulla RTX 3090, aggiungi
`--ckpt dornet=nessuno --ckpt dornet_t=nessuno` (DORNet e DORNet-T sono le uniche reti di cui
il setup trova sempre i pesi, perché stanno nel loro repository).

Durata: circa **5–8 minuti per rete**, quindi circa un'ora per precisione e due ore in tutto.
Durante la misura **non usare la GPU per altro**: tutto finisce nel contatore di energia.
Su un server via SSH, lancia in modo che non si interrompa se chiudi la connessione:

```bash
nohup bash run_paper.sh --device cuda:0 > log_benchmark.txt 2>&1 &
tail -f log_benchmark.txt
```

### 8. Risultati

Alla fine di ogni precisione vengono stampate due tabelle (accuratezza/dimensioni e costo
per immagine) e salvati in `results/`:

- `tofdsr_<precisione>_<gpu>_<data>.csv`: una riga per rete, pronta per le tabelle del paper;
- `tofdsr_<precisione>_<gpu>_<data>.json`: tutto, comprese latenza e RMSE di ogni immagine
  e lo stato della GPU (driver, potenza, clock, temperatura).

## Reti incluse

Criterio: **RMSE pubblicata sul TOFDSR reale + codice ufficiale disponibile**. I pesi
addestrati su TOFDSR non sono richiesti: latenza, memoria ed energia non dipendono dal valore
dei pesi, e per le reti senza pesi si riporta la RMSE pubblicata.

| chiave | rete | codice | RMSE pubbl. (cm) | param. pubbl. (M) | pesi TOFDSR |
|---|---|---|---|---|---|
| `bicubic` | Bicubic (riferimento non appreso) | — | — | — | — |
| `dkn` | DKN (IJCV 2021) | cvlab-yonsei/dkn | 5.50 | 1.16 | no |
| `fdkn` | FDKN (IJCV 2021) | cvlab-yonsei/dkn | 5.77 | 0.69 | no |
| `fdsr` | FDSR (CVPR 2021) | lingzhi96/RGB-D-D-Dataset | 5.03 | 0.60 | no |
| `dctnet` | DCTNet (CVPR 2022) | Zhaozixiang1228/GDSR-DCTNet | 5.16 | 0.48 | no |
| `suft` | SUFT (ACM MM 2022) | ShiWuxuan/SUFT | 4.37 | 22.01 | no |
| `sgnet` | SGNet (AAAI 2024) | yanzq95/SGNet | 4.33 | 8.97 | no |
| `dornet_t` | DORNet-T (CVPR 2025) | yanzq95/DORNet | 4.87 | 0.46 | **sì** |
| `dornet` | DORNet (CVPR 2025) | yanzq95/DORNet | 4.21 | 3.05 | **sì** |
| `ducos` | DuCos (ICCV 2025) | yanzq95/DuCos | 4.29 | 34.38 (con DAv2) | **sì** (RealTOFDSR) |

Valori pubblicati: Tab. 1 di DORNet (CVPR 2025) per le prime otto reti; DuCos dal proprio paper.

**Escluse:** SFG (nessun codice pubblico trovato), DJF e DJFR (solo MATLAB/MatConvNet),
CUNet (codice non trovato), PromptDA e PriorDA (nessuna RMSE pubblicata su TOFDSR).

### Comandi utili

```bash
bash run_tofdsr.sh --device cuda:0 --models sgnet dctnet            # solo alcune reti
bash run_tofdsr.sh --device cuda:0 --precision fp16                 # solo FP16
bash run_tofdsr.sh --device cuda:0 --ckpt fdsr=/percorso/pesi.pth   # pesi tuoi per una rete
python run_benchmark.py --help                                      # tutte le opzioni
```

Per rilanciare una sola rete fallita non serve rifare tutto: `--models <rete>` con la stessa
precisione, poi copia la sua riga nel `.csv` della corsa completa.

### Problemi frequenti

| sintomo | causa | soluzione |
|---|---|---|
| `nvidia-smi` non funziona | driver NVIDIA assente | va installato (≥ 525) |
| torch stampa `+cpu`, o `cuda.is_available()` è `False` | PyTorch solo-CPU (spesso perché `requirements.txt` è stato installato prima di PyTorch) | `pip uninstall -y torch torchvision`, poi ripeti il passo 2 |
| `cuda.is_available()` è `False` con torch `+cu121` | driver troppo vecchio | aggiorna il driver ("CUDA Version" ≥ 12.1 in `nvidia-smi`) |
| `expected a torch.device with a specified index` | dispositivo senza indice | aggiungi `--device cuda:0` |
| `No module named ...` per una rete | pacchetto mancante | `pip install -r requirements.txt` nell'ambiente attivo |
| niente energia, `NVML non disponibile` | manca `nvidia-ml-py` | `pip install nvidia-ml-py` |
| SGNet `FALLITO` in FP16 | la FFT in FP16 accetta solo lati potenza di 2, e 384 non lo è | atteso: in FP16 SGNet si riporta come n/a |
| la potenza a riposo sembra troppo alta (> 100 W) | la GPU non è ancora tornata a riposo dopo il carico | `--settle-seconds 60`; sui server attiva `sudo nvidia-smi -pm 1` |

## Cosa misura (corrispondenza con il paper)

| quantità | come | sezione |
|---|---|---|
| **RMSE** | per immagine, poi media; gt in [0.1, 5.0] m, bordo di 6 px escluso (= `tofdsr_calc_rmse` di DORNet/DuCos). Solo per le reti con pesi TOFDSR; per le altre vale la colonna *pubbl.* | IV-A |
| **Latenza** | batch 1, `no_grad`, ingressi già sulla GPU; warm-up di 10 immagini; CUDA events + sincronizzazione per ognuna delle 560 immagini; 3 ripetizioni. Si riportano **mediana e IQR** (più media, dev. std. e mediana di ogni ripetizione) | IV-D |
| **Prior separato** | per DuCos il tempo di Depth Anything V2 (encoder + testa) è misurato a parte; ricostruzione = totale − prior | IV-D |
| **Memoria di picco** | `torch.cuda.max_memory_allocated` durante le passate misurate (contesto CUDA escluso) | IV-D |
| **Energia** | contatore cumulativo NVML della scheda, letto prima e dopo un ciclo di ≥ 60 s sul test set già in GPU; `E_img = (C1−C0)/N`, `P = (C1−C0)/Δt`; potenza a riposo misurata per 10 s prima di ogni ripetizione (dopo 5 s di attesa), `E_dyn = E_img − P_idle·Δt/N`; 3 ripetizioni, media ± dev. std. | IV-E |
| **Precisione** | `fp32` = riferimento con **TF32 disattivato**; `fp16` = autocast; (`tf32` = default di PyTorch, per confronto). La RMSE è ricalcolata in ogni configurazione | IV-F |
| **Parametri** | rete di ricostruzione, **escluso il prior pre-addestrato** (come la Tab. I); il prior è riportato a parte | IV-B |
| **Ambiente** | driver, limite di potenza, persistence mode, clock massimi, temperatura, pstate, versioni di torch/CUDA/cuDNN; la GPU NVML è abbinata a quella di PyTorch tramite UUID | IV-C |

Tutte le durate e i conteggi si cambiano da riga di comando (`--warmup`, `--repeats`,
`--energy-seconds`, `--energy-repeats`, `--idle-seconds`, `--settle-seconds`): `--help`
per l'elenco.

Ogni rete gira in un **processo separato**: il picco di memoria di una non contamina le
altre e un errore (es. memoria esaurita) non ferma il resto.

Preprocessing = quello del test ufficiale di ciascuna rete:

| rete | RGB | depth in ingresso |
|---|---|---|
| DKN, FDKN, FDSR | BGR / 255 (cv2) | LR ingrandita a HR, min-max |
| DCTNet | min-max | LR ingrandita a HR, min-max; uscita tagliata in [0,1] |
| SUFT, SGNet | min-max | LR a H/4 × W/4, min-max |
| DORNet, DORNet-T | min-max | LR ingrandita a HR, / 5000 |
| DuCos | min-max | LR ingrandita a HR, / 5000 |
| Bicubic | — | LR → HR con `F.interpolate` bicubico sulla GPU |

## Da sapere (e da scrivere nel paper)

1. **Pesi e RMSE.** Pesi TOFDSR pubblici esistono solo per DORNet, DORNet-T e DuCos. Le
   altre reti girano con pesi casuali: tempi, memoria ed energia sono quelli veri (le
   operazioni eseguite non dipendono dai valori dei pesi), la RMSE è quella pubblicata.
2. **DuCos.** La RMSE pubblicata (4.29) è di un modello **addestrato su Hypersim** e testato
   sul TOFDSR reale; il checkpoint scaricabile si chiama `RealTOFDSR`, quindi la RMSE misurata
   può differire: va spiegato nella nota della Tab. I. DuCos aggiorna anche i *bias* di Depth
   Anything (bias tuning): "congelato" vale per i pesi, non per i bias.
3. **Parametri misurati vs pubblicati.** DORNet 2.92 M vs 3.05 e DORNet-T 0.43 vs 0.46: la
   differenza è il modulo di degradazione (`Dab`), usato solo in addestramento. SGNet: 9.22 M
   con `num_feats=24`, la configurazione real-world (DORNet riporta 8.97 M). **Nella bozza la
   Tab. I riporta 35.42 M per SGNet: è la versione `num_feats=48` per NYU, da correggere.**
4. **DORNet senza mmcv.** Se `mmcv` non è installato la convoluzione deformabile usa
   `torchvision.ops.deform_conv2d` (stessa operazione; 4.2117 cm verificato). Il backend è
   scritto nei risultati: usa lo stesso su tutte le GPU.
5. **DCTNet** usa `torch.rfft`, rimosso in PyTorch 1.8: il benchmark lo ridefinisce con
   `torch.fft` (stessa trasformata), senza modificare il loro codice.
6. **FLOPs**: `torch.utils.flop_counter` non conta la convoluzione deformabile (DORNet) né
   `grid_sample` (DKN/FDKN): per quelle reti è un limite inferiore.
7. **Energia nel ciclo**: le immagini sono tutte precaricate sulla GPU (circa 1.8 GB a
   512×384). Se la memoria non basta il ciclo usa quelle che ci stanno, e lo stampa. Se la GPU
   non espone il contatore di energia, si integra la potenza campionata ogni 10 ms (scritto
   nei risultati). Sul portatile la GPU può pilotare anche lo schermo: per i numeri del paper
   usa le macchine desktop/server.
8. **cuDNN benchmark** è attivo (la scelta dei kernel avviene nel warm-up);
   `--no-cudnn-benchmark` lo disattiva.

## Redwood (solo per la tesi)

`run_redwood.bat` / `bash run_redwood.sh` esegue lo stesso protocollo su Redwood
(formulazione B, ×4; percorsi in `config.json` → `redwood`). Pesi RGB-D-D per DORNet,
DORNet-T, DuCos e DCTNet; la RMSE è semplice (maschera LiDAR + bordo), non quella per regioni.
