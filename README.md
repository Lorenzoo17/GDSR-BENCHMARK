# Benchmark GDSR su TOFDSR: accuratezza, latenza, memoria, energia

Codice di misura per il paper *Accuracy, Latency, and Energy of Guided Depth
Super-Resolution Across GPU Architectures*. Implementa il protocollo della Sezione IV
(latenza, memoria, energia, precisione numerica) sul test reale di TOFDSR ×4 (560 immagini).
Funziona uguale su Windows e Ubuntu.

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

## 1. Una volta sola: ambiente

PyTorch va installato **con CUDA** dal sito ufficiale:

```bash
# esempio CUDA 12.1 — comando giusto su https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Poi il setup (pacchetti + download delle reti):

| Windows | Ubuntu |
|---|---|
| `setup.bat --pip` | `bash setup.sh --pip` |

Su Windows lancia i `.bat` da un prompt con l'ambiente conda attivo (o imposta la variabile
`PYTHON`). Il setup si può rilanciare quando vuoi: scarica solo ciò che manca. Le cartelle
di PromptDA, PriorDA e i loro pesi, se le avevi scaricate, non servono più: puoi cancellare
`external/PromptDA`, `external/PriorDA`, `weights/promptda`, `weights/priorda`.

## 2. Dataset TOFDSR

Se il dataset non c'è, lo scarica e lo sistema uno script:

| Windows | Ubuntu |
|---|---|
| `download_tofdsr.bat --solo-test` | `bash download_tofdsr.sh --solo-test` |

Scarica l'archivio ufficiale da Google Drive (pagina del progetto TOFDC), lo estrae in
`datasets/TOFDSR`, controlla che ci siano le 560 immagini di test e aggiorna `tofdsr.root`
in `config.json` (copia del vecchio in `config.json.bak`). `--solo-test` estrae solo il test,
che è tutto ciò che serve al benchmark; senza, estrae anche il train. Se `config.json`
punta già a un TOFDSR completo, non fa niente.

Se Google Drive rifiuta il download (succede con i file molto scaricati), scarica l'archivio
a mano dal link di Google Drive o di Baidu (codice `nust`) della pagina del progetto e poi:
`download_tofdsr.bat --da-archivio C:\percorso\archivio.zip`. Con una cartella già estratta:
`--da-cartella`. Altra destinazione: `--dest`.

Licenza del dataset: solo uso accademico e non commerciale; va citato Yan et al., CVPR 2024
(Tri-Perspective View Decomposition).

In alternativa il percorso si imposta a mano in `config.json` → `tofdsr.root` (relativo a
questa cartella o assoluto), oppure con `--tofdsr-root`. La cartella deve contenere
`Test/RGB`, `Test/GT`, `Test/LR_Filled` e, se c'è, `TOFDSR_Test.txt`.

## 3. Eseguire

| | Windows | Ubuntu |
|---|---|---|
| protocollo completo, FP32 (riferimento) | `run_tofdsr.bat` | `bash run_tofdsr.sh` |
| FP32 **e** FP16, una dopo l'altra | `run_paper.bat` | `bash run_paper.sh` |

```bash
run_tofdsr.bat --limit 20 --repeats 1 --energy-seconds 10    # prova veloce (minuti)
run_tofdsr.bat --models dornet ducos                         # solo alcune reti
run_tofdsr.bat --precision fp16                              # seconda configurazione
run_tofdsr.bat --ckpt fdsr=C:\pesi\fdsr_tofdsr.pth           # se hai pesi tuoi per una rete
```

Durata del protocollo completo: per ogni rete 10 immagini di warm-up, 3 × 560 misure di
latenza, poi 3 cicli di energia da ≥ 60 s, ognuno preceduto da 5 + 10 s di riposo. Conta
circa **5–8 minuti per rete**, quindi circa un'ora per GPU e per precisione (DKN e SUFT sono
le più lente). Durante la misura **non usare la GPU per altro**: tutto finisce nel contatore
di energia.

Alla fine stampa due tabelle (accuratezza/dimensioni e costo per immagine) e salva
`results/tofdsr_<precisione>_<gpu>_<data>.json` (tutto, comprese latenza e RMSE di ogni
immagine e lo stato della GPU) e `.csv` (una riga per rete, pronta per le tabelle del paper).

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
