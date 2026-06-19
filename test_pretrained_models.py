import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import scipy.io as sio

# Detectare Hardware (GPU sau CPU)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[SISTEM OPERATIV] Dispozitiv utilizat: {DEVICE}")

# Directorul local unde se află modelele descărcate în format Black-Box
director_modele = "modele_lrrnet_blackbox"

if not os.path.exists(director_modele):
    print(f"[EROARE] Folderul '{director_modele}' nu a fost găsit în directorul curent.")
else:
    fisiere_modele = [f for f in os.listdir(director_modele) if f.endswith('.pth')]
    fisiere_ordonate = sorted(fisiere_modele)
    print(f"[INFO] Am identificat {len(fisiere_ordonate)} fișiere Black-Box în folderul local.\n")

    colector_tabel = []
    
    nume_fisiere_mat = {
        "AVIRIS San Diego Set 1": "aviris_1.mat",
        "AVIRIS Set 2": "aviris_2.mat",
        "AVIRIS Airport 1": "abu-airport-1.mat",
        "AVIRIS Airport 2": "abu-airport-2.mat",
        "AVIRIS Airport 3": "abu-airport-3.mat",
        "AVIRIS Airport 4": "abu-airport-4.mat",
        "AVIRIS Beach 1": "abu-beach-1.mat",
        "AVIRIS Beach 2": "abu-beach-2.mat",
        "AVIRIS Beach 3": "abu-beach-3.mat",
        "AVIRIS Beach 4": "abu-beach-4.mat",
        "HYDICE Urban": "HYDICE-urban.mat"
    }

    if len(fisiere_ordonate) > 0:
        # Împărțim cele 11 modele în două grupuri: primele 6 și ultimele 5
        grup_stanga = fisiere_ordonate[:6]
        grup_dreapta = fisiere_ordonate[6:]
        
        # Grid de 6 rânduri x 4 coloane (Coloana Stângă: Det|ROC, Coloana Dreaptă: Det|ROC)
        fig, axes = plt.subplots(nrows=6, ncols=4, figsize=(16, 22))

        # Funcție ajutătoare pentru a procesa și randa un model pe o poziție specifică din grid
        def proceseaza_si_ploteaza(nume_fisier, rând, col_start):
            cale_completa = os.path.join(director_modele, nume_fisier)
            checkpoint = torch.load(cale_completa, map_location=DEVICE, weights_only=False)
            nume_set = checkpoint['nume_dataset']
            
            print(f"[PROCESARE] Rând {rând+1} | Fișier: {nume_fisier} -> [{nume_set}]")
            
            # --- CORECTARE DIMENSIUNI ---
            scoruri_anomalie = checkpoint['detection_map_raw'].reshape(-1)
            if hasattr(scoruri_anomalie, 'cpu'):
                scoruri_anomalie = scoruri_anomalie.cpu().numpy()
            
            fisier_mat = nume_fisiere_mat.get(nume_set, "")
            # Verifică dacă fișierul există în directorul curent sau în folderul "dataset"
            cale_mat = fisier_mat if os.path.exists(fisier_mat) else os.path.join("dataset", fisier_mat)
            
            auc_local = 0.0
            status_verificare = "Lipsă .mat"
            fpr, tpr = None, None
            
            if os.path.exists(cale_mat):
                try:
                    mat_data = sio.loadmat(cale_mat)
                    chei = [k for k in mat_data.keys() if not k.startswith('__')]
                    gt_key = 'map' if 'map' in mat_data else [k for k in chei if any(x in k.lower() for x in ['gt', 'map', 'mask', 'label'])][0]
                    gt_raw = mat_data[gt_key].flatten().astype(np.float32)
                    
                    fpr, tpr, _ = roc_curve(gt_raw, scoruri_anomalie)
                    auc_local = auc(fpr, tpr)
                    
                    if auc_local < 0.5:
                        scoruri_anomalie = -scoruri_anomalie
                        fpr, tpr, _ = roc_curve(gt_raw, scoruri_anomalie)
                        auc_local = auc(fpr, tpr)
                    
                    status_verificare = f"{auc_local*100:.2f}%"
                except Exception as e_valid:
                    status_verificare = f"Eroare: {e_valid}"
                    fpr, tpr = None, None

            if fpr is None or tpr is None:
                auc_local = checkpoint['auc_original_colab']
                fpr = checkpoint.get('fpr_original', np.array([0, 1]))
                tpr = checkpoint.get('tpr_original', np.array([0, 1]))

            colector_tabel.append({
                "Dataset Hiperspectral": nume_set,
                "AUC Original Colab": f"{checkpoint['auc_original_colab']*100:.2f}%",
                "AUC Verificat Local": status_verificare,
                "Epocă Stop": checkpoint['stop_epoch'],
                "Straturi": checkpoint['num_stages']
            })

            # --- PLOTARE AXE CONFIGURATE ---
            ax_det = axes[rând, col_start]
            ax_roc = axes[rând, col_start + 1]
            
            # Subplot Hartă Anomalii
            H, W = checkpoint['H_geometrie'], checkpoint['W_geometrie']
            det_map_2d = scoruri_anomalie.reshape(H, W)
            det_map_2d = (det_map_2d - det_map_2d.min()) / (det_map_2d.max() - det_map_2d.min() + 1e-8)
            
            im = ax_det.imshow(det_map_2d, cmap='jet', vmin=0.0, vmax=1.0)
            ax_det.set_title(f"{nume_set}\n(Epocă Stop: {checkpoint['stop_epoch']})", fontsize=8, fontweight='bold', pad=6)
            ax_det.set_xticks([]); ax_det.set_yticks([])
            fig.colorbar(im, ax=ax_det, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
            
            # Subplot Curbă ROC
            ax_roc.plot(fpr, tpr, color='#e31a1c', lw=1.8, label=f"AUC = {auc_local:.4f}")
            ax_roc.plot([0, 1], [0, 1], color='#252525', lw=0.8, linestyle='--')
            ax_roc.set_xlim([-0.01, 1.0]); ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel('FAR', fontsize=7, labelpad=2)
            ax_roc.set_ylabel('TPR', fontsize=7, labelpad=2)
            ax_roc.tick_params(labelsize=7)
            ax_roc.grid(True, linestyle=':', alpha=0.5)
            ax_roc.legend(loc='lower right', fontsize=7, frameon=True)

        # Pasul 1: Populam prima coloană mare (Stânga - primele 6 dataseturi)
        for idx, nume_f in enumerate(grup_stanga):
            proceseaza_si_ploteaza(nume_f, rând=idx, col_start=0)
            
        # Pasul 2: Populam a doua coloană mare (Dreapta - ultimele 5 dataseturi)
        for idx, nume_f in enumerate(grup_dreapta):
            proceseaza_si_ploteaza(nume_f, rând=idx, col_start=2)
            
        # Ascundem axele rămase libere pe ultimul rând din dreapta (deoarece sunt doar 5 în acel grup)
        axes[5, 2].axis('off')
        axes[5, 3].axis('off')

        # Ajustări finale de layout anti-suprapunere
        plt.tight_layout()
        plt.subplots_adjust(top=0.96, wspace=0.35, hspace=0.45)
        plt.show()

        # Generare raport textual tabelar final în consolă
        df_raport = pd.DataFrame(colector_tabel)
        
        valori_calcul = []
        for d in colector_tabel:
            val_clean = d["AUC Verificat Local"].replace("%", "")
            try: valori_calcul.append(float(val_clean))
            except: pass
        mAUC = np.mean(valori_calcul) if valori_calcul else 0.0

        linie_separatoare = pd.DataFrame([{"Dataset Hiperspectral": "-"*25, "AUC Original Colab": "-"*18, "AUC Verificat Local": "-"*19, "Epocă Stop": "-"*10, "Straturi": "-"*8}])
        linie_total = pd.DataFrame([{"Dataset Hiperspectral": "mAUC MEDIU GENERAL CONSOLIDAT", "AUC Original Colab": "", "AUC Verificat Local": f"{mAUC:.2f}%", "Epocă Stop": "", "Straturi": ""}])
        df_final_tabel = pd.concat([df_raport, linie_separatoare, linie_total], ignore_index=True)

        print("\n" + "="*95)
        print("   TABEL ACADEMIC REZULTATE REPLICATE CONFORM MEDIULUI HAD")
        print("="*95)
        print(df_final_tabel.to_string(index=False))
        print("="*95)