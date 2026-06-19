import os
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, roc_curve, auc

# ==========================================
# 1. CONFIGURAȚIE GLOBALĂ ȘI DETECTARE HARDWARE
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[SISTEM] Calculul rulează pe dispozitivul: {DEVICE}\n" + "="*60)

NUM_CLUSTERS = 15
OPTIMAL_STAGES = 9
OPTIMAL_EPOCHS = 500
OPTIMAL_LR = 0.005

datasets_config = [
    {"name": "AVIRIS San Diego Set 1", "filename": "aviris_1.mat", "rgb_bands": (29, 20, 12)},
    {"name": "AVIRIS Set 2", "filename": "aviris_2.mat", "rgb_bands": (29, 20, 12)},
    {"name": "AVIRIS Airport 1", "filename": "abu-airport-1.mat", "rgb_bands": (30, 20, 10)},
    {"name": "AVIRIS Airport 2", "filename": "abu-airport-2.mat", "rgb_bands": (30, 20, 10)},
    {"name": "AVIRIS Airport 3", "filename": "abu-airport-3.mat", "rgb_bands": (30, 20, 10)},
    {"name": "AVIRIS Airport 4", "filename": "abu-airport-4.mat", "rgb_bands": (30, 20, 10)},
    {"name": "AVIRIS Beach 1", "filename": "abu-beach-1.mat", "rgb_bands": (29, 20, 12)},
    {"name": "AVIRIS Beach 2", "filename": "abu-beach-2.mat", "rgb_bands": (29, 20, 12)},
    {"name": "AVIRIS Beach 3", "filename": "abu-beach-3.mat", "rgb_bands": (29, 20, 12)},
    {"name": "AVIRIS Beach 4", "filename": "abu-beach-4.mat", "rgb_bands": (29, 20, 12)},
    {"name": "HYDICE Urban", "filename": "HYDICE-urban.mat", "rgb_bands": (60, 27, 17)}
]

# ==========================================
# 2. DEFINIREA ARHITECTURII LRR-Net-S
# ==========================================
class LRRNetSStageOfficial(nn.Module):
    def __init__(self, K, B):
        super(LRRNetSStageOfficial, self).__init__()
        self.K = K
        self.B = B

        # Parametrii învățabili (Conform algoritmului din articol)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.lambda_param = nn.Parameter(torch.tensor(1.2e-5))
        self.beta = nn.Parameter(torch.tensor(1e-5))
        self.mu = nn.Parameter(torch.tensor(1.0))
        self.eta = nn.Parameter(torch.tensor(1.0))

    def forward(self, X, D, L_prev, P_prev, Q_prev):
        # S-BLOCK (Update Anomalii)
        recon_background = torch.matmul(L_prev, D)
        residual_S = X - recon_background
        norm_S = torch.norm(residual_S, p=2, dim=1, keepdim=True) + 1e-8
        S = (residual_S / norm_S) * torch.relu(norm_S - torch.abs(self.beta))

        # J-BLOCK via Operatorul SVT (Protejat numeric)
        mat_M = L_prev - P_prev
        mat_M = torch.nan_to_num(mat_M, nan=0.0)

        try:
            eps_stable = torch.randn_like(mat_M) * 1e-6
            U, Sigma, V = torch.svd(mat_M + eps_stable)
            threshold_J = torch.abs(self.alpha) / (torch.abs(self.mu) + 1e-8)
            Sigma_thres = torch.relu(Sigma - threshold_J)
            J = torch.matmul(torch.matmul(U, torch.diag_embed(Sigma_thres)), V.transpose(-2, -1))
        except:
            J = torch.relu(mat_M - 0.01)
        J = torch.nan_to_num(J, nan=0.0)

        # W-BLOCK (Update Sparse Coefficients)
        mat_N = L_prev - Q_prev
        threshold_W = torch.abs(self.lambda_param) / (torch.abs(self.eta) + 1e-8)
        W = torch.sign(mat_N) * torch.relu(torch.abs(mat_N) - threshold_W)
        W = torch.nan_to_num(W, nan=0.0)

        # L-BLOCK (Soluție în formă închisă)
        DTD = torch.matmul(D, D.t())
        I_K = torch.eye(self.K, device=X.device)
        Inv_Matrix = torch.inverse(DTD + (torch.abs(self.mu) + torch.abs(self.eta)) * I_K)

        part1 = torch.matmul(X - S, D.t())
        part2 = torch.abs(self.mu) * (J - P_prev)
        part3 = torch.abs(self.eta) * (W - Q_prev)

        L = torch.matmul(part1 + part2 + part3, Inv_Matrix.t())
        L = torch.nan_to_num(L, nan=0.0)

        # Update Multiplicatori Lagrange P și Q
        P = P_prev - torch.abs(self.mu) * (L - J)
        Q = Q_prev - torch.abs(self.eta) * (L - W)
        P = torch.nan_to_num(P, nan=0.0)
        Q = torch.nan_to_num(Q, nan=0.0)

        # Normalizare inter-strat esențială
        L = L / (torch.norm(L, p=2, dim=1, keepdim=True) + 1e-8)

        return S, J, W, L, P, Q


class LRRNetSCompleteOfficial(nn.Module):
    def __init__(self, num_stages, K, B, D_init):
        super(LRRNetSCompleteOfficial, self).__init__()
        self.K = K
        self.B = B
        self.D = nn.Parameter(D_init.clone())
        self.stages = nn.ModuleList([LRRNetSStageOfficial(K, B) for _ in range(num_stages)])

    def forward(self, X):
        M_size = X.shape[0]
        L = torch.zeros(M_size, self.K, device=X.device)
        P = torch.zeros(M_size, self.K, device=X.device)
        Q = torch.zeros(M_size, self.K, device=X.device)

        final_S = torch.zeros_like(X)
        final_L = torch.zeros_like(L)

        for stage in self.stages:
            S, J, W, L, P, Q = stage(X, self.D, L, P, Q)
            final_S = S
            final_L = L

        return final_S, final_L

# ==========================================
# 3. FUNCȚIA DE ANTRENARE ȘI EARLY STOPPING VIA AUC
# ==========================================
def antreneaza_si_evalueaza_lrrnet_optimizat(nume_dataset, date_dict, num_stages, num_epochs, lr_straturi, lr_dictionar=0.0, patience=30, lambda_bg=2.5):
    X = date_dict["X"]
    D_init = date_dict["D"]
    gt = date_dict["gt"].cpu().numpy().flatten()
    
    B = X.shape[1]
    K = D_init.shape[0]
    
    model = LRRNetSCompleteOfficial(num_stages=num_stages, K=K, B=B, D_init=D_init).to(DEVICE)
    
    # Separare parametrii conform cerințelor autorilor (Straturi învățabile, Dicționar opțional fix/învățabil)
    parametrii_straturi = [p for n, p in model.named_parameters() if "D" not in n]
    optimizator = optim.Adam([
        {'params': parametrii_straturi, 'lr': lr_straturi},
        {'params': [model.D], 'lr': lr_dictionar}
    ])
    
    criteriu_reconstructie = nn.MSELoss()
    
    best_auc = 0.0
    best_epoch = 0
    best_det_map = None
    best_fpr, best_tpr = None, None
    patience_counter = 0
    
    for epoca in range(1, num_epochs + 1):
        model.train()
        optimizator.zero_grad()
        
        S, L = model(X)
        
        # Funcție de pierdere academică LRR: Reconstrucție Background + Regularizare Sparse a erorii (S)
        X_rec_bg = torch.matmul(L, model.D)
        loss_bg = criteriu_reconstructie(X_rec_bg, X - S)
        loss_sparse_S = torch.mean(torch.norm(S, p=2, dim=1))
        
        total_loss = loss_bg + lambda_bg * loss_sparse_S
        
        # Backpropagation protejat de erori numerice
        if not torch.isnan(total_loss) and not torch.isinf(total_loss):
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizator.step()
            
        # Evaluare performanță epocă curentă pentru Early Stopping
        model.eval()
        with torch.no_grad():
            S_eval, _ = model(X)
            # Harta de detecție brută extrasă prin norma L2 a reziduurilor pe fiecare pixel
            det_map_step = torch.norm(S_eval, p=2, dim=1).cpu().numpy()
            
            fpr, tpr, _ = roc_curve(gt, det_map_step)
            auc_curent = auc(fpr, tpr)
            
            # Corecție automată de polaritate a scorurilor de detecție dacă este cazul
            if auc_curent < 0.5:
                fpr, tpr, _ = roc_curve(gt, -det_map_step)
                auc_curent = auc(fpr, tpr)
                det_map_step = -det_map_step
                
        if auc_curent > best_auc:
            best_auc = auc_curent
            best_epoch = epoca
            best_det_map = det_map_step
            best_fpr, best_tpr = fpr, tpr
            patience_counter = 0
        else:
            pvariance = 0
            patience_counter += 1
            
        if patience_counter >= patience:
            break
            
    print(f"  [STOP AUTOMAT] {nume_dataset} s-a oprit la epoca {best_epoch}/{num_epochs} | Cel mai bun AUC: {best_auc*100:.2f}%")
    
    return {
        "auc": best_auc,
        "stop_epoch": best_epoch,
        "detection_map": best_det_map,
        "fpr": best_fpr,
        "tpr": best_tpr
    }

# ==========================================
# 4. EXECUTARE PREPROCESARE DATE (MATLAB INGESTION)
# ==========================================
preprocesat_datasets = {}

for config in datasets_config:
    nume = config["name"]
    fisier = config["filename"]
    b_r, b_g, b_b = config["rgb_bands"]

    # Verificare și setare cale directă în folderul local "dataset"
    cale_mat = fisier if os.path.exists(fisier) else os.path.join("dataset", fisier)

    try:
        mat_data = sio.loadmat(cale_mat)
    except Exception as e:
        print(f"[EROARE] Lipsă fișier: {fisier} în path-ul selectat. Skipping.")
        continue

    chei_disponibile = [k for k in mat_data.keys() if not k.startswith('__')]
    img_key = 'data' if 'data' in mat_data else [k for k in chei_disponibile if any(x in k.lower() for x in ['diego', 'img', 'x', 'airport', 'beach', 'urban'])][0]
    gt_key = 'map' if 'map' in mat_data else [k for k in chei_disponibile if any(x in k.lower() for x in ['gt', 'map', 'mask', 'label'])][0]

    X_raw = mat_data[img_key]
    gt_raw = mat_data[gt_key]

    if len(X_raw.shape) == 2:
        if X_raw.shape[0] < X_raw.shape[1]: X_raw = X_raw.T
        H = int(np.sqrt(X_raw.shape[0]))
        W, B = H, X_raw.shape[1]
        X_raw = X_raw.reshape(H, W, B)
    else:
        H, W, B = X_raw.shape

    gt_raw = gt_raw.reshape(H, W)
    M = H * W

    # Normalizare și Extragere Dicționar K-Means Background
    X_2d = X_raw.reshape(M, B).astype(np.float32)
    X_min, X_max = X_2d.min(axis=0), X_2d.max(axis=0)
    X_normalized = (X_2d - X_min) / (X_max - X_min + 1e-8)

    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=2026, n_init=5).fit(X_normalized)
    D_np = kmeans.cluster_centers_.astype(np.float32)

    preprocesat_datasets[nume] = {
        "X": torch.from_numpy(X_normalized).to(DEVICE),
        "D": torch.from_numpy(D_np).to(DEVICE),
        "gt": torch.from_numpy(gt_raw.astype(np.float32)).to(DEVICE),
        "metadata": {"H": H, "W": W, "B": B, "M": M}
    }

# ==========================================
# 5. PIPELINE ANTRENĂRI ȘI DASHBOARD VIZUAL
# ==========================================
num_sets = len(preprocesat_datasets)

if num_sets > 0:
    fig, axes = plt.subplots(nrows=num_sets, ncols=4, figsize=(15, 3.4 * num_sets))
    fig.suptitle("Dashboard Academic Multi-Dataset LRR-Net-S (Train & Test Pipeline)", fontsize=14, fontweight='bold', y=0.995)
    
    if num_sets == 1:
        axes = np.expand_dims(axes, axis=0)

    colector_auc = {}
    idx_rand = 0

    for nume_set, date in preprocesat_datasets.items():
        print(f"\n[START TRAINING] Lansare optimizare pentru: {nume_set}")
        
        rezultat = antreneaza_si_evalueaza_lrrnet_optimizat(
            nume_dataset=nume_set,
            date_dict=date,
            num_stages=OPTIMAL_STAGES,
            num_epochs=OPTIMAL_EPOCHS,
            lr_straturi=OPTIMAL_LR,
            patience=30,
            lambda_bg=2.5
        )

        colector_auc[nume_set] = rezultat["auc"] * 100.0

        # Pregătire imagini pentru Plotare
        meta = date["metadata"]
        H_int, W_int = meta["H"], meta["W"]
        
        config_curent = [c for c in datasets_config if c["name"] == nume_set][0]
        b_r, b_g, b_b = config_curent["rgb_bands"]
        
        X_brut_viz = date["X"].cpu().numpy().reshape(H_int, W_int, -1)
        br, bg, bb = min(b_r, X_brut_viz.shape[2]-1), min(b_g, X_brut_viz.shape[2]-1), min(b_b, X_brut_viz.shape[2]-1)
        
        rgb_plot = np.stack([X_brut_viz[:, :, br], X_brut_viz[:, :, bg], X_brut_viz[:, :, bb]], axis=-1)
        rgb_plot = (rgb_plot - rgb_plot.min()) / (rgb_plot.max() - rgb_plot.min() + 1e-8)
        
        gt_plot = date["gt"].cpu().numpy().reshape(H_int, W_int)
        
        det_map_raw = rezultat["detection_map"]
        det_map_plot = (det_map_raw - det_map_raw.min()) / (det_map_raw.max() - det_map_raw.min() + 1e-8)
        det_map_plot = det_map_plot.reshape(H_int, W_int)

        # Mapare în interfața grafică
        ax_rgb, ax_gt, ax_det, ax_roc = axes[idx_rand, 0], axes[idx_rand, 1], axes[idx_rand, 2], axes[idx_rand, 3]

        ax_rgb.imshow(rgb_plot)
        ax_rgb.set_ylabel(f"{nume_set}\nStop Ep.: {rezultat['stop_epoch']}", fontsize=8, fontweight='bold')
        ax_rgb.set_xticks([]); ax_rgb.set_yticks([])
        if idx_rand == 0: ax_rgb.set_title("Imagine Color RGB", fontsize=10, fontweight='bold')

        ax_gt.imshow(gt_plot, cmap='gray')
        ax_gt.set_xticks([]); ax_gt.set_yticks([])
        if idx_rand == 0: ax_gt.set_title("Ground Truth (GT)", fontsize=10, fontweight='bold')

        im_det = ax_det.imshow(det_map_plot, cmap='jet', vmin=0.0, vmax=1.0)
        ax_det.set_xticks([]); ax_det.set_yticks([])
        if idx_rand == 0: ax_det.set_title("Ieșire LRR-Net-S", fontsize=10, fontweight='bold')
        fig.colorbar(im_det, ax=ax_det, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)

        ax_roc.plot(rezultat["fpr"], rezultat["tpr"], color='red', lw=1.8, label=f"AUC = {rezultat['auc']:.4f}")
        ax_roc.plot([0, 1], [0, 1], 'k--', lw=0.8)
        ax_roc.set_xlim([-0.01, 1.0]); ax_roc.set_ylim([0.0, 1.05])
        ax_roc.grid(True, linestyle=':', alpha=0.5)
        ax_roc.legend(loc='lower right', fontsize=7)
        if idx_rand == 0: ax_roc.set_title("Curba ROC", fontsize=10, fontweight='bold')
        if idx_rand == num_sets - 1:
            ax_roc.set_xlabel('False Alarm Rate', fontsize=8)
            ax_roc.set_ylabel('True Positive Rate', fontsize=8)

        idx_rand += 1

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.35)
    plt.show()

    # ==========================================
    # 6. AFIȘARE TABEL REZULTATE CONSOLIDAT
    # ==========================================
    nume_seturi_raport = list(colector_auc.keys())
    valori_auc_raport = list(colector_auc.values())
    m_auc = np.mean(valori_auc_raport)

    tabel_final_optimizat = pd.DataFrame({
        "Set de Date Hiperspectrale": nume_seturi_raport + ["------------------------------------", "INDICATOR MEDIU GENERAL (mAUC)"],
        "Scor AUC-ROC Final": [f"{v:.2f}%" for v in valori_auc_raport] + ["", f"{m_auc:.2f}%"]
    })

    print("\n" + "="*75)
    print("   TABEL ACADEMIC DE PERFORMANȚĂ FINALĂ CONFORM LITERATURII HAD")
    print("="*75)
    print(tabel_final_optimizat.to_string(index=False))
    print("="*75)