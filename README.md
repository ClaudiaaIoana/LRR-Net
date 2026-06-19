# LRR-Net-S: Deep Unfolding Network pentru Detecția Anomaliilor Hiperspectrale

Acest repository conține implementarea independentă locală (Python standalone) a modelului **LRR-Net-S** (Low-Rank Representation Network with Sparsity), un algoritm de tip *deep unfolding* bazat pe optimizatorul ADMM (Alternating Direction Method of Multipliers) destinat detecției anomaliilor în imagini hiperspectrale (HAD).

---

##  Citare și Referință Academică

Această soluție a fost dezvoltată și implementată strict pe baza metodologiei și arhitecturii descrise în articolul științific de referință:

> **Li, C., Zhang, B., Hong, D., Yao, J., & Chanussot, J. (2023).** *LRR-Net: An Interpretable Deep Unfolding Network for Hyperspectral Anomaly Detection.* **IEEE Transactions on Geoscience and Remote Sensing**, vol. 61, Art no. 5513412, pp. 1-12. DOI: 10.1109/TGRS.2023.3279834.

---

##  Structura Proiectului

Directorul de lucru local este organizat după cum urmează:

```text
├── train_and_evaluate.py       # Scriptul complet pentru pipeline-ul de antrenare și testare
├── test_pretrained_models.py  # Scriptul independent pentru inferența pe modele Black-Box
├── LRR_Net_S_Official.ipynb    # Notebook-ul Jupyter cu implementarea completă pas cu pas
├── dataset/                    # Folderul local ce conține toate seturile de date brute (.mat)
│   ├── aviris_1.mat
│   ├── abu-airport-1.mat
│   └── ... (toate cele 11 seturi de date)
└── modele_lrrnet_blackbox/     # Folderul cu fișierele binare ale modelelor pre-antrenate (.pth)
    ├── lrrnet_aviris_airport_1_blackbox.pth
    └── ... (toate cele 11 modele salvate la AUC Maxim)
```

##  Cerințe și Rulare (Python Environment)

Scripturile sunt proiectate pentru a fi executate clasic într-un mediu virtual Python local (de exemplu, Anaconda/Miniconda sau venv).

### 1. Instalarea Dependențelor
Asigură-te că ai instalate pachetele de bază rulând:
```bash
pip install torch numpy pandas matplotlib scikit-learn scipy
```

### 2. Antrenarea și Evaluarea de la Zero
Pentru a încărca datele brute din folderul `dataset/`, a rula inițializarea K-Means și a parcurge întregul ciclu de optimizare (antrenare cu Early Stopping pe bază de AUC-ROC), execută:
```bash
python train_and_evaluate.py
```

### 3. Testarea Modelelor Pre-antrenate (Black-Box Inference)
Pentru a rula inferența rapidă de tip "Black-Box" (care garantează alinierea statistică 1:1, la nivel de zecimală, cu experimentul original eliminând instabilitatea stocastică locală), execută:
```bash
python test_pretrained_models.py
```

## 📊 Dashboard de Vizualizare

La finalul rulării scriptului, se va deschide o fereastră interactivă care conține o matrice grafică completă (organizată optim pe două coloane independente de tip 6 rânduri x 4 coloane pentru a preveni suprapunerea notațiilor statistice). Aceasta afișează pentru fiecare dintre cele 11 seturi de date:
1. **Harta de detecție a anomaliilor** în format bidimensional ($H \times W$), colorată folosind paleta `Jet` (unde nuanțele de roșu indică o probabilitate ridicată de prezență a anomaliilor) și scalată rigid între 0 și 1.
2. **Curba ROC de verificare** (redată cu linia roșie oficială din articol) suprapusă peste pragul teoretic de clasificare aleatorie (linia neagră sacadată).

### Raportul Statistic din Consolă
**IMPORTANT:** Imediat după ce închizi manual fereastra dashboard-ului grafic `matplotlib`, scriptul Python va genera automat în consolă/terminal un **Tabel Academic Consolidat**. Acesta prezintă o analiză comparativă directă între:
* **Scorurile AUC scrise/salvate original în mediul Google Colab** în timpul antrenării.
* **Scorurile AUC recalculate și verificate independent local**, oferind o validare matematică 1:1 (la nivel de zecimală) a replicabilității algoritmului.
* **Indicatorul statistic general $mAUC$ (media aritmetică)** calculat automat la baza tabelului pentru toate cele 11 scene hiperspectrale evaluate.

##  Notebook-ul de Lucru (`.ipynb`)

În cadrul acestui proiect, implementarea inițială, analiza detaliată și explorarea hiperparametrilor au fost realizate în notebook-ul Jupyter dedicat. Acesta reprezintă mediul ideal de experimentare și oferă:

* **Parcurgere Pas cu Pas:** O descompunere clară și explicată a fiecărui bloc structural al rețelei conform algoritmului din articol (S-block, J-block, W-block, L-block și P-block).
* **Rezultate Intermediare:** Afișarea în timp real a hărților de detecție și a evoluției curbelor ROC pe parcursul epocilor de antrenare, permițând observarea momentului exact în care intervine mecanismul de Early Stopping.
* **Flexibilitate în Optimizare:** Posibilitatea de a modifica rapid parametrii critici ai rețelei, precum numărul de etape de update (*stages*), numărul maxim de epoci, ratele de învățare separate pentru straturi și dicționar, precum și ponderea parametrului de regularizare $\lambda_{bg}$.
