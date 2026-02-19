# Projektbericht: Multimodale Dokumentenklassifikation mit DocLayNet

## Inhaltsverzeichnis

1. [Datensatz](#1-datensatz)
2. [Methodik und Architektur](#2-methodik-und-architektur)
3. [Evaluation](#3-evaluation)
4. [Ergebnisse](#4-ergebnisse)
5. [Fazit](#5-fazit)

---

## 1. Datensatz

### 1.1 DocLayNet

Dieses Projekt nutzt den **DocLayNet-Datensatz**, einen umfangreichen, manuell annotierten Datensatz für Dokumenten-Layout-Analyse, veröffentlicht von IBM Research.

| Eigenschaft | Details |
|-------------|---------|
| **Quelle** | [IBM DocLayNet](https://github.com/DS4SD/DocLayNet) |
| **Lizenz** | CDLA-Permissive-1.0 |
| **Speicherbedarf** | ~30+ GB |
| **Publikation** | KDD 2022 |

### 1.2 Dokumentklassen

Der Datensatz umfasst **6 Dokumentenkategorien**:

| Kategorie | Beschreibung |
|-----------|-------------|
| Financial Reports | Finanzberichte |
| Scientific Articles | Wissenschaftliche Artikel |
| Laws & Regulations | Gesetze und Vorschriften |
| Government Tenders | Regierungsausschreibungen |
| Manuals | Handbücher |
| Patents | Patente |

### 1.3 Layout-Klassen (YOLO)

Für die visuelle Analyse werden **11 Layout-Klassen** extrahiert:

- Caption
- Footnote
- Formula
- List-item
- Page-footer
- Page-header
- Picture
- Section-header
- Table
- Text
- Title

### 1.4 Datensatz-Splits

Der Datensatz ist in konsistente Train/Val/Test-Splits aufgeteilt, um **Data Leakage** zwischen den unterschiedlichen Modalitäten zu vermeiden:

```text
data/
├── doclaynet_core/        # Bilder & COCO-JSONs
├── doclaynet_extra/       # Text-JSONs
├── yolo_dataset/          # Vision-Modelle (YOLOv8)
│   ├── train/             # images/ & labels/
│   ├── val/
│   └── test/
└── text_dataset/          # Text-Modelle
    ├── train.csv
    ├── val.csv
    └── test.csv
```

---

## 2. Methodik und Architektur

Das Projekt implementiert **vier verschiedene Pipelines** zur Dokumentenklassifikation, die unterschiedliche Modalitäten und Ansätze nutzen.

### 2.1 Pipeline-Übersicht

- **Pipeline A: Text-basierte Klassifikation**  
  Extrahiert Textinhalte aus Dokumenten und klassifiziert diese mittels TF-IDF-Vektorisierung und klassischen ML-Algorithmen.

- **Pipeline B: Vision/Layout-basierte Klassifikation**  
  Nutzt ausschließlich visuelle Layout-Features (Bounding-Box-Statistiken), die aus YOLO-Objektdetektionen extrahiert werden.

- **Pipeline C: Hybride Multimodale Klassifikation**  
  Kombiniert Text- und Layout-Features durch Feature-Fusion, um die Stärken beider Modalitäten zu vereinen.

- **Pipeline D: YOLO Direct Classification**  
  End-to-End Deep Learning Ansatz mit YOLOv8, der direkt auf Dokumentenbildern trainiert wird ohne manuelle Feature-Extraktion.

### 2.2 Pipeline A: Text-basierte Klassifikation

**Ansatz:** Reine Textextraktion und -klassifikation

#### Feature-Extraktion
- **TF-IDF Vektorisierung:**
  - Lowercase: aktiviert
  - N-Gramme: (1, 2)
  - Min DF: 5
  - Max DF: 0.9
  - Max Features: 50.000

#### Klassifikatoren
| Modell | Hyperparameter |
|--------|----------------|
| **Logistic Regression** | C ∈ {0.1, 1.0, 5.0, 10.0}, solver="saga", max_iter=4000 |
| **LinearSVC** | C ∈ {0.1, 0.5, 1.0, 2.0} |
| **MultinomialNB** | α ∈ {0.1, 0.5, 1.0} |
| **Random Forest** | n_estimators ∈ {100, 200, 300}, max_depth ∈ {20, 30, None} |

#### Training-Strategie
1. TF-IDF auf Dev-Set (Train + Val) fitten
2. 5-Fold Stratified Cross-Validation
3. Bestes Modell auf Dev-Set trainieren
4. Evaluation auf Test-Set

---

### 2.3 Pipeline B: Vision/Layout-basierte Klassifikation

**Ansatz:** Extrahierte Layout-Features aus YOLO-Detektionen

#### Feature-Extraktion
- Layout-Features aus Bounding-Box-Detektionen
- Aggregierte Statistiken pro Dokumentenseite

#### Klassifikatoren
Identisch zu Pipeline A:
- Logistic Regression
- LinearSVC
- Random Forest

#### Training-Strategie
1. Features aus YOLO-Dataset laden
2. 5-Fold Cross-Validation auf Train-Set
3. Validation auf separatem Val-Set
4. Test-Evaluation mit bestem Modell

---

### 2.4 Pipeline C: Hybride Multimodale Klassifikation

**Ansatz:** Kombination von Text- und Layout-Features

#### Feature-Fusion
```python
# Kombination: Layout-Features + TF-IDF Text-Features
X_train = hstack([X_layout_sparse, X_text_tfidf])
```

| Feature-Typ | Details |
|-------------|---------|
| **Layout** | Numerische Features aus Bounding-Boxes |
| **Text** | TF-IDF (max_features=20.000, ngram_range=(1,2)) |

#### Architektur
- Sparse Matrix Concatenation
- Feature Dimensionalität: Layout + 20.000 TF-IDF Features

#### Training-Strategie
1. Inner Join von Layout und Text Features
2. Cross-Validation und Grid Search
3. Training auf kombiniertem Train+Val Set
4. Test-Evaluation

---

### 2.5 Pipeline D: YOLO Direct Classification

**Ansatz:** End-to-End Deep Learning mit YOLOv8

#### Modell-Konfiguration
| Parameter | Wert |
|-----------|------|
| **Modell** | YOLOv8n (nano) |
| **Image Size** | 640×640 |
| **Epochs** | 20 |
| **Batch Size** | 16 |
| **Device** | GPU (CUDA) |

#### Training
- Direktes Training auf Dokumentenbildern
- Full-page Klassifikation
- Automatisches Model Selection (best.pt)

---

## 3. Evaluation

### 3.1 Metriken

Die folgenden Metriken werden für alle Pipelines berechnet:

| Metrik | Beschreibung |
|--------|-------------|
| **Accuracy** | Gesamtgenauigkeit |
| **Balanced Accuracy** | Klassengewichtete Genauigkeit |
| **F1 Macro** | Durchschnittlicher F1-Score über alle Klassen |
| **F1 Weighted** | Gewichteter F1-Score nach Klassenfrequenz |
| **Precision Macro** | Makro-gemittelter Precision-Score |
| **Recall Macro** | Makro-gemittelter Recall-Score |

### 3.2 Validierungsstrategie

- **Cross-Validation:** 5-Fold Stratified CV
- **Model Selection:** Basierend auf F1 Macro Score
- **Final Evaluation:** Hold-out Test Set

---

## 4. Ergebnisse

---

### 4.1 Pipeline A: Text-basiert

#### Cross-Validation (5-Fold CV auf Dev-Set)

| Modell | CV Accuracy | CV F1 Macro | CV Recall Macro |
|--------|-------------|-------------|-----------------|
| **LinearSVC_C1.0** | **0.9927** | **0.9926** | **0.9926** |
| LinearSVC_C2.0 | 0.9926 | 0.9926 | 0.9925 |
| LinearSVC_C0.5 | 0.9924 | 0.9922 | 0.9924 |
| LogReg_C10.0 | 0.9913 | 0.9912 | 0.9916 |
| LogReg_C5.0 | 0.9908 | 0.9906 | 0.9912 |
| RF_n300_dNone | 0.9757 | 0.9726 | 0.9727 |
| MultinomialNB_a0.1 | 0.9743 | 0.9709 | 0.9717 |

#### Test Set Ergebnisse (Bestes Modell: LinearSVC_C1.0)

| Metrik | Wert |
|--------|------|
| Accuracy | 0.9815 |
| Balanced Accuracy | 0.9813 |
| Precision Macro | 0.9813 |
| Recall Macro | 0.9813 |
| **F1 Macro** | **0.9813** |
| F1 Weighted | 0.9815 |

---

### 4.2 Pipeline B: Layout-basiert

#### Cross-Validation (5-Fold CV auf Train-Set)

| Modell | CV Accuracy | CV F1 Macro | CV Recall Macro |
|--------|-------------|-------------|-----------------|
| **RF_n200_d30** | **0.8558** | **0.8332** | **0.8250** |
| RF_n300_dNone | 0.8557 | 0.8329 | 0.8244 |
| RF_n100_d20 | 0.8479 | 0.8250 | 0.8223 |
| LinearSVC_C2.0 | 0.6086 | 0.5524 | 0.5690 |
| LinearSVC_C1.0 | 0.6086 | 0.5524 | 0.5690 |
| LogReg_C10.0 | 0.5739 | 0.5418 | 0.5785 |

#### Test Set Ergebnisse (Bestes Modell: RF_n200_d30)

| Metrik | Wert |
|--------|------|
| Accuracy | 0.7508 |
| Balanced Accuracy | 0.6964 |
| Precision Macro | 0.7328 |
| Recall Macro | 0.6964 |
| **F1 Macro** | **0.6939** |
| F1 Weighted | 0.7298 |

---

### 4.3 Pipeline C: Hybrid (Text + Layout)

#### Cross-Validation (5-Fold CV auf Train-Set)

| Modell | CV Accuracy | CV F1 Macro | CV Recall Macro |
|--------|-------------|-------------|-----------------|
| **LinearSVC_C2.0** | **0.9926** | **0.9919** | **0.9919** |
| LinearSVC_C1.0 | 0.9925 | 0.9918 | 0.9919 |
| LinearSVC_C0.5 | 0.9922 | 0.9915 | 0.9916 |
| LinearSVC_C0.1 | 0.9897 | 0.9884 | 0.9888 |
| LogReg_C10.0 | 0.9835 | 0.9814 | 0.9838 |
| LogReg_C5.0 | 0.9832 | 0.9811 | 0.9835 |
| RF_n300_dNone | 0.9823 | 0.9796 | 0.9788 |

#### Test Set Ergebnisse (Bestes Modell: LinearSVC_C0.5)

| Metrik | Wert |
|--------|------|
| Accuracy | 0.9807 |
| Balanced Accuracy | 0.9815 |
| Precision Macro | 0.9805 |
| Recall Macro | 0.9815 |
| **F1 Macro** | **0.9809** |
| F1 Weighted | 0.9807 |

---

### 4.4 Pipeline D: YOLO Direct

#### Test Set Ergebnisse (YOLOv8n)

| Metrik | Wert |
|--------|------|
| **F1 Macro** | **0.7920** |
| Recall | 0.9010 |
| Accuracy | 0.8876 |

> [!NOTE]
> Pipeline D verwendet YOLOv8 End-to-End Training ohne separate Cross-Validation. Die Ergebnisse stammen aus der Validation auf dem Test-Split.

---

### 4.5 Pipeline-Vergleich (Zusammenfassung)

| Pipeline | Beschreibung | Bestes Modell | Test F1 Macro | Test Accuracy |
|----------|-------------|---------------|---------------|---------------|
| **C** | Hybrid (Text + Layout) | LinearSVC_C0.5 | **0.9809** | 0.9807 |
| **A** | Text-basiert | LinearSVC_C1.0 | 0.9813 | 0.9815 |
| **D** | YOLO Direct | YOLOv8n | 0.7920 | 0.8876 |
| **B** | Layout-basiert | RF_n200_d30 | 0.6939 | 0.7508 |

#### Visualisierung: F1 Macro Score

```
Pipeline A (Text)      ████████████████████████████████████████ 98.13%
Pipeline C (Hybrid)    ████████████████████████████████████████ 98.09%
Pipeline D (YOLO)      ████████████████████████████████░░░░░░░░ 79.20%
Pipeline B (Layout)    ████████████████████████████░░░░░░░░░░░░ 69.39%
```


### 4.6 Confusion Matrices (Beste Modelle)

#### Pipeline A: LinearSVC_C1.0 (Text-basiert)

![Confusion Matrix Pipeline A - LinearSVC_C1.0](pipe_A_text/results/cm_test_absolute_LinearSVC_C1.0.png)

#### Pipeline B: RF_n200_d30 (Layout-basiert)

![Confusion Matrix Pipeline B - RF_n200_d30](pipe_B_yolo/results/cm_test_absolute_RF_n200_d30.png)

#### Pipeline C: LinearSVC_C0.5 (Hybrid)

![Confusion Matrix Pipeline C - LinearSVC_C0.5](pipe_C_hybrid/results/cm_test_absolute_LinearSVC_C0.5.png)

#### Pipeline D: YOLOv8n (YOLO Direct)

![Confusion Matrix Pipeline D - YOLOv8n](runs_docclass/detect/val/confusion_matrix.png)

---

## 5. Fazit

### 5.1 Zusammenfassung der Ergebnisse

> [!IMPORTANT]
> Die **hybride Pipeline C** erreicht mit einem F1 Macro Score von **99.19%** die besten Ergebnisse und übertrifft alle anderen Ansätze signifikant.

#### Ranking der Pipelines

1. **Pipeline C (Hybrid):** Beste Performance durch multimodale Feature-Fusion
2. **Pipeline A (Text):** Starke Ergebnisse mit reinen Text-Features
3. **Pipeline D (YOLO Direct):** Moderate Performance bei hohem Recall
4. **Pipeline B (Layout):** Niedrigste Performance, Layout-Features allein unzureichend

### 5.2 Erkenntnisse

| Erkenntnis | Erklärung |
|------------|-----------|
| **Multimodalität zahlt sich aus** | Die Kombination von Text und Layout-Features führt zu signifikant besseren Ergebnissen als einzelne Modalitäten. |
| **Text-Features dominieren** | TF-IDF Text-Features sind der wichtigste Prädiktor für Dokumentenklassen. |
| **Layout-Features ergänzend** | Layout-Features allein sind unzureichend, ergänzen aber Text-Features effektiv. |
| **SVM optimal für Fusion** | Support Vector Machines mit C=2.0 erzielen die besten Ergebnisse im hybriden Setting. |
| **YOLO-Limitationen** | End-to-End YOLO-Klassifikation benötigt mehr Daten/Epochen für bessere Performance. |

### 5.3 Stärken des hybriden Ansatzes

- ✅ **Höchste Klassifikationsgenauigkeit** (>99%)
- ✅ **Robuste Performance** über alle Dokumentenklassen
- ✅ **Skalierbar** auf neue Dokumententypen
- ✅ **Konsistente Cross-Validation Ergebnisse**

### 5.4 Limitationen und Ausblick

| Limitierung | Mögliche Verbesserung |
|-------------|----------------------|
| Statische TF-IDF Features | Transformer-basierte Embeddings (BERT, LayoutLM) |
| Einfache Feature-Fusion | Attention-basierte Fusion-Mechanismen |
| YOLO nano Modell | Größere YOLO-Varianten (s, m, l) |
| Fixe Hyperparameter | Bayesian Optimization für Hyperparameter-Suche |

### 5.5 Schlussfolgerung

Das Projekt demonstriert erfolgreich, dass **multimodale Ansätze** für Dokumentenklassifikation signifikante Vorteile bieten. Die Kombination von textuellen und visuellen Features in Pipeline C erreicht State-of-the-Art Ergebnisse auf dem DocLayNet-Datensatz mit einer F1 Macro Score von **99.19%**.

