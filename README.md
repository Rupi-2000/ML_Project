# DocLayNet Data Pipeline 📄➡️📊

Dieses Repository stellt eine **reproduzierbare Datenpipeline** zur Vorbereitung des [DocLayNet-Datensatzes](https://github.com/DS4SD/DocLayNet) für **multimodales Training** bereit. Es verarbeitet sowohl

* **visuelle Layout-Daten** (für YOLOv8) als auch
* **reine Textdaten** (für NLP / Dokumentklassifikation)

und stellt sicher, dass **identische Train/Val/Test-Splits** in beiden Modalitäten verwendet werden (Vermeidung von Data Leakage).

---

## 🚀 Features

* **Automatischer Download**
  Lädt die *Core*- und *Extra*-Datensätze automatisiert herunter und entpackt sie.

* **YOLOv8-kompatibles Format**
  Konvertiert COCO-Annotationen (Bounding Boxes) in das YOLO-Format.

* **Parallele Textextraktion (Multiprocessing)**
  Extrahiert Textinhalte aus den JSON-Dateien des Extra-Datensatzes und speichert sie als CSV. Nutzt alle verfügbaren CPU-Kerne (I/O- und CPU-optimiert).

* **Konsistente Datensplits**
  Garantiert, dass Dokumente im Vision-Training exakt denselben Splits (Train/Val/Test) im Text-Datensatz zugeordnet sind.

---

## 📂 Ordnerstruktur (Output)

Nach erfolgreicher Ausführung aller Skripte ergibt sich folgende Struktur:

```text
data/
├── doclaynet_core/        # Original-Download (Bilder & COCO-JSONs)
├── doclaynet_extra/       # Original-Download (Text-JSONs)
├── yolo_dataset/          # Output für Vision-Modelle (YOLOv8)
│   ├── train/             # images/ & labels/
│   ├── val/
│   └── test/
└── text_dataset/          # Output für Text-Modelle
    ├── train.csv
    ├── val.csv
    └── test.csv
```

---


## 🔧 Installation

```bash
# Repository klonen
git clone https://github.com/Rupi-2000/ML_Project.git
cd ML_Project
```
---

## 🧪 Environment Setup (Recommended)

Empfohlen wird die Verwendung eines Python Virtual Environments zur
Gewährleistung der Reproduzierbarkeit.

**Python-Version:** ≥ 3.12 (getestet mit Python 3.12.12)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip

# Pipeline A (Text)
pip install -e .[core]

# Pipeline B + C (Text + Vision)
pip install -e .[core,vision]

# Alles inkl. Entwicklung
pip install -e .[core,vision,dev]
```
The required dependencies are defined in pyproject.toml.

---

## Wichtiger Hinweis (Torch & CUDA)

Absichtlich nicht in TOML enthalten:

```bash
# CPU
pip install torch torchvision torchaudio

# CUDA (z. B. 12.1)
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121
```

➡️ Das ist Best Practice, besonders bei Python 3.12.

---

## ⚙️ Nutzung (Schritt-für-Schritt)

Die Skripte **müssen in der angegebenen Reihenfolge** ausgeführt werden.

---

### 1️⃣ Daten herunterladen

Lädt `DocLayNet_core.zip` und `DocLayNet_extra.zip` herunter und entpackt beide Datensätze.

⚠️ **Hinweis:** Es werden ca. **30 GB+ Speicherplatz** benötigt.

```bash
python prepare_data.py
```

---

### 2️⃣ Vision-Daten vorbereiten (YOLO)

Konvertiert die originalen COCO-Annotationen (`train.json`, `val.json`, `test.json`) in das YOLO-Format und kopiert die zugehörigen Bilder in die entsprechenden Ordner.

```bash
python prepare_core_yolo.py
```

**Output:**

```
data/yolo_dataset/
```
Erstelle eine (`data.yaml`) Datei und füge den Output von Terminal ein!

---

### 3️⃣ Text-Daten vorbereiten (CSV)

Verarbeitet die Text-JSON-Dateien aus dem *Extra*-Datensatz. Die Dokumente werden anhand der Dateinamen den Core-Splits (Train/Val/Test) zugeordnet.

**Optimierung:**
Verwendet `ProcessPoolExecutor`, um tausende Dateien parallel zu verarbeiten.

```bash
python prepare_extra_text.py
```

**Output:**

```
data/text_dataset/train.csv
data/text_dataset/val.csv
data/text_dataset/test.csv
```

---

### 4️⃣ Klassenverteilung prüfen (optional)

Gibt Statistiken zur Klassenverteilung der erzeugten CSV-Dateien aus, um mögliche Unwuchten frühzeitig zu erkennen.

```bash
python check_class_distro_text_df.py
```

---

## 📊 Dokumentklassen

Die Pipeline verarbeitet und filtert **6 Dokumentenkategorien**:

* Financial Reports
* Scientific Articles
* Laws & Regulations
* Government Tenders
* Manuals
* Patents

---

## 🧩 Layout-Klassen (YOLO)

Für den Vision-Teil werden **11 Layout-Klassen** extrahiert:

* Caption
* Footnote
* Formula
* List-item
* Page-footer
* Page-header
* Picture
* Section-header
* Table
* Text
* Title

---

## 📝 Lizenz & Referenz

Der DocLayNet-Datensatz wurde von **IBM Research** veröffentlicht. Bitte beachte die Lizenzbedingungen des Originaldatensatzes:

* **Lizenz:** CDLA-Permissive-1.0

**Paper:**
*DocLayNet: A Large Human-Annotated Dataset for Document-Layout Analysis* (KDD 2022)
