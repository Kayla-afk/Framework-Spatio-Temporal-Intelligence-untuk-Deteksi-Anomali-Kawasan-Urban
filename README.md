# Urban Anomaly Intelligence Framework

Framework Spatio-Temporal Intelligence untuk Deteksi Anomali Kawasan Urban Berbasis Data Mobilitas dan Aktivitas Transaksi Digital.

Studi Kasus: Metropolitan Surabaya

---

## Deskripsi

Framework ini mengimplementasikan pipeline analitik empat tahap untuk mengidentifikasi kawasan urban yang menunjukkan perilaku abnormal berdasarkan dua sumber data utama: GPS trajectory (mobilitas) dan pola transaksi digital (aktivitas ekonomi). Output akhirnya berupa peta anomali urban interaktif, identifikasi kawasan prioritas monitoring, dan laporan ringkasan risiko per zona.

Positioning metodologi yang perlu dipahami:

- Framework dirancang bersifat generalizable, bukan terikat satu kota.
- Dataset yang digunakan adalah public datasets (GeoLife GPS dan IEEE-CIS Fraud Detection).
- Surabaya digunakan sebagai konteks implementasi (bounding box studi kasus dan nama zona).
- Tujuan demonstrasi adalah menunjukkan bahwa framework ini dapat diimplementasikan dalam konteks metropolitan Indonesia dengan data publik yang tersedia secara bebas.

---

## Arsitektur Framework

```
Data Mobilitas (GeoLife)          Data Transaksi (IEEE-CIS)
        |                                    |
        v                                    v
[Stage 1] Urban Mobility Mapping    [Stage 3] Transaction Contextualization
   HDBSCAN Clustering                  Irregularity Scoring
   Hotspot Identification              Temporal Fluctuation Analysis
        |                                    |
        v                                    |
[Stage 2] Spatio-Temporal                    |
   Anomaly Detection                         |
   Isolation Forest                          |
        |                                    |
        +---------------+--------------------+
                        |
                        v
             [Stage 4] Urban Anomaly Intelligence
                Spatial Join + Composite Scoring
                Risk Classification
                Priority Area Identification
                        |
                        v
              Output: Urban Anomaly Map
                      Priority Monitoring Report
                      Static Figures
```

---

## Dataset

### Dataset Mobilitas: GeoLife GPS Trajectory Dataset

Sumber: Microsoft Research
URL: https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/

Cara mendapatkan dataset:
1. Buka URL di atas, scroll ke bagian download.
2. Download file GeoLife1.3.zip (sekitar 298 MB).
3. Ekstrak ke `data/raw/mobility/`.

Struktur folder yang diharapkan setelah ekstrak:

```
data/raw/mobility/
    000/
        Trajectory/
            20081023025304.plt
            20081024020959.plt
            ...
    001/
        Trajectory/
            ...
    ...
    182/
        Trajectory/
            ...
```

Dataset ini berisi 182 pengguna, lebih dari 17.000 trajectory, total 1,2 miliar GPS point dari Beijing (2007-2012). Framework akan remap koordinat ke bounding box Surabaya secara otomatis karena pola spasial relatif yang dipertahankan adalah yang relevan, bukan koordinat absolutnya.

Format file .plt:
- 6 baris header (dilewati otomatis)
- Baris data: `latitude,longitude,0,altitude,date_days,date,time`

### Dataset Transaksi: IEEE-CIS Fraud Detection Dataset (Primer)

Sumber: Kaggle
URL: https://www.kaggle.com/competitions/ieee-fraud-detection/data

Cara mendapatkan:
1. Login ke Kaggle, accept competition rules.
2. Download `train_transaction.csv` (sekitar 470 MB).
3. Letakkan di `data/raw/transactions/train_transaction.csv`.

Kolom yang digunakan: `TransactionID`, `TransactionDT`, `TransactionAmt`, `isFraud`

### Dataset Transaksi: Credit Card Fraud Dataset (Alternatif)

Sumber: Kaggle
URL: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Cara mendapatkan:
1. Download `creditcard.csv` (sekitar 143 MB).
2. Letakkan di `data/raw/transactions/creditcard.csv`.

Framework mendeteksi format dataset secara otomatis (IEEE-CIS atau Credit Card Fraud).

### Synthetic Data Mode

Jika kamu belum mendownload dataset di atas, framework dapat dijalankan dengan data sintetis menggunakan flag `--synthetic`. Mode ini menghasilkan GPS trajectory dan transaksi dengan karakteristik statistik realistis termasuk anomali yang disengaja untuk demonstrasi deteksi.

---

## Instalasi

Direkomendasikan menggunakan Python 3.10 atau lebih baru.

```bash
# Clone atau buat folder project
cd urban-anomaly-framework

# Install dependencies
pip install -r requirements.txt
```

Dependencies utama: `hdbscan`, `scikit-learn`, `pandas`, `numpy`, `folium`, `matplotlib`, `geopandas`, `shapely`, `scipy`, `plotly`.

---

## Cara Menjalankan

Mode synthetic (tidak perlu download dataset):

```bash
python main.py --synthetic
```

Mode dengan dataset asli (setelah download dan letakkan di folder yang benar):

```bash
python main.py
```

Opsi tambahan:

```bash
# Hanya jalankan analisis, skip visualisasi
python main.py --synthetic --skip-viz

# Gunakan config berbeda
python main.py --config config/config_custom.yaml

# Log lebih detail
python main.py --synthetic --log-level DEBUG
```

---

## Output

Setelah pipeline selesai, output tersimpan di:

```
outputs/
    maps/
        urban_anomaly_map.html       # Peta interaktif (buka di browser)
    figures/
        urban_anomaly_map.png        # Peta anomali utama (untuk laporan)
        congestion_hotspot.png       # Distribusi dan spasial kemacetan anomali
        temporal_pattern.png         # Pola mobilitas per jam + distribusi risiko
        anomaly_correlation.png      # Korelasi skor mobilitas vs transaksi
    reports/
        priority_areas_report.html   # Laporan kawasan prioritas monitoring
```

### Deskripsi Output Utama

Urban Anomaly Map (HTML interaktif):
Peta dengan empat layer yang bisa di-toggle: mobility heatmap dari GPS, anomaly zone grid dengan gradient warna berdasarkan skor, HDBSCAN hotspot markers, dan priority monitoring area markers. Setiap elemen memiliki popup informasi detail.

Urban Anomaly Map (PNG):
Scatter plot overlay di atas hexbin density GPS. Warna menunjukkan composite anomaly score (merah = tinggi). Bintang menandai top-5 priority areas.

Priority Areas Report (HTML):
Tabel kawasan prioritas monitoring dengan ranking, risk level, koordinat, dan breakdown skor per komponen.

---

## Struktur Project

```
urban-anomaly-framework/
    config/
        config.yaml              # Semua parameter pipeline
    data/
        raw/
            mobility/            # Letakkan GeoLife di sini
            transactions/        # Letakkan IEEE-CIS / creditcard.csv di sini
        processed/               # Intermediate files (auto-generated)
    src/
        ingestion/
            mobility_ingestion.py       # Load dan preprocess GeoLife
            transaction_ingestion.py    # Load dan preprocess dataset transaksi
            synthetic_generator.py      # Generator data sintetis
        clustering/
            hdbscan_clustering.py       # Stage 1: HDBSCAN spatial clustering
        anomaly/
            isolation_forest.py         # Stage 2: Isolation Forest detection
        transaction/
            contextualization.py        # Stage 3: Transaction analysis
        integration/
            urban_intelligence.py       # Stage 4: Integration dan scoring
        visualization/
            map_renderer.py             # HTML map dan static figures
    outputs/
        maps/
        figures/
        reports/
    main.py                      # Entry point pipeline
    requirements.txt
    config/config.yaml
```

---

## Metode

### Stage 1: Urban Mobility Mapping

Input: GPS trajectory data (latitude, longitude, timestamp, vehicle/user ID)

HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) digunakan untuk mendeteksi cluster pergerakan tanpa perlu menentukan jumlah cluster di awal. Keunggulan dibanding K-Means untuk data GPS urban: mampu menangani cluster dengan bentuk dan kepadatan berbeda, robust terhadap noise (titik GPS outlier diberi label -1), dan secara natural mengidentifikasi hotspot berkepadatan tinggi.

Parameter yang dikonfigurasi: `min_cluster_size`, `min_samples`, `cluster_selection_epsilon`. Konfigurasi default dioptimalkan untuk skala metropolitan (grid sekitar 300 meter).

Output: cluster label per titik, cluster summary (centroid, ukuran, density score), hotspot flags.

### Stage 2: Spatio-Temporal Anomaly Detection

Input: Output Stage 1 + fitur temporal dan traffic

Isolation Forest mendeteksi anomali berdasarkan kemudahan isolasi suatu titik dalam feature space multi-dimensional. Titik yang mudah diisolasi (berada di area sparse dari distribusi) dianggap anomali. Feature yang digunakan mencakup koordinat spasial ternormalisasi, encoding sirkular jam dan hari, kecepatan kendaraan, congestion index, kepadatan cluster lokal, dan point density per window spatio-temporal.

Parameter `contamination` (default 0.05) merepresentasikan estimasi proporsi anomali. Nilai ini dapat disesuaikan berdasarkan domain knowledge tentang frekuensi kejadian abnormal di kawasan yang dianalisis.

Output: anomaly prediction per titik, normalized anomaly score (0-1), agregasi ke level zona grid.

### Stage 3: Transaction Contextualization

Input: IEEE-CIS atau Credit Card Fraud dataset

Dataset transaksi tidak memiliki koordinat GPS, sehingga zona spasial disimulasikan dan di-assign ke transaksi berdasarkan distribusi probabilistik yang mencerminkan konsentrasi aktivitas ekonomi (lebih tinggi di pusat kota). Irregularity score per area dihitung dari kombinasi: volume anomali (z-score), fraud rate, volatilitas temporal, dan nilai transaksi ekstrem.

Output: contextual activity score per area (0-1), temporal transaction profile per jam.

### Stage 4: Urban Anomaly Intelligence

Input: Output Stage 2 (mobility zones) + Output Stage 3 (transaction indicators)

Spatial join menggunakan nearest-neighbor matching (KD-tree) untuk menghubungkan mobility zones dengan transaction areas. Composite anomaly score dihitung sebagai weighted sum dari tiga komponen dengan bobot yang dikonfigurasi di `config.yaml`:

- Mobility anomaly score dari Isolation Forest (default: 45%)
- Anomaly rate density di zona (default: 25%)
- Transaction contextual score (default: 30%)

Klasifikasi risiko: HIGH (>= 0.65), MEDIUM (0.35-0.65), LOW (< 0.35).

Priority monitoring areas diidentifikasi berdasarkan priority score yang memperhitungkan composite score dan bonus untuk zona dengan sinyal ganda (anomali di kedua dimensi mobilitas dan transaksi).

---

## Konfigurasi

Semua parameter dapat disesuaikan di `config/config.yaml` tanpa mengubah kode. Parameter penting:

- `study_area`: bounding box dan referensi koordinat studi kasus
- `hdbscan.min_cluster_size`: semakin kecil, semakin banyak cluster terdeteksi
- `isolation_forest.contamination`: estimasi proporsi anomali (0.0-0.5)
- `integration.weight_*`: bobot masing-masing komponen skor (harus berjumlah 1.0)
- `integration.high_risk_threshold`: ambang batas klasifikasi HIGH risk

---

## Catatan Metodologi

Dataset GeoLife berasal dari Beijing. Framework ini me-remap koordinat ke bounding box Surabaya untuk kontekstualisasi studi kasus yang menganalisis pola spasial relatif (dimana hotspot terbentuk, bagaimana distribusi pergerakan, anomali temporal) bukan koordinat absolutnya. Pola mobilitas urban memiliki karakteristik universal yang dapat dipelajari dari dataset manapun dan kemudian dicontextualize ke area studi.
Untuk produksi atau penelitian yang membutuhkan validasi empiris pada data Surabaya aktual, langkah selanjutnya adalah mengganti data GeoLife dengan data GPS lokal dari sumber seperti OpenStreetMap traffic data, data Dinas Perhubungan Surabaya, atau trace dari platform ride-hailing dengan izin akses yang sesuai.
