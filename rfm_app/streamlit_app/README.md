# RFM Segmentation & Donor Ladder Intelligence — Dashboard Streamlit

Dashboard interaktif untuk proyek **RFM Segmentation & Donor Ladder Intelligence**
(Mizan Amanah — Kelompok 1). Seluruh logika analitik pada aplikasi ini meniru 1:1
pipeline pada notebook final **`RFM_Mizan_Amanah_revisi.ipynb`**:

1. Load & cleaning data (dedup, hapus nominal ≤ 0, normalisasi akad, dst.)
2. Feature engineering RFM (Recency, Frequency, Monetary, dst.)
3. Deteksi Reactivated Donor (gap > 90 hari) + klasifikasi **9 segmen Tangga Donatur**
4. Rekomendasi strategi engagement per donor
5. Tren bulanan, distribusi program, pola donasi
6. Segment mapping table & ringkasan segmen
7. Dashboard distribusi donor (visualisasi interaktif)
8. Distribusi Recency, Frequency, Monetary
9. Export final (CSV & Excel multi-sheet)

## Cara Menjalankan

1. Pastikan Python 3.9+ sudah terpasang.
2. Install dependency:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan aplikasi dari folder ini:
   ```bash
   streamlit run app.py
   ```
4. Browser akan terbuka otomatis ke `http://localhost:8501`.

Dataset bawaan (`data/data_set_donasi_ma_2020_2025.csv`) sudah disertakan di folder
`data/`, sehingga aplikasi langsung bisa dijalankan tanpa upload manual. Jika ingin
mengganti dataset, gunakan tombol **"Ganti dataset (opsional)"** di sidebar — pastikan
struktur kolomnya sama: `tanggal, transaksi_id, donor_id, program, akad, nominal`.

## Struktur Folder

```
streamlit_app/
├── app.py                 # Aplikasi utama
├── requirements.txt
├── README.md
└── data/
    └── data_set_donasi_ma_2020_2025.csv
```

## Fitur Dashboard

- **Sidebar**: filter Segmen Donor & Akad Dominan, info snapshot date & ringkasan data
- **Tab Ringkasan & Segmentasi**: KPI, bar chart jumlah donor per segmen, pie chart
  kontribusi nominal, tabel ringkasan segmen
- **Tab Tren & Program**: tren bulanan (nominal vs transaksi), Top 10 program, pola
  donasi (Meningkat/Stabil/Menurun)
- **Tab Distribusi RFM**: histogram Recency, Frequency, Monetary
- **Tab Segment Mapping & Rekomendasi**: tabel kriteria 9 segmen + rekomendasi strategi
  per segmen
- **Tab Donor Explorer & Export**: tabel donor individual (filter + pencarian
  donor_id) serta download `rfm_output_FINAL.csv` dan `Kebutuhan_informasi_data_HASIL.xlsx`
  (5 sheet)

Catatan: filter di sidebar memengaruhi seluruh tab (KPI, chart, tabel) kecuali file
export final di tab terakhir, yang selalu mengekspor **seluruh** 20.469 donor agar
tetap menjadi single source of truth yang konsisten dengan deliverable resmi proyek.
