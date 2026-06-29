#!/usr/bin/env bash
set -e

# ============================================================
# Setup & Jalankan RFM Donor Ladder Streamlit App di Codespace
# ============================================================

# 1) Pastikan file zip sudah ada di workspace Codespace.
#    Cara upload: drag & drop file RFM_Donor_Ladder_Streamlit_App.zip
#    ke panel Explorer di VS Code (kiri), lalu jalankan script ini
#    dari terminal pada folder yang sama.

ZIP_NAME="RFM_Donor_Ladder_Streamlit_App.zip"
APP_DIR="rfm_app"

if [ ! -f "$ZIP_NAME" ]; then
  echo "Error: file $ZIP_NAME tidak ditemukan di folder ini ($(pwd))."
  echo "Upload dulu filenya ke Codespace, lalu jalankan ulang script ini."
  exit 1
fi

# 2) Unzip ke folder rfm_app
echo "Mengekstrak $ZIP_NAME ..."
unzip -o "$ZIP_NAME" -d "$APP_DIR"

# 3) Masuk ke folder aplikasi (hasil ekstrak: rfm_app/streamlit_app)
cd "$APP_DIR/streamlit_app"
echo "Masuk ke $(pwd)"

# 4) Buat virtual environment (sekali saja; aman dijalankan ulang)
if [ ! -d "venv" ]; then
  echo "Membuat virtual environment..."
  python3 -m venv venv
fi
source venv/bin/activate

# 5) Install dependency
echo "Install dependency..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# 6) Jalankan Streamlit
#    --server.address 0.0.0.0  -> wajib agar port forwarding Codespace berfungsi
#    --server.port 8501        -> port default Streamlit
echo "Menjalankan Streamlit di port 8501..."
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
