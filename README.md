# Projek UTS TKI — Temu Kembali Informasi

Deskripsi
- Proyek tugas akhir/perkuliahan untuk mata kuliah Temu Kembali Informasi.
- Berisi modul preprocessing, indexing, retrieval, dan evaluasi untuk dataset opini.

Persyaratan
- Python 3.8+ (direkomendasikan 3.10+)
- Virtual environment (disarankan)

Instalasi
1. Buat dan aktifkan virtualenv (Windows PowerShell):

```powershell
python -m venv .venv
& ".venv\Scripts\Activate.ps1"
```

2. Pasang dependensi:

```bash
pip install -r requirements.txt
```

Menjalankan
- Jalankan script helper (Windows):

```powershell
./run_app.bat
```

- Atau jalankan langsung (setelah mengaktifkan venv):

```bash
python app.py
```

Struktur proyek (ringkasan)
- `app.py` — entry point / runner (jalankan aplikasi atau skrip utama)
- `requirements.txt` — daftar dependensi Python
- `run_app.bat` — skrip batch untuk Windows
- `dataset_opini.csv` — dataset opini yang dipakai
- `modules/` — kode modular:
  - `preprocessing.py` — preprocessing teks
  - `indexing.py` — pembuatan indeks
  - `retrieval.py` — pengambilan dokumen/pencarian
  - `evaluation.py` — metrik dan evaluasi

Catatan
- Sesuaikan interpreter Python jika tidak menggunakan virtualenv.
- Jika aplikasi memiliki server web atau antarmuka, jalankan `app.py` atau lihat dokumentasi internal pada kode.

Jika mau, saya bisa menambahkan instruksi lebih rinci, contoh pemakaian fungsi, atau file README bahasa Inggris.
