# DialogSum AI — Streamlit

Aplikasi demonstrasi skripsi **Peringkasan Dialog Abstraktif Menggunakan T5-small dan T5-base**.

## Berkas

- `app.py` — aplikasi Streamlit dengan mode Dataset Uji dan Input Manual, filter topik, pilihan model/skenario, hasil ringkasan, statistik panjang, dan ROUGE tiga referensi.
- `model_manifest.json` — 20 skenario; **baru S05 dan S20** mempunyai `hf_repo_id` berdasarkan pola unggah notebook Kaggle.
- `data/hasil_validasi_20_skenario.csv` — metrik validasi asli dari notebook evaluasi.
- `data/hasil_uji_4_model.csv` — metrik uji akhir yang dilaporkan.
- `.streamlit/config.toml` — tema.
- `requirements.txt` — dependensi.

## Jalankan

1. Install Python 3.10+ lalu masuk ke folder proyek.
2. `python -m pip install -r requirements.txt`
3. Untuk model **private**, simpan token **read** Hugging Face di `.streamlit/secrets.toml` secara lokal (jangan diunggah):

   ```toml
   HF_TOKEN = "TOKEN_HF_BERIZIN_READ"
   ```

4. `streamlit run app.py`
5. Pilih `Dataset Uji`, topik, dialog, model, lalu klik **Ringkas**. Pada mode `Input Manual`, ROUGE per dialog tidak ditampilkan tanpa referensi.

## Sebelum deploy ke Streamlit Community Cloud

1. Periksa **dua repo ID** di `model_manifest.json` apakah identik dengan repo yang sudah diunggah. ID berdasarkan kode *upload* notebook:
   - S05: `zalfazaah16/t5-small-s05-lr5e-05-bs4`
   - S20: `zalfazaah16/t5-base-s20-lr5e-05-bs8`
2. Jika nama repo berbeda, **ubah hanya `hf_repo_id`** pada skenario terkait.
3. Upload folder proyek ini (tanpa secret) ke repositori GitHub.
4. Buka https://share.streamlit.io/, buat app baru, arahkan ke `app.py` di branch GitHub.
5. Di **Advanced settings → Secrets**, isi `HF_TOKEN = "..."` dengan token Hugging Face yang mempunyai izin baca model private.
6. Klik **Deploy**. Jika server mengalami kehabisan RAM saat memuat T5-base, gunakan hosting dengan sumber daya lebih besar atau pisahkan backend inferensinya.

## Menambah 18 checkpoint

Setelah checkpoint lainnya berhasil diunggah dari Kaggle, isi `hf_repo_id` yang masih `null` pada `model_manifest.json`. Kode tidak menganggap skenario tersebut dapat menjalankan inferensi sebelum lokasi model ditambahkan.

## Kesetaraan dengan notebook penelitian

- Data uji dibaca dari repositori resmi DialogSum pada saat runtime. Bila ingin tanpa unduhan internet, taruh file asli `dialogsum.test.jsonl` di `data/`.
- Model hanya menerima `summarize: ` + dialog, tanpa topik atau ringkasan referensi.
- Data uji mempertahankan teks dialog asli, sama seperti fungsi inferensi pada notebook evaluasi final.
- Tokenisasi input dibatasi 512 token, panjang keluaran `max_length=128`, `num_beams=1`, `do_sample=False`.
- ROUGE memakai `rouge_score.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)` dan rata-rata Precision, Recall, F1 secara terpisah terhadap tiga referensi. Skor aplikasi adalah **per dialog**, bukan skor rata-rata 500 data uji.
- S05 dan S20 adalah skenario terbaik dari validasi masing-masing model. Ke-18 skenario lain tetap ditampilkan dalam hasil validasi tetapi tidak dapat dipilih untuk inferensi sebelum bobotnya tersedia di Hugging Face.

## Batasan

- Kode sudah diperiksa sintaksnya secara lokal, tetapi **belum diuji langsung terhadap repositori private dan Streamlit Community Cloud** karena tidak ada akses token dan runtime cloud pengguna.
- T5-base bisa membutuhkan RAM yang melebihi batas hosting gratis. Aplikasi memuat satu model pada satu waktu dan memakai cache terbatas, tetapi kebutuhan RAM bergantung lingkungan.
- Jangan mengunggah `.streamlit/secrets.toml` ke GitHub. `.gitignore` sudah mengecualikannya.
