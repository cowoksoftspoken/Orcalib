# Orca Framework - Agent Instructions

Welcome to the Orca project! You are an AI agent assisting the user (the "pilot") in building a lightweight, modular, and fast Machine Learning framework from scratch. The core is written in **Rust** for performance and memory safety, while the frontend is in **Python** (via PyO3/Maturin) to mimic the PyTorch API.

## 🎯 Visi Proyek
**"Simple by default. Powerful when needed."**
- Orca harus mudah dipahami secara arsitektur (modular).
- Tidak mengorbankan performa (siap menggunakan GPU Backend).
- Menghindari *spaghetti code* dan *circular dependencies*.

## 🏗️ Struktur Repositori
- `orca-core/`: Mendefinisikan *traits* utama (`Backend`), `Shape`, `DType`, `Device`, dan *Error handling*.
- `orca-tensor/`: Representasi `Tensor<B: Backend>` dan implementasi *forward-pass operations*.
- `orca-backend-cpu/`: Implementasi referensi untuk *Backend* yang berjalan di CPU (Single-threaded).
- `orca-autograd/`: *Reverse-mode Automatic Differentiation Engine* yang membungkus *backend* lain (`Autodiff<B>`) menggunakan *Tape-based computation graph*.
- `orca-python/`: *Rust to Python bindings* menggunakan PyO3.
- `python/orca/`: *Python library frontend* (implementasi OOP untuk `nn.Module`, `optim.SGD`, `data.DataLoader`).
- `train_mnist.py`, `test_xor.py`: *Script* untuk memverifikasi fungsionalitas end-to-end (Python -> Rust -> Python).

## 🚦 Status Proyek (Completed Phases)
✅ **Phase 1: Foundation**
- Konfigurasi Cargo Workspace & Arsitektur dasar.
- Implementasi `Shape`, `Tensor`, dan CPU Backend (operasi dasar primitif `add`, `mul`, `matmul`).

✅ **Phase 2: Autograd Engine & PyO3**
- Implementasi *Autograd Tape* dan `BackwardOp`.
- *Python bindings* awal menggunakan Maturin.

✅ **Phase 3: Broadcasting & Advanced Ops**
- Implementasi *Broadcasting* via *explicit expand* & *sum_to_shape* untuk `BackwardOp`.
- Tambahan ops: `reshape`, `exp`, `log`, `transpose`.
- Sukses melatih model klasifikasi XOR sederhana (100% akurasi).

✅ **Phase 4: ML Primitives & MNIST Verification**
- Implementasi `nn.Linear`, `nn.ReLU`, `nn.Flatten`, `nn.CrossEntropyLoss`.
- Perbaikan *bug* mematikan pada **Gradient Accumulation** (gradien ditimpa/di-*overwrite* jika dipanggil >1 kali dalam *forward pass*). *Fixed via* `accumulate_grad()` di level Backend.
- Verifikasi *Training Loop* pada dataset scikit-learn Digits (versi kecil MNIST) berhasil menembus akurasi **~87%** dengan loss konvergen.

## 🚀 Next Objective
**Phase 5: GPU Acceleration (WGPU)**
Fokus selanjutnya adalah menciptakan `orca-backend-gpu` menggunakan *crate* `wgpu`.
- Memindahkan operasi numerik yang berat ke GPU *shaders*.
- Mempertahankan kompatibilitas API sehingga *Autograd* dan *Python frontend* tidak perlu diubah secara drastis (cukup *switch backend*).

## ⚠️ Rules & Coding Standards for Agents
1. **Pahami Dulu, Eksekusi Kemudian**: Jangan hanya menjadi 'Yes-Man'. Teliti apakah instruksi *user* secara teknis solid. Berikan spekulasi, data, dan alasan teknis sebelum menulis *code*.
2. **Error Handling**: JANGAN PERNAH gunakan `.unwrap()` atau `panic!` pada *library code* (`src/`), gunakan *proper Error propagation* (`Result`, `OrcaError`). `.unwrap()` hanya boleh di *script testing* atau *Autograd unwrap* jika tipe data sudah dipastikan benar 100%.
3. **No Circular Dependencies**: Jaga hirarki crate. `orca-core` tidak boleh bergantung pada yang lain. `orca-autograd` hanya bergantung pada `orca-tensor` dan `orca-core`.
4. **PyO3 Signatures**: Pastikan kompatibel dengan versi terbaru PyO3 0.21+ (cth: `m: &Bound<'_, PyModule>`).
5. **Workflow Build**: Gunakan `maturin develop` dari root *workspace* (ingat `.venv\Scripts\Activate.ps1`) untuk melakukan *build* ke Python environment. Selalu tes *code* dengan menjalankan ulang _script_ training.

Bacalah file ini dengan seksama setiap kali kamu mengambil alih sesi!
