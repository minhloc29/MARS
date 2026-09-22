# HƯỚNG DẪN CHẠY BASELINE LKH-3 & HGS (PyVRP)

Tài liệu này hướng dẫn chi tiết cách cài đặt môi trường, kiểm thử và chạy benchmark 2 thuật toán cổ điển **HGS** và **LKH-3** trên bài toán CVRP (sizes 50, 100, 200, 500, 1000 × phân phối Uniform / Gaussian).

---

## 1. Yêu cầu môi trường & Cài đặt thư viện

Khuyến nghị dùng **Python 3.10 trở lên** (Python 3.10, 3.11, 3.12, 3.13 đều được).

### Bước 1: Cài đặt các thư viện cần thiết

Mở Terminal / PowerShell và chạy:

```bash
# Cài đặt solver PyVRP (HGS) và các gói phụ trợ
pip install pyvrp numpy tqdm pytest
```

> **Lưu ý về PyTorch (`torch`)**:
> - Nếu bạn chạy test trên file dữ liệu `.npz` (Numpy): **Không bắt buộc** phải có PyTorch.
> - Nếu bạn đọc trực tiếp file dữ liệu `.pt` trong `data/slot_datasets_v2/`: Cần cài đặt `torch` (ví dụ `pip install torch`).

### Bước 2: Kiểm tra file thực thi LKH-3
- **Trên Windows**: File `LKH.exe` đã được tải sẵn và đặt tại:
  ```text
  baselines/lkh/LKH.exe
  ```
  *(Bạn không cần cài compiler C/C++ gì thêm).*
- **Trên Linux / macOS (nếu bạn dùng Linux/WSL)**:
  Tải mã nguồn LKH-3 và gõ lệnh `make`:
  ```bash
  curl -O http://akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.14.tgz
  tar -xzf LKH-3.0.14.tgz
  cd LKH-3.0.14 && make
  cp LKH ../baselines/lkh/LKH
  ```

## 2. Sinh dữ liệu kiểm thử (Data Generation)

Nếu trong thư mục `data/` chưa có sẵn dữ liệu test cho các kích thước, bạn có thể sinh dữ liệu chuẩn (chuẩn Kool et al. 2019) trực tiếp bằng file `scripts/gen_data.py`:

### Sinh đầy đủ toàn bộ kích thước (50, 100, 200, 500, 1000) và 2 phân phối (Uniform & Gaussian):
```bash
python scripts/gen_data.py \
  --sizes 50 100 200 500 1000 \
  --loc_dist uniform gaussian \
  --n_inst 1000 \
  --seed 1234
```

> **Tùy chỉnh linh hoạt**:
> - Nếu muốn chạy thử nhanh với ít instance hơn (ví dụ 100 bài mỗi size để test trước):
>   ```bash
>   python scripts/gen_data.py --sizes 50 100 200 500 1000 --loc_dist uniform gaussian --n_inst 100
>   ```
> - Hoặc sinh cho 1 size duy nhất (ví dụ N=100 Uniform):
>   ```bash
>   python scripts/gen_data.py --num_loc 100 --loc_dist uniform --n_inst 1000
>   ```
> - Dữ liệu sẽ được tự động nén và lưu tại: `data/test/cvrp_{N}_{dist}_seed1234.npz` (chỉ cần `numpy`, tự động chạy mượt mà ngay cả khi máy không có PyTorch/Lightning).

---

## 3. Chạy kiểm thử nhanh (Verification Test)

Trước khi chạy dữ liệu lớn, hãy chạy test tự động để đảm bảo môi trường máy bạn đã nhận diện đầy đủ cả 2 solver:

```bash
python -m pytest tests/test_classical_baselines.py -v
```

Nếu màn hình hiện **`5 passed`** như dưới đây là môi trường hoàn toàn chuẩn xác:
```text
tests/test_classical_baselines.py::test_build_instance_and_capacity_table PASSED
tests/test_classical_baselines.py::test_vrplib_writer_and_tour_parser PASSED
tests/test_classical_baselines.py::test_validate_routes_catches_coverage_and_capacity_violations PASSED
tests/test_classical_baselines.py::test_hgs_solve_and_idx_offset PASSED
tests/test_classical_baselines.py::test_lkh_solve_native PASSED
```

---

## 4. Hướng dẫn chạy Benchmark (`run_classical_baselines.py`)

### 4.1. Chạy thử nghiệm nhanh (Quick Run)
Chạy thử 5 bài toán $N=50$ để kiểm tra tốc độ:

```bash
# Chạy HGS (PyVRP):
python run_classical_baselines.py --solver hgs --sizes 50 --dists uniform --n_instances 5 --n_workers 2

# Chạy LKH-3:
python run_classical_baselines.py --solver lkh --sizes 50 --dists uniform --n_instances 5 --n_workers 2
```

---

### 4.2. Chạy Benchmark đầy đủ (Full Run: Sizes 50, 100, 200, 500, 1000)

Khi chạy chính thức trên toàn bộ kích thước và cả 2 phân phối:

#### A. Chạy HGS (PyVRP):
```bash
python run_classical_baselines.py \
  --solver hgs \
  --sizes 50 100 200 500 1000 \
  --dists uniform gaussian \
  --n_workers 8 \
  --time_limits 50=5 100=10 200=20 500=60 1000=120 \
  --resume
```

#### B. Chạy LKH-3:
```bash
python run_classical_baselines.py \
  --solver lkh \
  --sizes 50 100 200 500 1000 \
  --dists uniform gaussian \
  --n_workers 8 \
  --time_limits 50=5 100=10 200=20 500=60 1000=120 \
  --resume
```

---

## 5. Giải thích các tham số dòng lệnh quan trọng

| Tham số | Mặc định | Ý nghĩa |
| :--- | :--- | :--- |
| `--solver` | **Bắt buộc** | Chọn solver muốn chạy: `hgs` hoặc `lkh`. |
| `--sizes` | `50 100 200 500 1000` | Danh sách số lượng khách hàng $N$. |
| `--dists` | `uniform gaussian` | Danh sách phân phối (tự nhận diện alias `gaussian` $\leftrightarrow$ `clustered`). |
| `--n_workers` | Số CPU core của máy | Số tiến trình chạy song song (ví dụ máy 8 cores thì để 8). |
| `--time_limits` | `None` | Giới hạn thời gian (giây) cho mỗi instance theo từng size, ví dụ: `50=5 100=10 200=20 500=60 1000=120`. |
| `--n_instances` | `None` (chạy hết) | Giới hạn số instance cần giải (rất hữu ích khi muốn test thử 10 hoặc 50 bài trước). |
| `--data_root` | `./data` | Thư mục chứa dữ liệu (tự quét cả `data/slot_datasets_v2/` và `data/test/`). |
| `--data_format` | `auto` | Định dạng file: `auto` (ưu tiên file có sẵn), hoặc ép kiểu `pt` / `npz`. |
| `--resume` | `False` | Bỏ qua những file `(size, dist)` đã có kết quả JSON trước đó, tránh chạy lại từ đầu. |
| `--save_routes` | `False` | Nếu bật cờ này, JSON kết quả sẽ lưu kèm toàn bộ mảng lộ trình (dùng khi muốn vẽ đồ thị/visualize). |

---

## 6. Kết quả đầu ra (Results)

Mỗi lần chạy xong một cặp `(size, dist)`, kết quả được lưu tại thư mục `results/`:
- Ví dụ: `results/eval_hgs_100_uniform.json`, `results/eval_lkh_100_uniform.json`.

Cấu trúc file JSON hoàn toàn khớp với schema của mô hình Deep Learning trong repo:
```json
{
  "solver": "hgs",
  "num_loc": 100,
  "dist": "uniform",
  "n_inst": 1000,
  "mean_tour_length": 15.6542,
  "std_tour_length": 1.2014,
  "elapsed_seconds": 240.5,
  "time_per_instance": 0.2405,
  "time_limit_per_instance": 10.0,
  "data_path": "..."
}
```

> **Ghi chú về tính công bằng**: 
> Chỉ số `mean_tour_length` ở đây **đã được tính toán lại trực tiếp trên tọa độ số thực $[0, 1]^2$ gốc của bài toán**, đảm bảo đối chiếu so sánh độ lệch Gap (%) chuẩn xác 100% với các mô hình Neural (MARS, POMO, SIL, ELG...).
