# CNN Lab — Nhận dạng chữ số viết tay và hình học đơn giản

Bài tập lớn môn **Xử lý ảnh**, lớp **INT13146-N09**, **Nhóm 01**.

| Thành viên | Mã sinh viên |
|---|---|
| Nguyễn Tuấn Dũng | B23DCCN203 |
| Trần Tuấn Anh | B23DCCN049 |

Ứng dụng chạy trên máy cá nhân, kết hợp thuật toán xử lý ảnh tự cài bằng NumPy với hai mạng CNN PyTorch. Người dùng có thể vẽ hoặc tải ảnh, xem nhãn dự đoán, vùng nhận dạng và từng bước xử lý. Bản final là **giai đoạn 5**; không cần huấn luyện lại để demo.

## Chức năng và phạm vi

- **Chữ số:** phân loại **một chữ số 0–9**, hiển thị điểm softmax, ảnh chuẩn hóa 28×28 và tám bản đồ kích hoạt của lớp tích chập đầu tiên. Chưa đọc chữ cái, văn bản hoặc chuỗi số.
- **Hình học:** tìm và đếm **hình tròn / hình chữ nhật** tách rời nhau. Hình vuông thuộc lớp chữ nhật. CNN còn học tam giác, ellipse và nền để hạn chế nhầm; các lớp này không được cộng vào hai lớp mục tiêu.
- **Quan sát xử lý ảnh:** mức xám, median, Otsu, các nhánh sáng/tối/màu/biên, hình thái học, lọc vùng và đầu vào CNN.
- **Ảnh thực:** 40 ảnh có nguồn và nhãn, 68 đối tượng mục tiêu; đối chiếu nhãn với dự đoán và xem các ca sai.
- **Kết quả:** chỉ số đã lưu, ma trận nhầm lẫn và các giới hạn của thí nghiệm.

Ảnh được xử lý tại máy đang chạy Flask, không gửi đến dịch vụ AI. Cài thư viện lần đầu cần Internet; trọng số và dữ liệu cần thiết đã kèm để demo và kiểm thử offline.

## Cài đặt trên Windows

Cần **Python 3.12 64-bit kèm Python Launcher**. Không sao chép `.venv` từ máy khác và không dùng Python mặc định chưa kiểm tra phiên bản.

```powershell
git clone https://github.com/NTDung123/CNN-Lab.git
cd CNN-Lab
```

1. Tải/clone đầy đủ repository.
2. Chạy `SETUP_WINDOWS.bat` để tạo môi trường và cài thư viện.
3. Chạy `RUN_DEMO_WINDOWS.bat`.
4. Mở **http://127.0.0.1:5000** và giữ cửa sổ server hoạt động. Nhấn `Ctrl+C` để dừng.

Cách chạy bằng PowerShell tại thư mục chứa `app.py`:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe scripts/check_release.py
.\.venv\Scripts\python.exe app.py
```

Nếu `.venv` đã tồn tại nhưng hỏng hoặc sai phiên bản, đổi tên thư mục đó rồi tạo lại. Script không tự xóa môi trường cũ. Nếu cổng 5000 đang bận:

```powershell
.\.venv\Scripts\python.exe app.py --port 5001
```

Khi đó mở http://127.0.0.1:5001. Cảnh báo development server của Flask là phù hợp với demo local; repository chưa cấu hình vận hành công khai trên Internet.

## Linux và Docker

Linux x86-64 có Python 3.12:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/check_release.py
.venv/bin/python app.py
```

Docker là cách chạy tùy chọn, cần Docker Engine đang hoạt động:

```bash
docker build -t cnn-lab .
docker run --rm -p 127.0.0.1:5000:5000 cnn-lab
```

Ảnh Docker dùng Flask cho demo. Trạng thái kiểm tra build/runtime được ghi riêng trong [lần rà soát phát hành](docs/RELEASE_REVIEW.md), không suy ra từ việc có Dockerfile.

## Pipeline và mô hình

**Chữ số:** mức xám / alpha → xác định nền và cực tính → Otsu hoặc trừ nền → bỏ dòng kẻ → giữ cụm nét chính → làm dày nét quá mảnh → giữ tỷ lệ 20×20, đệm/căn trọng tâm 28×28 → CNN 10 lớp.

**Hình học:** mức xám / RGB → median → đề xuất vùng từ các nhánh ngưỡng, trừ nền, màu và biên → hình thái học, thành phần liên thông → lọc nền / vùng nhỏ, sửa một số viền hở → chuẩn hóa 32×32 → CNN 5 lớp → lọc nền, chi tiết bên trong và khung trùng.

CNN dùng hai khối Conv/ReLU/MaxPool rồi FC64/ReLU/Dropout và đầu ra phân loại. Có **105.866 tham số** ở mạng chữ số và **136.261 tham số** ở mạng hình học. Các đặc trưng hình học phục vụ hiển thị và một số luật lọc/hậu xử lý; không được nối vào vector FC. Nhãn “đa giác” là luật hậu xử lý, không phải lớp thứ sáu của CNN.

NumPy thực hiện các thuật toán ảnh; Pillow đọc/ghi ảnh và vẽ nhãn; PyTorch thực hiện mạng và tối ưu. Pipeline không gọi thuật toán ảnh của OpenCV, SciPy hoặc scikit-image. Điểm softmax chưa được hiệu chỉnh thành xác suất dự đoán đúng.

## Dữ liệu và kết quả đã lưu

| Phép đo | Kết quả | Phạm vi |
|---|---:|---|
| MNIST tensor gốc | 99,08% | 10.000 ảnh test |
| MNIST qua pipeline ứng dụng | 98,71% | Cùng 10.000 ảnh test |
| Phân loại crop hình tổng hợp | 1.199/1.200 đúng | Không đo khả năng tự tìm vật trong toàn ảnh |
| Phát hiện 90 cảnh tổng hợp | 267 TP / 0 FP / 0 FN | Bộ hồi quy đã dùng từ giai đoạn 3 |
| 12 cảnh tổng hợp chạm/chồng | 0 TP / 1 FP / 17 FN | Hạn chế chưa giải quyết |
| 11 ảnh thực mang vai trò test | 9 TP / 3 FP / 1 FN | Precision 75%, recall 90%, F1 0,8182; test đã mở từ giai đoạn 4 |

Các số trên lấy từ hồ sơ kèm repository, không gộp thành một accuracy chung cho toàn ứng dụng. Tập ảnh thực chỉ có 40 ảnh từ 33 nhóm nguồn. Test thực ở giai đoạn 5 là **đối chiếu hồi quy, không còn độc lập**. Hai hình chạm/chồng, vật trắng gần nền và nền có kết cấu vẫn dễ sai.

- MNIST: 55.000 train / 5.000 validation / 10.000 test.
- Hình tổng hợp cơ sở: 8.000 train / 1.600 validation / 1.200 test.
- Dữ liệu hình học hybrid: 16.154 đầu vào train; validation gồm 1.600 crop tổng hợp và 1.095 vùng ảnh thực.
- Ảnh thực: development 19 ảnh / 48 mục tiêu; validation 10 / 10; test đã xem 11 / 10. Bảy ảnh âm đã nằm trong tổng 40 ảnh.

Chi tiết điều kiện, lựa chọn mô hình, nguồn dữ liệu và các thay đổi giữa giai đoạn nằm trong [nhật ký kỹ thuật](docs/EXPERIMENTS.md). Kết quả chuẩn: [`results/evaluation.json`](results/evaluation.json), [`results/real_evaluation.json`](results/real_evaluation.json), [`results/phase5_benchmark.json`](results/phase5_benchmark.json).

## Kiểm tra bản clone

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/check_release.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`check_release.py` chỉ đọc dữ liệu: kiểm tra checksum, 14 tài sản suy luận trong ba báo cáo, khóa giao thức, dữ liệu cache và HTTP contract bằng Flask test client, rồi suy luận hai ảnh mẫu. Script không huấn luyện, không sinh lại dữ liệu và không ghi đè benchmark. Đây không phải kiểm tra trực quan trình duyệt hay phép đo lại accuracy.

GitHub Actions được cấu hình cho Windows và Ubuntu, Python 3.12. Kết quả chạy local và các phần chưa kiểm tra nằm trong [RELEASE_REVIEW.md](docs/RELEASE_REVIEW.md); CI chỉ có trạng thái sau khi repository được push và workflow thực sự chạy.

## Huấn luyện và đánh giá lại

**Không cần thực hiện phần này để demo.** Huấn luyện có thể ghi đè checkpoint và làm số đo đã lưu không còn khớp. Giữ bản sao dữ liệu, trọng số, cấu hình và kết quả trước khi mở thí nghiệm khác.

```powershell
.\.venv\Scripts\python.exe scripts/train.py --task digits
.\.venv\Scripts\python.exe scripts/train_geometry.py --epochs 12 --threads 2
.\.venv\Scripts\python.exe scripts/evaluate.py --exploratory-real
```

`prepare_geometry.py` và `prepare_real_training.py` mặc định kiểm tra/tái dùng cache; `--force` mới sinh lại dữ liệu. Cache hybrid bị từ chối khi manifest nhãn/vai trò hoặc bộ hình học gốc đổi. Mã xử lý thay đổi vẫn được ghi nhận riêng để giữ khả năng dùng bộ dữ liệu đã khóa của giai đoạn trước.

`benchmark_phase5.py --skip-mnist` bỏ phép đo riêng trên MNIST test. Khi không truyền tùy chọn này, script còn đọc 3.000 ảnh MNIST test; không mô tả toàn bộ script là chỉ dùng validation. Ảnh thực test đã xem không trở thành test mới khi chạy lại. Muốn đánh giá khái quát hóa cần thu một nhóm test nguồn mới và khóa quy trình trước khi mở dự đoán.

## Tổ chức repository

| Đường dẫn | Vai trò |
|---|---|
| `app.py`, `templates/`, `static/` | Flask, giao diện canvas và trình bày kết quả |
| `vision/` | Thuật toán ảnh, CNN, phân vùng, dữ liệu và giao thức đánh giá |
| `scripts/`, `tests/` | Chuẩn bị dữ liệu, huấn luyện, đánh giá, kiểm tra phát hành và hồi quy |
| `data/`, `weights/`, `examples/` | Dữ liệu, checkpoint và ảnh mẫu cần cho demo offline |
| `results/` | Kết quả đã lưu, ảnh đối chiếu và lịch sử thí nghiệm |
| `reference_original/` | ANN kế thừa và checkpoint baseline |
| `reference_v3/`, `reference_v4/` | Các bản đối chiếu; bản v3 còn cung cấp warm start cho huấn luyện hình học |
| `docs/` | Nhật ký kỹ thuật và biên bản rà soát trước khi push |

Không đưa `.venv`, bản môi trường từ máy khác, cache, khóa bí mật và tệp tạm vào Git. Giữ dữ liệu, checkpoint và kết quả chuẩn; không ignore toàn bộ `*.npz`, `*.pt` hay `results/`. `.gitattributes` giữ LF cho mã/manifests để checkout Windows không làm sai SHA-256 đã khóa.

## Nguồn kế thừa và ghi nhận

Dự án bắt đầu từ [phanmanhcuongdev/identity_number](https://github.com/phanmanhcuongdev/identity_number), sau đó điều chỉnh cho môn Xử lý ảnh. Link đó là **repo nguồn**, không phải repo final của nhóm. Bản này chưa tự gán một giấy phép chung cho mã kế thừa và dữ liệu.

Xem [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), [nguồn MNIST](data/source.json), [tác giả và giấy phép từng ảnh thực](data/real/ATTRIBUTION.md). Khi phân phối lại ảnh nhãn, ảnh đối chiếu hoặc dữ liệu dẫn xuất, giữ ghi nhận nguồn tương ứng.

Báo cáo Word và bộ thuyết trình của nhóm được chuẩn bị riêng. Bảng nhiệm vụ kỹ thuật và phần trăm đóng góp cần do hai thành viên xác nhận theo thực tế trước khi nộp.
