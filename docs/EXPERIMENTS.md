# Nhận dạng chữ số và hình học bằng CNN — giai đoạn 5

Bản chạy thử môn Xử lý ảnh. Giai đoạn 5 sửa các lỗi xử lý ảnh tìm thấy khi rà soát giai đoạn 4 (xem mục **Sửa lỗi giai đoạn 5**); trọng số CNN, dữ liệu và nhãn không đổi. Đây là nhật ký kỹ thuật được tách từ README cũ; hướng dẫn phát hành hiện tại nằm ở [README](../README.md). Các đường dẫn mã/dữ liệu bên dưới tính từ gốc repository. Báo cáo và bộ thuyết trình được chuẩn bị riêng.

## Mở ứng dụng

1. Cài Python 3.12 64-bit, giải nén đầy đủ ZIP.
2. Windows: chạy `SETUP_WINDOWS.bat` để cài thư viện lần đầu, sau đó `RUN_DEMO_WINDOWS.bat`.
3. Mở **http://127.0.0.1:5000**, giữ cửa sổ server đang chạy.

Có sẵn trọng số và dữ liệu; **không cần train lại để dùng demo**. Cài dependencies cần Internet lần đầu. Ứng dụng/đánh giá chạy offline, không gửi ảnh đến dịch vụ AI.

Chạy bằng terminal:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe app.py
```

Linux dùng `python3.12 -m venv .venv` và `.venv/bin/python`. Đổi cổng bằng `python app.py --port 5001`. Các ghi nhận Linux/trình duyệt desktop/mobile ở nhật ký này thuộc lần kiểm tra trước. Bằng chứng kiểm tra Windows và trạng thái BAT hiện tại nằm ở [RELEASE_REVIEW.md](RELEASE_REVIEW.md).

## Thử chức năng

- **Chữ số:** vẽ/tải một số 0–9; xem softmax, chuẩn hóa và feature maps. Chưa đọc chuỗi số, chữ cái hoặc tiếng Việt.
- **Hình học:** vẽ hình kín tách rời, hoặc tải PNG/JPEG/WebP/BMP; xem tròn/chữ nhật, bbox và ảnh CNN nhận. Hình vuông thuộc chữ nhật.
- **Từng bước xử lý:** chọn từng nhánh sáng/tối/màu/biên/bao lồi để xem mask trước/sau đóng hình thái, sau lọc nền và sau CNN. Không hợp một mask với phần bù để tạo ảnh minh họa.
- **Ảnh thực:** 40 ảnh / 68 vật thể mục tiêu; lọc theo development / validation / test, đối chiếu nhãn thủ công và dự đoán, mở ảnh chạy lại. Có cả ảnh không chứa mục tiêu.
- **Kiểm thử & kết quả:** MNIST, hình tổng hợp, kiểm tra trùng và các ca sai.

Ảnh hình học giữ tỉ lệ khi đưa vào canvas, backend thu nhỏ cạnh dài tối đa 800. Bbox `(x,y,rộng,cao)` dùng tọa độ `processed_size`, nửa mở. Request tối đa 8 MB; ảnh tối đa cạnh 4096 và 12 megapixel.

## Sửa lỗi giai đoạn 5

Các lỗi dưới đây được tái hiện bằng ca thử dựng sẵn trước khi sửa. Việc sửa và chọn tham số chỉ dùng ca thử, ảnh **validation** của MNIST và ảnh thực **development/validation**; không chạy ảnh test thực khi sửa. Các điều kiện chữ số dựng từ validation được đo bằng `scripts/benchmark_phase5.py --skip-mnist`; kết quả lưu ở `results/phase5_benchmark.json`. Nếu bỏ `--skip-mnist`, script còn có phép kiểm tra riêng trên 3.000 ảnh MNIST test; không mô tả toàn bộ script là chỉ dùng validation.

**Chữ số** (200 ảnh validation MNIST mỗi điều kiện, đặt trên tờ giấy 300×300):

| Điều kiện | Giai đoạn 4 | Giai đoạn 5 | Cách sửa (Chương) |
|---|---:|---:|---|
| Ánh sáng lệch 40% / 60% | 90,0% / 24,0% | 99,5% / 99,5% | Ước lượng nền bằng phép đóng/mở ảnh xám, trừ nền (black-hat/top-hat) rồi Otsu; cực tính chọn sau khi làm phẳng ánh sáng (Ch.5, Ch.6) |
| Ánh sáng lệch 60% + nhiễu | 25,0% | 99,5% | như trên |
| Giấy kẻ dòng | 85,0% | 99,5% | Xóa đoạn chạy nối hai mép ảnh, giữ chỗ nét chữ cắt qua (Ch.5) |
| Nét mảnh / bút bi 3 px / 5 px | 94,5% / 68,5% / 79,5% | 99,5% / 83,5% / 87,0% | Đo độ dày nét 2A/B, giãn ảnh xám tới ~12% chiều cao chữ (Ch.5, Ch.7) |
| Chuẩn, nhiễu Gauss, muối tiêu 5%, mực màu, phấn trên bảng | 99,5% | 99,5% | không đổi |

Ảnh nhỏ (dưới 64 px, như ảnh MNIST 28×28) giữ nguyên cách xử lý cũ: MNIST test qua pipeline vẫn **98,71%**.

**Hình học** (20 ca thử dựng sẵn; giai đoạn 4 qua 10, giai đoạn 5 qua 18):

| Lỗi | Giai đoạn 4 | Giai đoạn 5 | Cách sửa |
|---|---|---|---|
| Hình vẽ tay chưa khép kín (hở 6–16 px, nét 3–6 px) | không thấy gì | đúng | Với vùng là nét mảnh hở, thử phép đóng phần tử vuông lớn dần trong hộp bao của vùng cho tới khi viền khép lại (Ch.5) |
| Viền rộng 1 px (ảnh tải lên) | không thấy gì | đúng | Lọc trung vị xóa nét 1 px; khi ảnh ít nhiễu và lọc đã xóa ≥25% vật, thêm nhánh ngưỡng trên ảnh gốc |
| Ánh sáng lệch 60%: chữ nhật thành "ellipse" | sai | đúng | Nền nhánh trừ nền ước lượng bằng phép đóng/mở thay lọc trung bình 101×101 (lọc trung bình tạo quầng sáng giả quanh vật) |
| Hình tròn bán kính 5 px thành chữ nhật | sai | "Chưa xác định" + cảnh báo | Vùng < 12 px không phân loại |
| Lục giác đều thành hình tròn 97% | sai | "Đa giác" | Biên dạng r(θ) có ≥ 5 cực đại thì không gọi là hình tròn (Ch.7) |
| Độ tròn hiển thị: tròn 0,61, vuông xoay 45° 0,39 | sai | tròn 0,90, vuông 0,79 dù xoay | Theo dõi biên Moore, mã chuỗi Freeman, chu vi với bước chéo √2 (Ch.7 tr.9–11, 20, 26); khớp `cv2.arcLength` |
| Ảnh kết quả chỉ có khung, khó thấy vùng nào được nhận | khó đọc | tô màu cả vùng theo mặt nạ: **xanh** = tròn/chữ nhật, **cam** = hình khác, **đỏ** = chưa xác định; chữ nhãn và dòng trong danh sách cùng màu; nút "Ẩn vùng chưa xác định" cho ảnh nền phức tạp | Chỉ hiển thị (`vision/annotate.py`); có thêm `unknown_reason` = `low_confidence` / `too_small` |
| Chữ số: không có ảnh kết quả; điểm dưới 75% vẫn hiện màu xanh | khó đọc | ảnh kết quả tô **xanh** nét được đọc nếu điểm ≥ 75%, **đỏ** nếu dưới ngưỡng; vết nhỏ bị bỏ qua tô **đỏ**; nhãn ghi số cụm nét bị gộp; chữ số lớn và thanh điểm cùng màu | Chỉ hiển thị; API thêm `confident`, `merged_parts`, `ignored_parts`, `annotated` |

Không tách được: **hai hình chạm nhau** (vẫn 0/17 trên bộ stress) và **bát giác** (quá gần hình tròn). `fill_holes` viết lại bằng gán nhãn theo đoạn chạy, trùng khớp bản cũ, nên toàn bộ đánh giá ảnh thực nhanh hơn (~48 s so với ~64 s).

**Ảnh thực, cùng trọng số CNN** (giai đoạn 4 → 5): development 34 TP / 5 FP / 14 FN → **34 / 3 / 14**; validation 9 / 1 / 1 → 9 / 1 / 1; test 9 / 3 / 1 → 9 / 3 / 1. Không ảnh nào kém đi.

**Giao thức:** kết quả giai đoạn 4 lưu ở `results/history/v4/`, mã giai đoạn 4 ở `reference_v4/`. Cấu hình giai đoạn 5 chọn lại trên validation (vẫn `repaired_regions`, ngưỡng 0,90) rồi khóa. Ảnh test thực **đã mở lần đầu ở giai đoạn 4**, nên điểm test giai đoạn 5 không còn là đánh giá trên ảnh chưa từng thấy; bản khóa và giao diện ghi rõ điều này. Sau khi khóa, một lỗi chọn cực tính ở nhánh chữ số được phát hiện khi thử giao diện; bản khóa được **sửa đổi có ghi lại** (`amendments` trong `protocol_lock.json`, lệnh `--amend`). Dự đoán hình học trên cả 40 ảnh thực giống hệt trước và sau sửa đổi. Các sửa đổi sau đó (chỉ đổi cách vẽ ảnh kết quả hình học và chữ số) cũng được ghi lại như vậy; 40 ảnh thực và 102 cảnh tổng hợp cho nhãn, khung và điểm giống hệt.

## Đã kiểm tra lỗi ở giai đoạn 4

Bản giai đoạn 3 qua 30 kiểm tra nhưng chưa bao phủ các vấn đề sau; đã bổ sung hồi quy:

| Vấn đề | Cách xử lý giai đoạn 4 |
|---|---|
| Preview sau đóng bị trắng vì hợp vùng sáng và tối | Lưu từng nhánh riêng; preview chính chỉ hợp các ứng viên đã qua lọc nền |
| Validation chỉ có 6 chữ nhật, không có hình tròn/ảnh âm | Validation mới có 2 tròn, 8 chữ nhật, 2 ảnh không có mục tiêu |
| Nhãn theo vật thể tròn ngoài đời mâu thuẫn lớp ellipse | Quy định theo hình dạng 2D nhìn thấy; hai ảnh cũ đổi thành distractor ellipse, lưu thay đổi và kết quả gốc |
| Loại mọi vùng chạm ít nhất 3 mép làm mất cả đĩa tròn tiếp xúc khung | Kết hợp số mép và tỉ lệ pixel biên, giữ các vùng tròn chỉ tiếp xúc mép ở ít điểm |
| Các vùng gần chữ nhật bị thiếu lõm đường bao | Thử bao lồi có giới hạn cho đề xuất gần lồi; giữ đề xuất gốc để đối chiếu CNN |
| CNN 4 lớp buộc vùng nền thành một trong các hình | Học thêm lớp thứ 5 `background`, loại vùng nền trước lọc cha/con và NMS |
| Đếm trùng scene bằng cách chuẩn hóa cũ | Hash toàn bộ tensor 32×32 thực tế của mọi nhánh, trước suy luận; ghi rõ scene cũ là kiểm tra hồi quy |
| Kết quả JSON hỏng gây lỗi không rõ | API trả JSON/503 với hướng dẫn chạy lại đánh giá |

Các sửa giai đoạn trước vẫn giữ: kernel lẻ dương, xử lý checkpoint hỏng, ghép dự đoán–nhãn một–một, chặn số đo cũ khi đổi mô hình/mã. Đặc trưng hình học trên giao diện tính từ mask thực sự đưa đi chuẩn hóa, kể cả mask đã sửa bằng bao lồi.

## Kiến thức áp dụng

| Nội dung | Tài liệu | Mã |
|---|---|---|
| Mức xám, median, trung bình nền và phép trừ cường độ | Chương 2 | `vision/processing.py` |
| Giãn/co ảnh xám, phép đóng/mở ước lượng nền, black-hat/top-hat | Chương 5 | `gray_dilate`, `gray_erode`, `estimate_background`, `digit_foreground` |
| Các thành phần màu RGB | Chương 3 | `segmentation.py`, `service.py` |
| Giãn/co/đóng, lấp lỗ, thành phần liên thông, bao lồi | Chương 5, PDF trang 50–52, 66–73 | `processing.py`, `segmentation.py` |
| Otsu, Sobel và phân đoạn | Chương 6, PDF trang 18–22, 77–83 | `processing.py`, `segmentation.py` |
| Theo dõi biên Moore, mã chuỗi Freeman, biên dạng r(θ) | Chương 7, PDF trang 9–11, 17 | `trace_boundary`, `chain_code`, `boundary_signature` |
| Diện tích, chu vi mã chuỗi, độ tròn, độ lấp đầy, độ lệch tâm | Chương 7, PDF trang 20, 25–28 | `geometric_features`, `oriented_extent` |
| CNN, train/validation/test và khái quát hóa | Chương 8, PDF trang 8–9, 49–58 | `models.py`, `training_data.py`, các script train/evaluate |

Phù hợp yêu cầu Chương 0, PDF trang 8–9 và 15: CNN cho MNIST và tròn/chữ nhật; tự cài các bước xử lý ảnh. Thuật toán ảnh dùng NumPy, không gọi thuật toán xử lý ảnh OpenCV/SciPy/skimage. Pillow chỉ đọc/ghi ảnh và vẽ nhãn; PyTorch thực hiện mạng và tối ưu. Bao lồi/nhánh màu là các lựa chọn thí nghiệm, không phải mọi thuật toán trong giáo trình đều bắt buộc.

**Hình học:** gray/RGB → median → Otsu hai cực tính, trừ nền khi biên nền biến thiên, đề xuất từ biên/màu → đóng hình thái → vùng liên thông → lọc nền/vùng nhỏ → lấp lỗ; thêm biến thể bao lồi cho vùng gần lồi → chuẩn hóa 32×32 → CNN 5 lớp → loại nền, chi tiết trong và khung trùng. Ngưỡng cơ bản 0,75; với nền có độ lệch chuẩn mép ≥12, ngưỡng chọn trên validation là 0,90. Các ngưỡng này là cấu hình thí nghiệm, không phải xác suất đã hiệu chỉnh.

**CNN:** Conv(1→16,3×3)/ReLU/MaxPool → Conv(16→32,3×3)/ReLU/MaxPool → FC64/ReLU/Dropout → FC5. Năm lớp: circle, rectangle, triangle, ellipse, background. Hai lớp mục tiêu vẫn là circle/rectangle. CNN quyết định nhãn; luật hình học chỉ lọc/đề xuất vùng, trừ hai ngoại lệ ở giai đoạn 5: vùng < 12 px ghi "Chưa xác định", và "hình tròn" có biên dạng ≥ 5 góc ghi "Đa giác". Độ lấp đầy khung quay lấy mẫu cách 3°; độ tròn dùng chu vi theo mã chuỗi 8 hướng (sai số ~8% ở góc 30° do chỉ có 8 hướng).

**Chữ số:** gray/alpha → (nền đều: Otsu/cực tính | nền không đều: trừ nền bằng phép đóng/mở rồi Otsu) → xóa dòng kẻ → vùng nét chính → giãn nét quá mảnh → cắt/giữ tỉ lệ 20×20 → đệm 28×28/căn trọng tâm → CNN 10 lớp. Trọng số chữ số không thay đổi; giao diện ghi rõ bước nào đã được áp dụng.

## Nhãn và dữ liệu

Tổng **40 ảnh / 68 vật thể mục tiêu**, thuộc **33 nhóm nguồn/chuỗi chụp** bảo thủ. Trong 21 ảnh bổ sung có hai bản quét vật thật được ghi `capture_type=scan`; các ảnh này không được gọi là ảnh camera. Không dùng ảnh AI hoặc ghép vật thể để làm ảnh thực.

| Vai trò | Ảnh | Tròn | Chữ nhật | Ảnh không có mục tiêu |
|---|---:|---:|---:|---:|
| Development, cho phép học thêm CNN | 19 | 38 | 10 | 3 |
| Validation, chọn mô hình và pipeline | 10 | 2 | 8 | 2 |
| Test đã mở từ giai đoạn 4 | 11 | 6 | 4 | 2 |

Số ảnh âm nằm trong tổng số ảnh, không cộng thêm. Cùng nguồn/cùng chuỗi ảnh luôn giữ một vai trò. Tám ảnh test đã xem ở giai đoạn 3 chuyển sang development; không tiếp tục gọi chúng là test độc lập.

Nhãn dựa vào **đường bao 2D**: hình tròn nhìn nghiêng rõ thành ellipse là distractor, không gán circle chỉ vì vật ngoài đời là đĩa. Chữ, lỗ, trục, hình in và bóng bên trong không là đối tượng riêng. Những vùng che khuất vẫn là ca stress với bbox ước lượng; vùng cắt mép dùng phần nhìn thấy. Nhãn kiểm tra trực quan trên lưới tọa độ trước dự đoán, chưa có người gán nhãn thứ hai hay mask pixel chính xác.

Nguồn/tác giả/giấy phép ở `data/real/ATTRIBUTION.md`; file và pixel có SHA-256 trong `manifest.json`. Thay nhãn ở `annotation_changes.json`, khóa nhãn ở `annotation_freeze.json`. Kết quả và nhãn giai đoạn cũ ở `results/history/v3/`.

## Dữ liệu học và chọn cấu hình

CNN học **16,154 đầu vào 32×32** (dấu phẩy phân cách hàng nghìn): gồm 8.000 mẫu tổng hợp gốc, biến thể biên/góc vát, vùng development phù hợp bbox và các vùng nền không đều. Có 2.400 mẫu lớp background. **1,095 ứng viên validation** được giữ riêng, cùng 1.600 mẫu validation tổng hợp.

Các vùng real được gán nhãn bằng độ khớp bbox theo quy tắc công bố trước, là giám sát yếu, không phải mask thủ công hoàn hảo. Tăng cường chỉ ở train: xoay/nội suy, co/giãn nhẹ, biên không hoàn hảo. Hash ảnh chuẩn hóa uint8 không ghép nhãn: không trùng trong train, không trùng train–validation, không trùng phần synthetic dành riêng. Nhóm nguồn cũng không giao nhau. Chưa bảo đảm loại hết gần trùng về cảm nhận.

Chọn epoch bằng macro-F1 phát hiện toàn ảnh validation, có kiểm tra accuracy validation tổng hợp ≥0,995 để giữ khả năng nhận hình đơn giản. Sau đó so sánh các nhánh tách vùng và ngưỡng trên cùng validation, không dùng test. Điểm validation không phải độ chính xác độc lập ngoài thực tế.

`protocol_lock.json` khóa mô hình, dữ liệu và mã; `test_access_log.json` ghi lần mở nhóm test mới. Sau khi xem test, không ghi lại cấu hình để tiếp tục coi cùng ảnh là test chưa thấy. `select_geometry_pipeline.py`/`freeze_geometry_protocol.py` từ chối ghi đè sau khi đã mở test của thử nghiệm này.

## Kết quả chạy lại

- MNIST gốc 10.000 ảnh: **99,08%**; qua tiền xử lý ứng dụng: **98,71%**.
- ANN gốc trên cùng MNIST raw: **98,50%**; quy trình huấn luyện khác nên không phải ablation chỉ thay kiến trúc.
- Phân loại 1.200 hình synthetic test cố định: **99,92%**, macro-F1 **0.9996**.
- Hồi quy 90 scene tổng hợp: **267 TP / 0 FP / 0 FN**.
- 12 scene chạm/chồng: **0 TP / 1 FP / 17 FN**; chưa có tách vật thể dính/che khuất.

Các số trên giống hệt giai đoạn 4 (mã sửa không làm tụt số đo đã khóa nào).

So sánh **cùng 11 ảnh test và cùng nhãn**, IoU ≥0,5, mỗi dự đoán/nhãn ghép tối đa một lần:

| Hệ thống | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Hệ thống giai đoạn 3 | 7 | 14 | 3 | 33,33% | 70,00% | 0.4516 |
| Tách vùng mới + CNN cũ | 9 | 4 | 1 | 69,23% | 90,00% | 0.7826 |
| Tách vùng mới + CNN mới (giai đoạn 4) | 9 | 3 | 1 | 75,00% | 90,00% | 0.8182 |
| Giai đoạn 5 (cùng CNN, sửa xử lý ảnh) | 9 | 3 | 1 | 75,00% | 90,00% | 0.8182 |

Ảnh test đã mở ở giai đoạn 4 nên dòng giai đoạn 5 không phải đánh giá độc lập; mức cải thiện của giai đoạn 5 thấy trên development (FP 5 → 3) và trên ca thử. Tách vùng mới giúp phần lớn mức tăng của giai đoạn 4; CNN học thêm giảm tiếp một khung dư trong test này. Đây là mẫu nhỏ có các ảnh nền đơn giản, hai ảnh âm và ca khó; không suy ra accuracy chung cho mọi ảnh thực. Không so trực tiếp với 3/9 của tập test khác ở giai đoạn 3.

Ca sai test còn lại: đĩa mềm trắng gần màu nền, bị cắt khung (bỏ sót một vật và hai khung dư), cùng một khung dư trên vân gỗ. Hình chồng nhau trong bộ development/stress vẫn khó. Phần chẩn đoán phân biệt không có đề xuất phù hợp, có ứng viên bị loại, sai lớp/điểm thấp và vùng đã ghép cho vật khác.

Chẩn đoán khi cho sẵn bbox thật: đúng **51/68 vùng (75,00%)**. Có trợ giúp vị trí nên không phải accuracy phát hiện toàn ảnh.

Bộ 90 scene là hồi quy dùng lại từ giai đoạn 3. Audit nay hash mọi đầu vào CNN thực tế trước suy luận, ghi cả trùng giữa các nhánh/scene và giao với train; không loại scene theo kết quả. Benchmark phân loại synthetic test và test real nguồn mới được ghi riêng. Macro-F1 phân loại lấy trung bình trên các lớp có nhãn thật; cột background vẫn ghi lỗi từ chối khi dự đoán nó trên test hình.

## Kiểm tra và tái lập

Chạy từ thư mục chứa `app.py`, bằng Python trong môi trường đã cài:

```powershell
python -m unittest discover -s tests -v
python scripts/prepare_geometry.py
python scripts/prepare_real_training.py
python scripts/compare_segmentation.py --include-test
python scripts/evaluate.py
python scripts/benchmark_phase5.py
```

Bản đi kèm qua **68 kiểm tra tự động** (52 kiểm tra cũ + 16 kiểm tra hồi quy giai đoạn 5 trong `tests/test_phase5.py`) và kiểm tra giao diện trên trình duyệt desktop (chữ số, hình học, kết quả, ảnh thực; không lỗi JavaScript). Kiểm tra trên mobile và chạy tệp BAT trên Windows chưa được làm lại ở giai đoạn 5.

Đổi mã xử lý ảnh sau khi đã mở test của một giai đoạn: chạy `select_geometry_pipeline.py --amend "lý do"` rồi `freeze_geometry_protocol.py --amend "lý do"`; chỉ dùng cho thay đổi không xuất phát từ kết quả test, mọi sửa đổi được lưu trong `protocol_lock.json`.

Lần rà vận hành bổ sung đã kiểm tra cấu trúc JSON, giữ nguyên báo cáo đầy đủ khi chạy development, khôi phục huấn luyện khi thiếu checkpoint/thư mục, và ngưỡng validation cố định 0,75. Chi tiết lần rà đó (giai đoạn 4) ở `results/history/v4/operational_audit.json`.

- `python scripts/evaluate_real.py` ghi `results/real_development_evaluation.json`, không ghi đè gallery đầy đủ. Thêm `--include-test` mới cập nhật báo cáo chính theo cấu hình khóa.
- `python scripts/compare_segmentation.py` không có `--include-test` ghi `segmentation_comparison_development.json` riêng.
- `evaluate_real.py --exploratory-real` ghi báo cáo khám phá riêng, không gắn số đo mới với ảnh đối chiếu cũ.
- `evaluate.py --exploratory-real` ghi số đo/ảnh/CSV trong `results/exploratory/`; chạy tập con bằng `--scene-count`/`--shape-test-count` ghi trong `results/partial/`. Các tệp chuẩn của demo được giữ nguyên.
- JSON ghi theo cách thay thế nguyên tệp; lỗi ghi không xóa báo cáo trước đó. Metadata ghi ngưỡng thực tế từng ảnh và chính sách 0,75/0,90; không gọi mọi ảnh là cùng ngưỡng 0,75.
- `--limit-train` chỉ dành cho digits. Bộ geometry đã khóa nên `--shape-samples` phải giữ 8000; dữ liệu học thực tế sau bổ sung là 16.154 mẫu như bảng trên.

Hai lệnh prepare mặc định kiểm tra/cache dữ liệu đã có; không sinh/train lại. `--force` chỉ dùng khi chủ động lập thử nghiệm mới. `evaluate.py` chỉ đọc trọng số và ghi số đo. Đổi mô hình/mã sẽ làm API chặn số đo cũ và yêu cầu khởi động lại server; cấu hình khác khóa chỉ được đánh giá với `--exploratory-real`, không coi cùng test là chưa thấy.

Muốn huấn luyện lại để nghiên cứu:

```powershell
python scripts/train_geometry.py --epochs 12 --threads 2
python scripts/evaluate.py --exploratory-real
```

Train ghi đè checkpoint; không cần thực hiện để demo. Thử nghiệm tiếp theo cần quy định lại development/validation và chuẩn bị nhóm test nguồn mới. `train.py --task shapes` chuyển tới quy trình geometry mới; `--task digits` vẫn dùng MNIST 55.000 train / 5.000 validation, augmentation chỉ ở train.

## Tệp chính

- `vision/processing.py`, `segmentation.py`, `service.py`: xử lý ảnh, vùng và suy luận dùng chung; `annotate.py` vẽ ảnh kết quả (tô màu theo nhãn).
- `vision/training_data.py`, `scripts/prepare_real_training.py`, `scripts/train_geometry.py`: dữ liệu học bổ sung và chọn CNN bằng validation.
- `vision/real_data.py`, `real_evaluation.py`, `metrics.py`, `provenance.py`: audit, đánh giá, ghép một–một, chặn số đo cũ.
- `scripts/select_geometry_pipeline.py`, `freeze_geometry_protocol.py`: chọn và khóa thử nghiệm trước test.
- `reference_v3/`, `reference_v4/`: mã/trọng số hình học giai đoạn 3 và 4 để so sánh cặp; không phải luồng app chính.
- `results/history/v4/`: kết quả, bản khóa và nhật ký mở test của giai đoạn 4.
- `scripts/benchmark_phase5.py`, `tests/test_phase5.py`: bộ đo và kiểm thử hồi quy cho các lỗi đã sửa.
- `reference_original/`: ANN ban đầu để đối chiếu.
- `results/evaluation.json`, `real_evaluation.json`, `segmentation_comparison.json`: kết quả và ablation; ảnh đối chiếu ở `real_*_comparison.png`.

API: `GET /api/health`, `POST /api/digits`, `POST /api/shapes`, `GET /api/metrics`, `GET /api/real`; `/predict` là alias chữ số. POST nhận JSON `{"image":"data:image/png;base64,..."}`. Dataset MNIST kèm sẵn từ bản phân phối Keras/TensorFlow, checksum trong `data/source.json`.
