# Rà soát trước khi đưa mã nguồn lên GitHub

Ngày rà soát: **21/11/2026**. Phạm vi: bản final của Nhóm 01 trên Desktop, giai đoạn 5. Bản cũ tại ổ D không được dùng để đóng gói.

## Nhận xét về mức độ phù hợp

Dự án phù hợp phạm vi đơn giản “nhận dạng chữ viết và hình dạng đơn giản bằng mạng neural” khi nêu rõ đối tượng chữ viết ở đây là **một chữ số 0–9**, còn hai hình mục tiêu là **tròn và chữ nhật**. Có hai CNN thật, dữ liệu học/validation/test, thuật toán ảnh tự cài, giao diện demo và kết quả có nguồn kiểm chứng.

Điểm mạnh cho môn Xử lý ảnh là có thể quan sát và giải thích các bước tạo đầu vào mạng: cực tính, phân ngưỡng, ước lượng nền, hình thái học, thành phần liên thông, chuẩn hóa và lọc vùng. Trọng tâm bảo vệ nên là tác động của các bước này, thay vì chỉ nêu accuracy của CNN.

| Yêu cầu học phần | Minh chứng và phần còn cần chốt |
|---|---|
| Bài toán và ứng dụng | README, giao diện một chữ số / hai hình mục tiêu; chưa triển khai OCR văn bản |
| Dữ liệu và đặc điểm đầu vào | `data/source.json`, các manifest, nguồn và phân chia vai trò trong nhật ký kỹ thuật |
| Pipeline xử lý ảnh | `vision/processing.py`, `segmentation.py`, các ảnh trung gian của giao diện |
| Cài đặt và demo | Flask, hai BAT Windows, Docker tùy chọn, checkpoint đi kèm |
| Thiết kế thí nghiệm và đánh giá | Kết quả đã lưu, phân biệt crop classification với detection, ghép một–một theo IoU |
| Phân tích thất bại | Ca sai MNIST, hình chạm/chồng, vật trắng và ảnh âm có kết cấu |
| Báo cáo và trình bày | Đã chuẩn bị riêng bản Word và Canva; kiểm tra nội dung bản nộp cùng phiên bản source |
| Tỷ lệ đóng góp kỹ thuật | Hai thành viên cần điền theo thực tế; hiện để trống theo yêu cầu của nhóm |

## Những sửa đổi trong lần chuẩn bị này

1. Thêm `.gitignore`, không đưa môi trường ảo, cache và tệp bí mật vào Git; giữ các dataset, checkpoint và kết quả cần cho demo offline.
2. Thêm `.gitattributes` để mã Python/manifests JSON được checkout LF trên Windows, bảo toàn SHA-256 của mô hình/pipeline đã khóa.
3. Cập nhật `.dockerignore`, loại cả `.venv_from_other_pc` và các môi trường `.venv*` khỏi Docker context.
4. Kiểm tra Python 3.12 64-bit trong hai BAT; bỏ fallback tạo venv bằng Python mặc định chưa kiểm tra. Script không tự xóa venv cũ. BAT chạy demo nhận thêm đối số như `--port 5001`.
5. Sửa cache dữ liệu hybrid: khi manifest nhãn/vai trò hoặc bộ hình học gốc thay đổi, yêu cầu sinh lại bằng `--force`. Vẫn cho phép dùng nguyên bộ dữ liệu đã khóa nếu chỉ mã tạo/xử lý dữ liệu thay đổi, và ghi nhận sự khác biệt đó.
6. Thêm ba kiểm thử hồi quy cho cache dữ liệu và một kiểm thử bảo đảm script kiểm tra phát hành từ chối cache thiếu mà không sinh lại dữ liệu. Sửa mô tả giao diện để không gọi test đã xem là “kiểm tra mới”.
7. Viết lại README cho người clone lần đầu; chuyển nhật ký kỹ thuật cũ sang `docs/EXPERIMENTS.md`, làm rõ benchmark mặc định còn có phép đo MNIST test riêng.
8. Ghi nhận repo kế thừa, nguồn ảnh và phạm vi giấy phép trong `THIRD_PARTY_NOTICES.md`; không gán giấy phép MIT chung cho tài sản chưa xác định quyền cấp phép lại.
9. Thêm `scripts/check_release.py` kiểm tra đọc dữ liệu và HTTP contract, cùng workflow kiểm thử Windows/Ubuntu trên GitHub Actions.

Các thay đổi trên không sửa trọng số, nhãn, cấu hình hình học, khóa giao thức hoặc các số đo chuẩn đã lưu.

## Kiểm chứng đã thực hiện

Môi trường local: **Windows 11, Python 3.12.10 64-bit**, PyTorch **2.7.1+cpu**, NumPy **2.3.5**, Flask **3.1.3**, Pillow **13.0.0**.

| Kiểm tra | Kết quả |
|---|---|
| `python -m pip check` | Đạt, không có phụ thuộc bị xung đột |
| Bộ kiểm thử bản cuối | **72 kiểm thử đạt**, 38,113 giây trên bản clone sạch với venv cài mới; đây không phải thời gian suy luận một ảnh |
| `python scripts/check_release.py` | Đạt; ba báo cáo khớp toàn bộ 14 tài sản suy luận |
| Dataset và giao thức | MNIST khớp checksum; 40 ảnh thực, 33 nhóm nguồn, 68 mục tiêu; khóa giao thức không có tài sản thay đổi |
| HTTP contract | Mười GET trả 200; ảnh mẫu chữ số 6 được nhận là 6, ảnh hình học mẫu trả 1 tròn / 3 chữ nhật |
| JavaScript | `node --check static/script.js` đạt; kiểm tra tĩnh ID và trường dữ liệu frontend khớp backend |
| BAT Windows | `SETUP_WINDOWS.bat` thoát 0 với venv Python 3.12 hiện có; `RUN_DEMO_WINDOWS.bat --port ...` mở server trên cổng thử riêng, năm route chính trả 200; đã dừng đúng tiến trình thử |
| Clone sạch và cài lại | Clone local sang thư mục tạm với `core.autocrlf=true`, tạo venv mới bằng `SETUP_WINDOWS.bat`, chỉ cài `requirements.txt`; `pip check`, `check_release.py` và toàn bộ 72 kiểm thử đều đạt trên Windows |

Bản clone sạch không chứa môi trường ảo hoặc cache từ thư mục gốc. Mã/manifests khóa vẫn khớp SHA-256 sau checkout Windows; ba báo cáo tiếp tục khớp đủ 14 tài sản. Đây là kiểm tra trên môi trường mới của cùng máy, không phải xác nhận đã thử một máy tính khác.

`check_release.py` và kiểm tra giao thức không chạy lại accuracy toàn bộ tập test, không huấn luyện và không ghi đè dữ liệu/kết quả đã khóa. Các kiểm thử có huấn luyện fixture rất nhỏ chỉ ghi trong thư mục tạm, không ghi đè checkpoint của ứng dụng.

Các tệp log và môi trường trong `results/history/`, `results/tested_environment.txt` giữ nguyên như hồ sơ cũ. Những số 67/68 kiểm thử trong tài liệu lịch sử không được đổi thành kết quả của lần chạy này.

## Giới hạn cần trình bày trung thực

- Test ảnh thực đã mở từ giai đoạn 4; kết quả giai đoạn 5 là đối chiếu hồi quy, không phải test mới chưa thấy.
- Kết quả phân loại 1.200 crop tổng hợp không phải accuracy tự tìm và phân loại vật trên ảnh thực.
- 20 probe hình học dùng tiêu chí riêng theo ca, không được chuyển 18/20 thành detection accuracy.
- Chưa giải quyết hình chạm/chồng; tập stress tổng hợp 12 cảnh có 0 TP / 1 FP / 17 FN.
- Dữ liệu thực nhỏ và chọn thuận tiện; softmax chưa hiệu chỉnh độ tin cậy. Chưa có kiểm chứng triển khai ngoài phạm vi demo.
- Lần rà soát phát hành này không kiểm tra trực quan lại Chrome/mobile. Các kiểm tra trình duyệt cũ được ghi rõ là bằng chứng lịch sử.
- Docker CLI có mặt nhưng Docker Engine chưa hoạt động ở lần kiểm tra; chưa build/chạy image trong lần này.
- Workflow Windows/Ubuntu đã được chuẩn bị. Trạng thái chạy thực tế được xem tại [GitHub Actions của CNN Lab](https://github.com/NTDung123/CNN-Lab/actions); kết quả local không được gọi thay là kết quả CI.

## Trước khi nộp và công khai

Giữ bản nộp source, checkpoint, dữ liệu cần thiết và báo cáo cùng phiên bản. Điền nhiệm vụ kỹ thuật và % đóng góp thực tế. Nhóm chưa chọn giấy phép chung cho mã tự phát triển; phần kế thừa và từng ảnh tiếp tục giữ phạm vi quyền tại nguồn đã ghi trong notices.

Repository đích: [NTDung123/CNN-Lab](https://github.com/NTDung123/CNN-Lab), chế độ **Public** theo lựa chọn của nhóm. Tên hiển thị trong README là CNN Lab; đường dẫn repository dùng dấu gạch nối thay khoảng trắng. Trạng thái nhánh/commit và workflow trên GitHub là căn cứ xác nhận bản đã công khai.
