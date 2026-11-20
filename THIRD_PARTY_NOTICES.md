# Nguồn kế thừa và tài sản bên thứ ba

Repository phục vụ bài tập lớn môn Xử lý ảnh của Nhóm 01, lớp INT13146-N09. Mã, trọng số, dữ liệu và ảnh trong repository không được mặc định xem là cùng một giấy phép.

## Dự án ban đầu

- Nguồn: [phanmanhcuongdev/identity_number](https://github.com/phanmanhcuongdev/identity_number).
- Phần kế thừa được giữ trong `reference_original/`: ANN NumPy, mã liên quan đến huấn luyện và checkpoint `model_infinity.npz`; một số đoạn đã được điều chỉnh để đối chiếu với dự án hiện tại.
- Mốc upstream được kiểm tra khi chuẩn bị phát hành: [`c7576655d4616f69a742fa9f5ffd6246b15c28a9`](https://github.com/phanmanhcuongdev/identity_number/commit/c7576655d4616f69a742fa9f5ffd6246b15c28a9). Đây là HEAD tại lần kiểm tra, **không khẳng định là commit đã clone ban đầu**; thư mục local hiện không giữ lịch sử Git của lần clone đó.
- Tại lần kiểm tra, upstream không có tệp LICENSE và GitHub không xác định giấy phép. Repository này không tự gán MIT hay giấy phép mở khác cho phần kế thừa đó. Việc công khai mã nguồn không tự xác định quyền cấp phép lại.

Luồng Flask/CNN hiện tại, các thuật toán xử lý ảnh, bộ dữ liệu hình học, phần bổ sung ảnh thực và kiểm thử được mô tả trong README cùng nhật ký kỹ thuật. `reference_v3/` và `reference_v4/` lưu các phiên bản để đối chiếu; không phải luồng suy luận chính. Không suy ra phần việc hoặc tỷ lệ đóng góp từng thành viên từ tên thư mục.

## MNIST

- Bộ dữ liệu MNIST do Yann LeCun, Corinna Cortes và Christopher J. C. Burges cung cấp.
- Bản `data/mnist.npz` lấy từ [bản phân phối Keras/TensorFlow](https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz); mô tả định dạng tại [Keras MNIST](https://keras.io/api/datasets/mnist/).
- URL, số mẫu và SHA-256 nằm trong [`data/source.json`](data/source.json). Các ví dụ `examples/digit_*.png` được trích từ tập test; chỉ số nằm trong `examples/mnist_sample_indices.json`.
- Việc lưu bản dữ liệu tại đây không chuyển quyền tác giả của dữ liệu sang nhóm, và không tự gán giấy phép mã nguồn của TensorFlow cho toàn bộ dữ liệu.

## Ảnh thực và các bản dẫn xuất

Nguồn, tác giả, liên kết giấy phép của **40 ảnh** được ghi theo từng tệp trong [`data/real/ATTRIBUTION.md`](data/real/ATTRIBUTION.md) và [`data/real/manifest.json`](data/real/manifest.json). Có ảnh CC BY, CC BY-SA, CC0, public domain và ảnh có tuyên bố không biết hạn chế bản quyền tại nguồn; không gọi chung toàn bộ ảnh là CC0.

Ảnh đã được giải mã, thu nhỏ, bỏ EXIF và lưu PNG. Ảnh nhãn trong `data/real/annotations/`, ảnh đối chiếu trong `results/real_*_comparison.png` và các vùng ảnh thực trong dữ liệu hybrid được dẫn xuất từ những nguồn này. Giữ ghi nhận tác giả, liên kết nguồn và điều kiện giấy phép tương ứng khi phân phối lại các bản dẫn xuất. Các lớp khung/nhãn do dự án bổ sung không thay thế quyền đối với ảnh gốc.

## Thư viện

PyTorch, NumPy, Flask, Pillow và các phụ thuộc giữ giấy phép riêng của từng dự án. Phiên bản trực tiếp nằm trong `requirements.txt`; repository không kèm hoặc cấp phép lại thư mục môi trường ảo.

## Giấy phép của bản tổng hợp

Nhóm chưa chọn giấy phép chung cho phần mã tự phát triển. Tệp này ghi nhận nguồn và phạm vi tài sản, không thay thế các giấy phép gốc và không cấp quyền rộng hơn các nguồn đã công bố.
