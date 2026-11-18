# Báo cáo Lab 2: Đặc trưng tiếng nói và nhận dạng từ bằng DTW

## 1. Mục tiêu

Lab này khảo sát đặc trưng tiếng nói trong miền thời gian và miền tần số, phát hiện điểm đầu/cuối phát âm, trích MFCC, tự cài đặt Dynamic Time Warping (DTW), và xây dựng bộ nhận dạng từ theo phương pháp nearest-template. Bảy lệnh tiếng Việt được xét là **bật, tắt, lên, xuống, trái, phải, dừng**.

## 2. Dữ liệu và cách đánh giá

Bộ dữ liệu có 35 tệp WAV (7 từ × 5 tệp), mono, 16 kHz, PCM 16-bit. Notebook chia tệp 01–03 làm template và 04–05 làm test. Kiểm tra MD5 phát hiện năm tệp trong mỗi lớp trùng byte: mỗi từ thực chất chỉ có **một bản ghi duy nhất** được sao chép nhiều lần. Vì vậy hai tệp test của mỗi từ cũng đã xuất hiện nguyên vẹn trong template. Đây là **rò rỉ dữ liệu** và khiến accuracy trên dữ liệu gốc không phải ước lượng hợp lệ khả năng tổng quát hóa.

Để quan sát cách DTW phản ứng với biến thiên, notebook tạo năm biến thể nhân tạo cho mỗi từ từ bản ghi gốc, sử dụng seed 457: tốc độ 0.80–1.25×, pitch ±1.5 bán âm, biến đổi phổ, âm lượng khác nhau và nhiễu trắng SNR 20–30 dB. Các biến thể này vẫn bắt nguồn từ cùng một lần nói; kết quả của chúng chỉ có tính minh họa, không thay thế các lần ghi độc lập.

## 3. Phương pháp

### 3.1. Đặc trưng miền thời gian và endpoint detection

Tín hiệu được chia frame 25 ms (400 mẫu), bước nhảy 10 ms (160 mẫu), dùng cửa sổ Hamming. Tính năng lượng, log-energy, RMS, magnitude, zero-crossing rate (ZCR) và autocorrelation trên từng frame. Endpoint detection dùng ngưỡng TOP_DB = 35 dB dưới đỉnh, sàn nền nhiễu + 10 dB, mở rộng biên khi ZCR > 0.25 (tối đa 10 frame), cùng margin 5 frame (50 ms).

Log-energy tách vùng im lặng khỏi vùng có tiếng tốt: silence khoảng −58 đến −64 dB, vùng hữu thanh khoảng +10 đến +13 dB. Tuy nhiên, phụ âm vô thanh/xát có năng lượng thấp nhưng ZCR cao (khoảng 0.45–0.70), vì vậy chỉ dùng năng lượng có thể cắt mất phụ âm. Kết hợp ZCR và margin giảm phần cắt tối đa từ 110 ms xuống không quá 40 ms theo kiểm tra trong notebook.

### 3.2. MFCC

Pipeline MFCC dùng pre-emphasis α = 0.97, cửa sổ Hamming, NFFT = 512, 24 bộ lọc Mel và 13 hệ số MFCC; áp dụng Cepstral Mean Normalization (CMN) theo từng utterance. Mỗi phát âm trở thành ma trận T × 13, trong đó T phụ thuộc độ dài sau khi cắt. MFCC có số chiều cố định, còn DTW xử lý khác biệt về số frame.

### 3.3. DTW và nhận dạng nearest-template

Khoảng cách cục bộ là Euclidean giữa các vector MFCC. Bảng quy hoạch động cho phép bước chéo, bước ngang và bước dọc; đường đi được truy vết ngược và tổng chi phí chia cho độ dài đường đi |P|. Khoảng cách của một từ là giá trị nhỏ nhất trong các template của lớp đó; nhãn có khoảng cách nhỏ nhất được dự đoán.

Triển khai được kiểm tra bằng DTW(X, X) = 0, so sánh local distance với `scipy.cdist`, và đối chiếu tổng chi phí với `librosa.sequence.dtw` (3299.8539 trong kiểm tra mẫu).

## 4. Kết quả

### 4.1. Chất lượng dữ liệu và tín hiệu

Các tệp gốc có peak không quá 0.59 và không có clipping. Im lặng đầu khoảng 0.09–0.19 s, im lặng cuối khoảng 0.19–0.25 s. Một ví dụ autocorrelation trên frame hữu thanh của từ “lên” có đỉnh tại k = 68 mẫu, tương ứng F0 xấp xỉ 235 Hz.

Sau endpoint detection, thời lượng bảy tệp mẫu `_01` lần lượt là:

| Từ | Trước trim (s) | Sau trim (s) |
|---|---:|---:|
| bật | 0.648 | 0.395 |
| tắt | 0.600 | 0.325 |
| lên | 0.768 | 0.585 |
| xuống | 0.864 | 0.685 |
| trái | 0.864 | 0.655 |
| phải | 0.888 | 0.675 |
| dừng | 0.744 | 0.585 |

### 4.2. Kiểm tra DTW

| So sánh | DTW chuẩn hóa | Độ dài đường đi |
|---|---:|---:|
| Hai bản sao cùng từ “tắt” trong dữ liệu gốc | 0.00 | 30 |
| Hai biến thể tăng cường của “tắt” | 16.62 | 31 |
| “tắt” và “xuống” | 33.12 | 58 |

Khoảng cách cùng từ tăng khi có biến thiên tốc độ/pitch; khoảng cách giữa hai từ khác nhau trong ví dụ lớn hơn gần hai lần. Đường đi DTW co giãn thời gian để ghép các đoạn phát âm có tốc độ khác nhau.

### 4.3. Nhận dạng và ablation

Baseline dùng trim, 13 MFCC và ba template mỗi từ đạt 14/14 trên cả dữ liệu gốc lẫn tập biến thể. Con số 100% trên dữ liệu gốc là kết quả của leakage. Con số 100% trên dữ liệu biến thể cũng không thể xem là hiệu năng ngoài thực tế vì cả train và test đều được sinh từ cùng một bản ghi gốc cho mỗi từ. Confusion matrix tương ứng là đường chéo.

Các thí nghiệm E1–E4 trong notebook cho kết quả:

| Thí nghiệm | Kết quả chính trên bộ biến thể |
|---|---|
| E1: bỏ endpoint trim | Accuracy vẫn 100%; margin top-2 trừ top-1 giảm 6.25 → 5.47. Silence làm khoảng cách các lớp gần nhau hơn. |
| E2: thêm delta MFCC | Accuracy vẫn 100%; margin tăng nhẹ 6.25 → 6.52. |
| E3: một thay vì ba template/lớp | Accuracy giảm 100% → 92.9%; `trai_04` bị nhận thành `phai`. Ba template giúp bao phủ biến thiên tốc độ/phát âm tốt hơn. |
| E4: cùng người nói so với khác người nói | Không thực hiện được vì chỉ có dữ liệu của một người nói. |

Cặp gần nhất theo ma trận khoảng cách trung bình là “bật”–“tắt” (xấp xỉ 20.4; cùng từ xấp xỉ 15.3). Một nhầm lẫn quan sát được ở E3 là “trái” thành “phải”; hai từ cùng vần /ai/. Đây là các giả thuyết từ bộ dữ liệu nhỏ và cần kiểm chứng trên các bản ghi độc lập.

## 5. Trả lời câu hỏi báo cáo

1. **Vì sao không dùng waveform làm template?** Độ dài, pha, biên độ, pitch và thời điểm phát âm thay đổi giữa các lần nói; so sánh từng mẫu waveform rất nhạy với các thay đổi đó. MFCC biểu diễn phổ theo frame, còn DTW căn chỉnh khác biệt thời gian.
2. **Energy và ZCR có vai trò gì trong endpoint detection?** Energy tách im lặng và tiếng hữu thanh; ZCR cao giúp nhận ra phụ âm vô thanh/xát năng lượng thấp mà ngưỡng năng lượng có thể bỏ sót.
3. **Vì sao các bộ lọc Mel rộng dần theo Hz?** Thang Mel mô phỏng độ phân giải thính giác: mịn hơn ở tần số thấp, thô hơn ở tần số cao. Do đó các bộ lọc cách đều trên Mel rộng dần khi biểu diễn trên Hz.
4. **Vai trò của log và DCT trong MFCC?** Log nén dải động và biến tác động nhân của phổ/kênh gần thành cộng; DCT giảm tương quan giữa log-energy các dải Mel và tạo hệ số cepstral.
5. **Ý nghĩa các bước ngang, dọc, chéo trong DTW?** Chéo ghép hai frame cùng tiến; ngang hoặc dọc ghép một frame với nhiều frame liên tiếp, biểu diễn việc một chuỗi bị kéo giãn hoặc nén thời gian.
6. **Vì sao chuẩn hóa chi phí theo độ dài đường đi?** Tổng chi phí tăng theo số cặp frame. Chia cho |P| tạo chi phí trung bình mỗi cặp và giúp so sánh utterance có độ dài khác nhau công bằng hơn.
7. **Vì sao MFCC của cùng từ khác nhau giữa các lần nói?** Tốc độ, pitch, cường độ, ngữ điệu, đồng phát âm, khoảng cách/kênh microphone, nhiễu và endpoint đều làm thay đổi phổ.
8. **Cặp từ nào dễ nhầm?** Ma trận khoảng cách gợi ý “bật”–“tắt”; E3 cũng nhầm “trái” thành “phải”. Cần thêm nhiều lần ghi để xác nhận các cặp này.
9. **Hạn chế với người nói mới là gì?** Nearest-template chỉ lưu mẫu đã ghi, không mô hình hóa biến thiên giữa người nói. Thêm người/từ làm tăng số phép so DTW; cần dữ liệu đa người nói và mô hình thống kê/khả năng thích nghi để tổng quát hóa tốt hơn.

## 6. Kết luận và việc cần làm tiếp

Notebook triển khai chuỗi endpoint detection → MFCC → DTW tự cài → nearest-template → đánh giá, và có kiểm tra chéo kết quả DTW. Tuy nhiên, dữ liệu hiện tại chỉ có một lần phát âm duy nhất cho mỗi từ, bị sao chép thành năm tệp. Vì vậy accuracy 100% không phải bằng chứng nhận dạng tốt. Cần ghi âm ít nhất năm lần độc lập cho mỗi từ; nên có thêm người nói thứ hai để thực hiện E4. Sau khi thay dữ liệu, chạy lại notebook và cập nhật các bảng/nhận xét. Chưa có nhận xét nghe thủ công cho tín hiệu đã trim; cần nghe các tệp đầu ra để xác nhận phụ âm không bị cắt.

## 7. Tệp kết quả

- Notebook và mã thí nghiệm: [`Lab2_Notebook.ipynb`](Lab2_Notebook.ipynb)
- Kết quả test dữ liệu hiện có: [`results.csv`](results.csv)
- Kết quả test trên biến thể tăng cường: [`results_augmented.csv`](results_augmented.csv)
- Dữ liệu WAV: [`audio_dataset/`](audio_dataset/)
