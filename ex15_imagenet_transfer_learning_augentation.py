"""
EX15 - ImageNet, Transfer Learning và Data Augmentation

Mục tiêu:
1. Hiểu vai trò của ImageNet / ILSVRC trong Deep Learning
2. Hiểu sự khác biệt giữa LeNet và AlexNet
3. Hiểu Transfer Learning là gì
4. Hiểu 2 cách dùng transfer learning:
   - fixed feature extractor
   - fine-tuning
5. Hiểu data augmentation là gì và vì sao cần
6. Hiểu các phép augmentation hình học cơ bản

Tài liệu gốc:
- PPT Deep Learning / Computer Vision
"""


# ============================================================
# 1. TỔNG QUAN
# ============================================================

print("1. TỔNG QUAN")
print("""
Sau khi đã hiểu:
- CNN hoạt động thế nào
- receptive field
- optimization
- backpropagation
- training stability

thì bước tiếp theo trong classification thực chiến là:

1. ImageNet / ILSVRC
2. Transfer Learning
3. Data Augmentation

Đây là các khái niệm cực kỳ quan trọng trong Computer Vision hiện đại.
""")


# ============================================================
# 2. IMAGENET LÀ GÌ?
# ============================================================

print("\n2. IMAGENET LÀ GÌ?")
print("""
ImageNet là một dataset ảnh rất lớn và rất nổi tiếng trong Computer Vision.

Ý nghĩa của ImageNet:
- cung cấp dữ liệu lớn để train CNN mạnh
- giúp tạo ra các mô hình pretrained nổi tiếng
- là nền tảng của nhiều kỹ thuật transfer learning sau này

Khi nhắc đến ImageNet,
người ta thường nghĩ đến:
- dữ liệu lớn
- nhiều class
- benchmark rất quan trọng
""")


# ============================================================
# 3. ILSVRC LÀ GÌ?
# ============================================================

print("\n3. ILSVRC LÀ GÌ?")
print("""
ILSVRC là viết tắt của:
ImageNet Large Scale Visual Recognition Challenge

Đây là cuộc thi / benchmark rất quan trọng trong lịch sử Deep Learning.

Vai trò:
- so sánh các mô hình nhận diện ảnh
- thúc đẩy phát triển CNN hiện đại
- giúp nhiều kiến trúc mạnh ra đời
""")


# ============================================================
# 4. VÌ SAO ILSVRC QUAN TRỌNG?
# ============================================================

print("\n4. VÌ SAO ILSVRC QUAN TRỌNG?")
print("""
ILSVRC quan trọng vì:
- nó tạo ra một chuẩn đánh giá lớn cho classification
- các mô hình phải cạnh tranh trên dữ liệu thật sự lớn
- nhờ đó Deep Learning phát triển rất nhanh

Nói đơn giản:
ILSVRC là một cột mốc giúp Deep Learning bùng nổ trong Computer Vision.
""")


# ============================================================
# 5. LENET VS ALEXNET
# ============================================================

print("\n5. LENET VS ALEXNET")
print("""
Trong PPT có phần LeNet vs AlexNet.

Ý nghĩa của việc so sánh này là:
- cho thấy CNN đã phát triển từ mạng nhỏ, đơn giản
- sang mạng sâu hơn, mạnh hơn và thực tế hơn

LeNet:
- sớm hơn
- đơn giản hơn
- thường gắn với bài toán nhỏ như chữ số viết tay

AlexNet:
- sâu hơn
- mạnh hơn
- nổi tiếng nhờ thành công trên ImageNet / ILSVRC

Đây là bước tiến rất lớn của CNN trong lịch sử Deep Learning.
""")


# ============================================================
# 6. TỪ IMAGENET ĐẾN PRETRAINED MODEL
# ============================================================

print("\n6. TỪ IMAGENET ĐẾN PRETRAINED MODEL")
print("""
Khi một mô hình được train trên ImageNet:
- nó đã học được rất nhiều feature hữu ích từ ảnh

Ví dụ:
- edge
- texture
- shape
- pattern trung gian

Nhờ đó:
- ta có thể tái sử dụng mô hình này cho bài toán mới

Đó chính là cơ sở của transfer learning.
""")


# ============================================================
# 7. TRANSFER LEARNING LÀ GÌ?
# ============================================================

print("\n7. TRANSFER LEARNING LÀ GÌ?")
print("""
Transfer Learning là cách:
- lấy kiến thức từ một model đã train trước
- rồi áp dụng cho bài toán mới

Thay vì train từ đầu:
- ta dùng model pretrained
- tận dụng feature mà nó đã học

Điều này đặc biệt hữu ích khi:
- dữ liệu mới không quá lớn
- không có nhiều tài nguyên tính toán
""")


# ============================================================
# 8. TẠI SAO TRANSFER LEARNING MẠNH?
# ============================================================

print("\n8. TẠI SAO TRANSFER LEARNING MẠNH?")
print("""
Vì model pretrained trên ImageNet đã học được feature rất mạnh.

Các layer đầu của CNN thường đã học được:
- cạnh
- texture
- pattern cơ bản

Những feature này thường vẫn hữu ích cho nhiều bài toán ảnh khác.

Do đó:
- không cần học lại từ đầu mọi thứ
- model học nhanh hơn
- cần ít dữ liệu hơn
""")


# ============================================================
# 9. CÁCH 1: USE CNN AS FIXED FEATURE EXTRACTOR
# ============================================================

print("\n9. USE CNN AS FIXED FEATURE EXTRACTOR")
print("""
Theo PPT, cách đầu tiên là:

Use CNN as fixed feature extractor

Ý tưởng:
- giữ nguyên phần lớn CNN pretrained
- không train lại backbone chính
- chỉ dùng nó để trích xuất feature
- sau đó train một classifier mới ở phía sau

Ưu điểm:
- nhanh
- ít tốn tài nguyên
- phù hợp khi dữ liệu mới ít
""")


# ============================================================
# 10. CÁCH 2: FINE-TUNE CNN
# ============================================================

print("\n10. FINE-TUNE CNN")
print("""
Theo PPT, cách thứ hai là:

Fine-tune CNN

Ý tưởng:
- bắt đầu từ model pretrained
- nhưng cho phép update một phần hoặc toàn bộ weights
- để model thích nghi tốt hơn với bài toán mới

Ưu điểm:
- thường đạt kết quả tốt hơn fixed feature extractor
- nhất là khi dữ liệu mới đủ lớn hoặc hơi khác ImageNet

Nhược điểm:
- tốn tài nguyên hơn
- dễ overfit hơn nếu dữ liệu ít
""")


# ============================================================
# 11. HOW TO CHOOSE APPROACH?
# ============================================================

print("\n11. HOW TO CHOOSE APPROACH?")
print("""
PPT có phần:
Transfer Learning: How to choose approach

Cách chọn thường dựa vào:
- kích thước dataset mới
- độ giống giữa dữ liệu mới và ImageNet
- tài nguyên tính toán

Gợi ý thực tế:
- dữ liệu ít -> ưu tiên fixed feature extractor
- dữ liệu nhiều hơn / khác bài toán hơn -> cân nhắc fine-tuning
""")


# ============================================================
# 12. VÍ DỤ MINH HỌA TRANSFER LEARNING
# ============================================================

print("\n12. VÍ DỤ MINH HỌA TRANSFER LEARNING")
print("""
Ví dụ:
- model pretrained trên ImageNet
- bài toán mới: phân loại chó mèo

Thay vì train từ đầu:
- dùng CNN pretrained
- thay lớp cuối thành 2 class
- train phần cuối hoặc fine-tune thêm

=> tiết kiệm rất nhiều thời gian và dữ liệu
""")


# ============================================================
# 13. DATA AUGMENTATION LÀ GÌ?
# ============================================================

print("\n13. DATA AUGMENTATION LÀ GÌ?")
print("""
Data Augmentation là kỹ thuật tạo thêm dữ liệu huấn luyện
bằng cách biến đổi dữ liệu gốc.

Ý tưởng:
- không làm thay đổi bản chất nhãn
- nhưng tạo ra nhiều phiên bản khác nhau của cùng một ảnh

Mục tiêu:
- tăng đa dạng dữ liệu
- giúp model tổng quát tốt hơn
- giảm overfitting
""")


# ============================================================
# 14. POSITION MANIPULATION / GEOMETRIC TRANSFORMATIONS
# ============================================================

print("\n14. POSITION MANIPULATION / GEOMETRIC TRANSFORMATIONS")
print("""
Theo PPT:
Data augmentation: Position manipulation
Another name: Geometric transformations

Nghĩa là:
- thay đổi vị trí / hình học của ảnh
- nhưng vẫn giữ nội dung chính

Ví dụ:
- dịch chuyển
- lật ảnh
- xoay nhẹ
- crop
- scale
""")


# ============================================================
# 15. VÌ SAO GEOMETRIC AUGMENTATION HỮU ÍCH?
# ============================================================

print("\n15. VÌ SAO GEOMETRIC AUGMENTATION HỮU ÍCH?")
print("""
Vì ngoài đời thực:
- object có thể lệch vị trí
- góc chụp khác nhau
- kích thước khác nhau
- bố cục khác nhau

Nếu model chỉ học trên ảnh "quá chuẩn":
- nó sẽ kém linh hoạt

Augmentation giúp model:
- quen với nhiều biến thể hơn
- tổng quát tốt hơn trên dữ liệu thật
""")


# ============================================================
# 16. VÍ DỤ CÁC PHÉP AUGMENTATION PHỔ BIẾN
# ============================================================

print("\n16. VÍ DỤ CÁC PHÉP AUGMENTATION PHỔ BIẾN")
print("""
Các phép augmentation phổ biến:

1. Horizontal flip
2. Random crop
3. Rotation
4. Translation
5. Scaling / Resize
6. Color jitter (nếu phù hợp)

Lưu ý:
- phải chọn augmentation phù hợp với bài toán
- không phải augmentation nào cũng hợp lý
""")


# ============================================================
# 17. AUGMENTATION KHÔNG PHẢI LÚC NÀO CŨNG TỐT
# ============================================================

print("\n17. AUGMENTATION KHÔNG PHẢI LÚC NÀO CŨNG TỐT")
print("""
Ví dụ:
- nếu bài toán nhận diện chữ số viết tay
- xoay quá mạnh có thể làm đổi nghĩa ảnh

Hoặc:
- trong y khoa, một số phép biến đổi có thể không phù hợp về mặt lâm sàng

Vì vậy:
augmentation phải hợp với bản chất dữ liệu và bài toán.
""")


# ============================================================
# 18. TRANSFER LEARNING + AUGMENTATION = CỰC MẠNH
# ============================================================

print("\n18. TRANSFER LEARNING + AUGMENTATION")
print("""
Trong thực tế, hai kỹ thuật này thường đi cùng nhau:

- Transfer Learning:
  tận dụng tri thức từ model pretrained

- Data Augmentation:
  làm giàu dữ liệu mới

Kết hợp lại:
- model học nhanh hơn
- generalize tốt hơn
- đạt kết quả mạnh hơn khi dữ liệu ít
""")


# ============================================================
# 19. VÍ DỤ MINH HỌA NHỎ BẰNG TORCHVISION
# ============================================================

print("\n19. VÍ DỤ MINH HỌA NHỎ")

from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

print("""
Ví dụ transform trong PyTorch:
- resize
- horizontal flip
- rotation
- convert to tensor

Đây là cách augmentation thường được gắn vào pipeline dữ liệu.
""")


# ============================================================
# 20. LIÊN HỆ VỚI PHẦN CLASSIFICATION THỰC CHIẾN
# ============================================================

print("\n20. LIÊN HỆ VỚI CLASSIFICATION THỰC CHIẾN")
print("""
Nếu chỉ học CNN cơ bản thì vẫn chưa đủ cho bài toán thật.

Trong thực tế classification:
- pretrained model rất quan trọng
- transfer learning gần như là kỹ thuật tiêu chuẩn
- augmentation giúp model bền vững hơn

Đây là lý do PPT đưa cụm này vào trước khi chuyển sang object detection.
""")


# ============================================================
# 21. NHỮNG NHẦM LẪN HAY GẶP
# ============================================================

print("\n21. NHỮNG NHẦM LẪN HAY GẶP")
print("""
Nhầm lẫn 1:
Transfer learning là copy nguyên model và dùng y hệt
-> Sai
Ta thường phải thay hoặc chỉnh phần đầu ra

Nhầm lẫn 2:
Fine-tuning luôn tốt hơn fixed feature extractor
-> Không hẳn
Tùy dữ liệu và tài nguyên

Nhầm lẫn 3:
Càng augment nhiều càng tốt
-> Sai
Augmentation phải hợp lý với bài toán

Nhầm lẫn 4:
ImageNet chỉ là dataset lớn, không quá quan trọng
-> Sai
ImageNet và ILSVRC có vai trò lịch sử rất lớn với CNN hiện đại
""")


# ============================================================
# 22. KẾT NỐI SANG PHẦN TIẾP THEO
# ============================================================

print("\n22. KẾT NỐI SANG PHẦN TIẾP THEO")
print("""
Sau khi hoàn thành cụm classification nâng cao này,
PPT mới hợp lý để chuyển sang:

- Object Detection
- Segmentation
- GANs

Vì lúc này ta đã có nền tảng đủ mạnh về classification.
""")


# ============================================================
# 23. TỔNG KẾT
# ============================================================

print("\n23. TỔNG KẾT")
print("""
1. ImageNet / ILSVRC là cột mốc rất quan trọng của Computer Vision hiện đại
2. AlexNet là một bước tiến lớn so với LeNet trong bối cảnh classification quy mô lớn
3. Transfer Learning giúp tận dụng model pretrained
4. Có 2 cách phổ biến:
   - fixed feature extractor
   - fine-tuning
5. Data augmentation giúp tăng độ đa dạng dữ liệu và giảm overfitting
6. Đây là cụm kiến thức classification thực chiến rất quan trọng
""")