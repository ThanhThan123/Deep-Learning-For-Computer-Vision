"""
EX2 - Phân tích sâu sự khác nhau giữa Machine Learning và Deep Learning

Mục tiêu:
1. Hiểu Machine Learning và Deep Learning khác nhau ở đâu
2. Hiểu vì sao Deep Learning mạnh hơn trong nhiều bài toán ảnh
3. Phân biệt:
   - Machine Learning
   - Neural Network
   - Deep Learning
4. Biết khi nào nên dùng ML, khi nào nên dùng DL

Ghi chú:
- File này tập trung phân tích lý thuyết, không phải file code train model.
- Nội dung được viết lại dựa trên phần mở đầu của PPT khóa học.
"""


# ============================================================
# 1. MACHINE LEARNING LÀ GÌ?
# ============================================================

print("1. MACHINE LEARNING LÀ GÌ?")
print("""
Machine Learning là lĩnh vực giúp máy tính học từ dữ liệu để đưa ra dự đoán hoặc quyết định.

Thay vì lập trình cứng từng luật một, ta đưa dữ liệu vào,
mô hình sẽ học ra quy luật từ dữ liệu đó.

Ví dụ:
- Phân loại email spam / không spam
- Dự đoán giá nhà
- Phân loại ảnh mèo / chó
""")


# ============================================================
# 2. DEEP LEARNING LÀ GÌ?
# ============================================================

print("2. DEEP LEARNING LÀ GÌ?")
print("""
Deep Learning là một nhánh của Machine Learning.

Điểm khác biệt quan trọng:
- Deep Learning dùng mạng nơ-ron nhiều tầng (nhiều hidden layers)
- Mô hình có thể tự học đặc trưng từ dữ liệu đầu vào
- Sau đó dùng các đặc trưng đó để đưa ra dự đoán
""")


# ============================================================
# 3. ĐIỂM KHÁC NHAU CỐT LÕI NHẤT
# ============================================================

print("3. ĐIỂM KHÁC NHAU CỐT LÕI NHẤT")
print("""
Theo slide của khóa học, Machine Learning truyền thống thường có 2 bước tách rời:

Input -> Feature Extraction -> Classification -> Output

Trong khi đó Deep Learning thường đi theo hướng:

Input -> Feature Extraction + Classification -> Output

Tức là:
- Machine Learning: con người thường phải hỗ trợ nhiều ở bước trích xuất đặc trưng
- Deep Learning: mô hình tự học luôn đặc trưng và học luôn cách phân loại
""")


# ============================================================
# 4. MACHINE LEARNING TRUYỀN THỐNG HOẠT ĐỘNG RA SAO?
# ============================================================

print("4. MACHINE LEARNING TRUYỀN THỐNG HOẠT ĐỘNG RA SAO?")
print("""
Trong Machine Learning truyền thống, pipeline thường là:

Bước 1: Thu thập dữ liệu
Bước 2: Tiền xử lý dữ liệu
Bước 3: Trích xuất đặc trưng (feature extraction)
Bước 4: Đưa đặc trưng vào mô hình
Bước 5: Mô hình dự đoán kết quả

Ví dụ với ảnh ô tô:
- Ta có thể phải tự nghĩ ra đặc trưng như:
  + cạnh
  + góc
  + màu
  + texture
  + hình dạng
- Sau đó mới dùng mô hình như:
  + SVM
  + Logistic Regression
  + Random Forest
  + KNN
để phân loại
""")


# ============================================================
# 5. FEATURE EXTRACTION TRONG ML TẠI SAO QUAN TRỌNG?
# ============================================================

print("5. VÌ SAO FEATURE EXTRACTION TRONG ML LẠI QUAN TRỌNG?")
print("""
Với Machine Learning truyền thống, chất lượng feature ảnh hưởng cực mạnh đến kết quả.

Nếu chọn đặc trưng tốt:
- mô hình học tốt hơn
- dễ phân biệt các lớp hơn

Nếu chọn đặc trưng kém:
- mô hình sẽ khó học
- kết quả dễ sai

Nói cách khác:
Trong ML truyền thống, con người đóng vai trò rất lớn ở bước thiết kế đặc trưng.
""")


# ============================================================
# 6. DEEP LEARNING GIẢI QUYẾT VẤN ĐỀ ĐÓ NHƯ THẾ NÀO?
# ============================================================

print("6. DEEP LEARNING GIẢI QUYẾT VẤN ĐỀ ĐÓ NHƯ THẾ NÀO?")
print("""
Deep Learning giảm bớt gánh nặng thiết kế feature thủ công.

Thay vì:
- con người tự nghĩ đặc trưng

thì nay:
- mô hình tự học các đặc trưng từ dữ liệu

Đây là lý do Deep Learning đặc biệt mạnh trong:
- Computer Vision
- Speech
- NLP

vì các dạng dữ liệu này phức tạp, khó mô tả hết bằng feature thủ công.
""")


# ============================================================
# 7. SO SÁNH TRỰC QUAN ML VÀ DL
# ============================================================

print("7. SO SÁNH TRỰC QUAN ML VÀ DL")
print("""
Machine Learning truyền thống giống như:
- bạn tự chọn nguyên liệu
- tự sơ chế
- rồi mới đưa cho đầu bếp nấu

Deep Learning giống như:
- bạn đưa nguyên liệu thô vào
- hệ thống tự học cách sơ chế và tự học luôn cách nấu

Điểm mạnh của Deep Learning là tính end-to-end:
- học từ đầu vào đến đầu ra trong cùng một mô hình
""")


# ============================================================
# 8. OUTPUT CỦA DEEP LEARNING "NHIỀU THÔNG TIN HƠN" NGHĨA LÀ GÌ?
# ============================================================

print("8. OUTPUT CỦA DEEP LEARNING NHIỀU THÔNG TIN HƠN NGHĨA LÀ GÌ?")
print("""
Trong slide minh họa, Machine Learning cho output kiểu đơn giản:
- CAR / NOT CAR

Trong khi Deep Learning có thể cho output giàu thông tin hơn, ví dụ:
- CAR
- Color: red
- Make: Ford
- Model: Mustang

Điều này không có nghĩa mọi mô hình Deep Learning đều tự động cho ra đầy đủ thông tin như vậy.

Ý của slide là:
- Deep Learning có khả năng học biểu diễn dữ liệu phong phú hơn
- nên có thể mở rộng ra các bài toán phức tạp hơn
- ví dụ không chỉ phân loại một nhãn, mà còn suy ra thêm nhiều thuộc tính liên quan
""")


# ============================================================
# 9. NEURAL NETWORK KHÁC GÌ DEEP LEARNING?
# ============================================================

print("9. NEURAL NETWORK KHÁC GÌ DEEP LEARNING?")
print("""
Theo slide 'Neural Network vs Deep Learning':

- Neural Network:
  thường có ít hidden layers hơn

- Deep Learning:
  là Neural Network nhưng có nhiều hidden layers hơn

Nói ngắn gọn:
- Deep Learning là một dạng Neural Network sâu hơn
- không phải cứ có Neural Network là đã là Deep Learning theo nghĩa mạnh hiện nay
""")


# ============================================================
# 10. VÌ SAO "ĐỘ SÂU" QUAN TRỌNG?
# ============================================================

print("10. VÌ SAO ĐỘ SÂU QUAN TRỌNG?")
print("""
Khi mạng có nhiều hidden layers, mô hình có thể học đặc trưng theo từng mức:

Layer đầu:
- học đặc trưng đơn giản
- ví dụ cạnh, góc, đường nét

Layer giữa:
- học tổ hợp của các đặc trưng đơn giản
- ví dụ mắt, tai, bánh xe, cửa sổ

Layer sâu hơn:
- học ra đối tượng hoàn chỉnh
- ví dụ mặt người, con mèo, chiếc xe

Độ sâu giúp mô hình biểu diễn được các quan hệ phức tạp hơn.
""")


# ============================================================
# 11. KHI NÀO MACHINE LEARNING VẪN HỮU ÍCH?
# ============================================================

print("11. KHI NÀO MACHINE LEARNING VẪN HỮU ÍCH?")
print("""
Machine Learning truyền thống vẫn rất hữu ích khi:

1. Dữ liệu không quá lớn
2. Bài toán không quá phức tạp
3. Feature đã khá rõ ràng
4. Cần mô hình nhẹ, dễ giải thích
5. Tài nguyên tính toán hạn chế

Ví dụ:
- dữ liệu bảng (tabular data)
- bài toán dự đoán cơ bản
- số lượng mẫu không quá lớn
""")


# ============================================================
# 12. KHI NÀO DEEP LEARNING PHÙ HỢP HƠN?
# ============================================================

print("12. KHI NÀO DEEP LEARNING PHÙ HỢP HƠN?")
print("""
Deep Learning phù hợp hơn khi:

1. Dữ liệu phức tạp
   - ảnh
   - âm thanh
   - văn bản
   - video

2. Có nhiều dữ liệu

3. Cần mô hình tự học feature mạnh

4. Bài toán có độ khó cao:
   - Image Classification
   - Object Detection
   - Segmentation
   - NLP
   - Speech Recognition
""")


# ============================================================
# 13. ƯU ĐIỂM VÀ NHƯỢC ĐIỂM
# ============================================================

print("13. ƯU ĐIỂM VÀ NHƯỢC ĐIỂM")

print("""
Machine Learning truyền thống
Ưu điểm:
- nhẹ hơn
- nhanh hơn trên dữ liệu nhỏ
- dễ giải thích hơn trong nhiều trường hợp
- ít tốn tài nguyên hơn

Nhược điểm:
- phụ thuộc nhiều vào feature engineering
- khó xử lý dữ liệu phức tạp như ảnh, tiếng nói, ngôn ngữ
""")

print("""
Deep Learning
Ưu điểm:
- học đặc trưng tự động
- mạnh với dữ liệu phức tạp
- hiệu quả cao trong Computer Vision, NLP, Speech
- dễ mở rộng cho các bài toán lớn

Nhược điểm:
- cần nhiều dữ liệu hơn
- cần nhiều tài nguyên tính toán hơn
- huấn luyện lâu hơn
- thường khó giải thích hơn
""")


# ============================================================
# 14. LIÊN HỆ TRỰC TIẾP VỚI COMPUTER VISION
# ============================================================

print("14. LIÊN HỆ TRỰC TIẾP VỚI COMPUTER VISION")
print("""
Trong Computer Vision, khác biệt này càng rõ.

Nếu dùng Machine Learning truyền thống cho ảnh:
- thường phải tự trích xuất đặc trưng từ ảnh
- rồi mới phân loại

Nếu dùng Deep Learning:
- ảnh có thể đi trực tiếp vào mạng
- mạng tự học feature
- sau đó tự dự đoán class

Đó là lý do Deep Learning trở thành nền tảng của:
- Image Classification
- Object Detection
- Segmentation
""")


# ============================================================
# 15. KẾT LUẬN QUAN TRỌNG NHẤT CẦN NHỚ
# ============================================================

print("15. KẾT LUẬN QUAN TRỌNG NHẤT CẦN NHỚ")
print("""
1. Deep Learning là một nhánh của Machine Learning
2. Điểm khác nhau lớn nhất:
   - ML truyền thống tách feature extraction và classification
   - DL gộp hai bước đó thành một mô hình end-to-end
3. Neural Network chưa chắc đã là Deep Learning
4. Deep Learning mạnh hơn khi dữ liệu phức tạp và có nhiều tầng học biểu diễn
5. Trong Computer Vision, Deep Learning thường hiệu quả hơn ML truyền thống
""")