"""
EX14 - Training Stability trong Deep Learning

Mục tiêu:
1. Hiểu vanishing gradient là gì
2. Hiểu exploding gradient là gì
3. Hiểu tại sao chúng xảy ra trong mạng sâu
4. Hiểu weight initialization
5. Hiểu gradient clipping
6. Hiểu cách làm training ổn định hơn

Tài liệu gốc:
- PPT Deep Learning / Computer Vision
"""


# ============================================================
# 1. TỔNG QUAN VẤN ĐỀ
# ============================================================

print("1. TỔNG QUAN")
print("""
Khi train neural network sâu:

- gradient có thể trở nên rất nhỏ (vanishing)
- hoặc rất lớn (exploding)

=> dẫn đến:
- model không học được
- hoặc học rất không ổn định

Đây là một trong những vấn đề lớn nhất trong Deep Learning.
""")


# ============================================================
# 2. VANISHING GRADIENT LÀ GÌ?
# ============================================================

print("\n2. VANISHING GRADIENT")
print("""
Vanishing gradient xảy ra khi:

- gradient trở nên rất nhỏ (gần 0)
- trong quá trình backpropagation

Kết quả:
- các layer đầu gần input gần như không được update
- model học rất chậm hoặc không học được

Đặc biệt xảy ra khi:
- mạng rất sâu
- dùng activation như sigmoid / tanh
""")


# ============================================================
# 3. VÌ SAO VANISHING GRADIENT XẢY RA?
# ============================================================

print("\n3. NGUYÊN NHÂN VANISHING GRADIENT")
print("""
Backpropagation sử dụng chain rule:

gradient = gradient_layer_n * gradient_layer_(n-1) * ...

Nếu mỗi gradient < 1:
- nhân nhiều lần -> rất nhỏ

Ví dụ:
0.5 * 0.5 * 0.5 * 0.5 = 0.0625

=> càng nhiều layer, gradient càng nhỏ

=> vanishing gradient
""")


# ============================================================
# 4. EXPLODING GRADIENT LÀ GÌ?
# ============================================================

print("\n4. EXPLODING GRADIENT")
print("""
Exploding gradient xảy ra khi:

- gradient trở nên rất lớn

Kết quả:
- weight update rất lớn
- model không ổn định
- loss có thể diverge (tăng vô hạn)

Đây là vấn đề ngược lại với vanishing gradient.
""")


# ============================================================
# 5. VÌ SAO EXPLODING GRADIENT XẢY RA?
# ============================================================

print("\n5. NGUYÊN NHÂN EXPLODING GRADIENT")
print("""
Tương tự chain rule:

Nếu mỗi gradient > 1:
- nhân nhiều lần -> rất lớn

Ví dụ:
2 * 2 * 2 * 2 = 16

=> gradient bùng nổ

=> weight update cực lớn -> model không ổn định
""")


# ============================================================
# 6. MINH HỌA NHỎ
# ============================================================

print("\n6. MINH HỌA VANISHING VS EXPLODING")

# vanishing
grad = 0.5
result = grad
for _ in range(5):
    result *= grad
print("Vanishing example:", result)

# exploding
grad = 2
result = grad
for _ in range(5):
    result *= grad
print("Exploding example:", result)


# ============================================================
# 7. HẬU QUẢ CỦA HAI VẤN ĐỀ NÀY
# ============================================================

print("\n7. HẬU QUẢ")
print("""
Vanishing gradient:
- model không học được layer đầu
- accuracy thấp

Exploding gradient:
- training không ổn định
- loss dao động mạnh hoặc diverge

=> cả hai đều rất nguy hiểm
""")


# ============================================================
# 8. GIẢI PHÁP - ACTIVATION FUNCTION
# ============================================================

print("\n8. GIẢI PHÁP - ACTIVATION FUNCTION")
print("""
Một giải pháp quan trọng:

- dùng ReLU thay vì sigmoid

Vì:
- ReLU không làm gradient nhỏ dần như sigmoid
- giúp giảm vanishing gradient

Đây là lý do ReLU phổ biến trong CNN.
""")


# ============================================================
# 9. WEIGHT INITIALIZATION LÀ GÌ?
# ============================================================

print("\n9. WEIGHT INITIALIZATION")
print("""
Weight initialization là cách khởi tạo giá trị ban đầu của weights.

Nếu khởi tạo sai:
- gradient có thể vanish hoặc explode ngay từ đầu

=> ảnh hưởng rất lớn đến training
""")


# ============================================================
# 10. CÁC CÁCH KHỞI TẠO CƠ BẢN
# ============================================================

print("\n10. CÁC CÁCH KHỞI TẠO")

import numpy as np

# zero initialization
W_zero = np.zeros((2, 2))

# random initialization
W_random = np.random.randn(2, 2)

print("Zero init:\n", W_zero)
print("Random init:\n", W_random)

print("""
Zero initialization:
- tất cả weight = 0
- KHÔNG dùng cho neural network

Random initialization:
- phổ biến
- giúp phá symmetry
""")


# ============================================================
# 11. VẤN ĐỀ VỚI ZERO INITIALIZATION
# ============================================================

print("\n11. ZERO INITIALIZATION VẤN ĐỀ")
print("""
Nếu tất cả weight = 0:

- các neuron giống nhau hoàn toàn
- gradient giống nhau
- model không học được gì

=> phải dùng random initialization
""")


# ============================================================
# 12. KHỞI TẠO TỐT HƠN (XAVIER / GLOROT)
# ============================================================

print("\n12. XAVIER / GLOROT INITIALIZATION")
print("""
PPT đề cập các phương pháp tốt hơn:

- Xavier / Glorot initialization

Ý tưởng:
- giữ variance ổn định qua các layer
- giúp gradient không quá lớn hoặc quá nhỏ

=> training ổn định hơn
""")


# ============================================================
# 13. GRADIENT CLIPPING LÀ GÌ?
# ============================================================

print("\n13. GRADIENT CLIPPING")
print("""
Gradient clipping là kỹ thuật:

- giới hạn độ lớn của gradient

Nếu gradient quá lớn:
- ta "cắt" nó lại

=> tránh exploding gradient
""")


# ============================================================
# 14. VÍ DỤ GRADIENT CLIPPING
# ============================================================

print("\n14. VÍ DỤ GRADIENT CLIPPING")

grad = np.array([10.0, -15.0, 5.0])

threshold = 5.0
grad_clipped = np.clip(grad, -threshold, threshold)

print("Original grad:", grad)
print("Clipped grad:", grad_clipped)


# ============================================================
# 15. TẠI SAO GRADIENT CLIPPING HIỆU QUẢ?
# ============================================================

print("\n15. TẠI SAO CLIPPING HIỆU QUẢ?")
print("""
Vì:
- tránh update quá lớn
- giữ training ổn định

Đặc biệt hữu ích khi:
- model sâu
- RNN
- hoặc loss surface phức tạp
""")


# ============================================================
# 16. CÁC GIẢI PHÁP KHÁC (THEO PPT)
# ============================================================

print("\n16. CÁC GIẢI PHÁP KHÁC")
print("""
PPT còn liệt kê:

- proper activation function
- batch normalization
- regularization

=> tất cả giúp training ổn định hơn
""")


# ============================================================
# 17. LIÊN HỆ VỚI EX TRƯỚC
# ============================================================

print("\n17. LIÊN HỆ")
print("""
Ex7:
- gradient descent

Ex8:
- backpropagation

Ex13:
- CNN architecture

=> Ex14:
- giải quyết vấn đề khi train mạng sâu
""")


# ============================================================
# 18. NHỮNG NHẦM LẪN HAY GẶP
# ============================================================

print("\n18. NHỮNG NHẦM LẪN")
print("""
Nhầm lẫn 1:
Gradient nhỏ là tốt
-> Sai

Nhầm lẫn 2:
Gradient lớn là tốt
-> Sai

Nhầm lẫn 3:
Initialization không quan trọng
-> Sai

Nhầm lẫn 4:
Gradient clipping luôn cần
-> Không phải lúc nào cũng cần
""")


# ============================================================
# 19. KẾT NỐI SANG PHẦN TIẾP
# ============================================================

print("\n19. KẾT NỐI")
print("""
Sau khi hiểu cách train ổn định:

PPT sẽ đi tiếp:
- ImageNet
- Transfer Learning
- Data Augmentation

=> bước vào classification thực chiến
""")


# ============================================================
# 20. TỔNG KẾT
# ============================================================

print("\n20. TỔNG KẾT")
print("""
1. Vanishing gradient làm model không học được
2. Exploding gradient làm training không ổn định
3. Nguyên nhân là chain rule trong mạng sâu
4. ReLU giúp giảm vanishing gradient
5. Weight initialization rất quan trọng
6. Gradient clipping giúp xử lý exploding gradient
7. Đây là các kỹ thuật nền tảng để train deep network hiệu quả
""")