"""
EX6 - Linear Classifier (Score Function)

Mục tiêu:
1. Hiểu Linear Classifier là gì
2. Hiểu cách biến ảnh thành vector (flatten)
3. Hiểu công thức: f(x) = Wx + b
4. Hiểu score function hoạt động như thế nào
5. Phân biệt cách hiểu trực quan và toán học
6. Nắm nền tảng để học Neural Network

Ghi chú:
- Đây là bước chuyển từ "dữ liệu" sang "model"
- Là nền tảng cực quan trọng cho Deep Learning
"""


# ============================================================
# 1. LINEAR CLASSIFIER LÀ GÌ?
# ============================================================

print("1. LINEAR CLASSIFIER LÀ GÌ?")
print("""
Linear Classifier là một mô hình đơn giản dùng để phân loại dữ liệu.

Ý tưởng:
- Đưa dữ liệu đầu vào x
- Tính score cho từng class
- Class có score cao nhất sẽ là dự đoán

Công thức:
f(x) = Wx + b

Trong đó:
- x: input (vector)
- W: weight (trọng số)
- b: bias
""")


# ============================================================
# 2. TẠI SAO PHẢI FLATTEN ẢNH?
# ============================================================

print("\n2. FLATTEN ẢNH")

print("""
Ảnh ban đầu có thể là:
- (H, W) với grayscale
- (H, W, C) với RGB

Nhưng Linear Classifier làm việc với vector → cần chuyển ảnh thành vector.

Ví dụ:
- ảnh 28x28 → flatten → vector 784
""")

import numpy as np

image = np.array([
    [1, 2],
    [3, 4]
])

flatten_image = image.flatten()

print("Ảnh ban đầu:\n", image)
print("Sau khi flatten:", flatten_image)


# ============================================================
# 3. SCORE FUNCTION LÀ GÌ?
# ============================================================

print("\n3. SCORE FUNCTION")

print("""
Score function là hàm tính điểm cho từng class.

Ví dụ:
- bài toán có 3 class: cat, dog, bird
- output sẽ là:
  [2.1, 0.5, -1.2]

=> cat có score cao nhất → dự đoán là cat
""")


# ============================================================
# 4. CÔNG THỨC TOÁN HỌC
# ============================================================

print("\n4. CÔNG THỨC TOÁN HỌC")

print("""
f(x) = Wx + b

Trong đó:
- x: vector input (D chiều)
- W: ma trận trọng số (K x D)
- b: vector bias (K chiều)
- output: vector score (K chiều)

K = số class
""")


# ============================================================
# 5. VÍ DỤ CỤ THỂ
# ============================================================

print("\n5. VÍ DỤ CỤ THỂ")

# giả sử:
# input 4 chiều
# 3 class

x = np.array([1, 2, 3, 4])  # input

W = np.array([
    [0.1, 0.2, 0.3, 0.4],  # class 1
    [0.5, 0.6, 0.7, 0.8],  # class 2
    [0.9, 1.0, 1.1, 1.2]   # class 3
])

b = np.array([0.1, 0.2, 0.3])

scores = np.dot(W, x) + b

print("Input x:", x)
print("Score:", scores)
print("Predicted class:", np.argmax(scores))


# ============================================================
# 6. HIỂU THEO CÁCH TRỰC QUAN
# ============================================================

print("\n6. HIỂU TRỰC QUAN")

print("""
Bạn có thể tưởng tượng:

- mỗi pixel có một "độ quan trọng"
- weight nói rằng pixel nào quan trọng với class nào

Quá trình:
- pixel * weight
- cộng tất cả lại
- cộng bias

=> ra score

=> score cao nhất → class đó thắng
""")


# ============================================================
# 7. HIỂU THEO MA TRẬN
# ============================================================

print("\n7. HIỂU THEO MA TRẬN")

print("""
Thay vì nhân từng pixel thủ công:

Ta dùng:
np.dot(W, x)

Ưu điểm:
- nhanh
- gọn
- tận dụng tối ưu phần cứng

Đây là cách chuẩn trong Deep Learning
""")


# ============================================================
# 8. VÍ DỤ MNIST (QUAN TRỌNG)
# ============================================================

print("\n8. VÍ DỤ MNIST")

print("""
Ảnh MNIST:
- size: 28x28
- flatten → 784

Số class:
- 10 (digits 0 → 9)

=> W sẽ có shape:
(10, 784)

=> output:
vector 10 chiều (score cho 10 class)
""")


# ============================================================
# 9. WEIGHT VISUALIZATION
# ============================================================

print("\n9. WEIGHT VISUALIZATION")

print("""
Trọng số của mỗi class có thể reshape thành ảnh.

Ý tưởng:
- W của class 0 → trông giống số 0
- W của class 1 → trông giống số 1

=> model đang học pattern của từng class

Đây là cách giúp hiểu model đang học gì
""")


# ============================================================
# 10. GIỚI HẠN CỦA LINEAR CLASSIFIER
# ============================================================

print("\n10. GIỚI HẠN")

print("""
Linear Classifier chỉ học được quan hệ tuyến tính.

=> không học được pattern phức tạp

Ví dụ:
- không phân biệt tốt dữ liệu phi tuyến
- không hiểu cấu trúc ảnh (spatial)

=> cần Neural Network để giải quyết
""")


# ============================================================
# 11. LIÊN HỆ VỚI NEURAL NETWORK
# ============================================================

print("\n11. LIÊN HỆ VỚI NEURAL NETWORK")

print("""
Neuron = Linear Classifier + Activation

Linear:
z = Wx + b

Activation:
a = f(z)

=> Neural Network = nhiều Linear + Activation
""")


# ============================================================
# 12. TỔNG KẾT
# ============================================================

print("\n12. TỔNG KẾT")

print("""
1. Ảnh phải flatten thành vector
2. Linear Classifier dùng công thức:
   f(x) = Wx + b
3. Output là score cho từng class
4. Class có score cao nhất được chọn
5. Đây là nền tảng của Neural Network
6. Linear model có hạn chế → cần model sâu hơn
""")