"""
EX7 - Linear Explanation

Mục tiêu:
1. Hiểu sâu cách Linear Classifier hoạt động
2. Hiểu rõ mapping từ input → output
3. Hiểu ví dụ MNIST: 784 → 10 class
4. Hiểu tại sao output là vector score
5. Hiểu cách model đưa ra quyết định cuối cùng

Ghi chú:
- Đây là phần "giải thích sâu" Linear Classifier
- Không phải model mới, mà là hiểu kỹ hơn model cũ
"""


# ============================================================
# 1. Ý TƯỞNG CHÍNH CỦA LINEAR EXPLANATION
# ============================================================

print("1. Ý TƯỞNG CHÍNH")

print("""
Trong PPT:

Input: 784 pixel
Output: 10 class

=> nghĩa là:

- ảnh được biến thành vector 784 chiều
- model trả ra 10 giá trị (score)
- mỗi giá trị tương ứng với 1 class

=> đây là mapping:
784 → 10
""")


# ============================================================
# 2. VÍ DỤ MNIST (QUAN TRỌNG NHẤT)
# ============================================================

print("\n2. VÍ DỤ MNIST")

print("""
Dataset MNIST:
- ảnh chữ số viết tay
- kích thước: 28 x 28

=> flatten:
28 x 28 = 784 pixel

=> input vector:
x ∈ R^784

Output:
- 10 class (0 → 9)

=> output vector:
f(x) ∈ R^10
""")


# ============================================================
# 3. CÁCH MODEL TÍNH TOÁN
# ============================================================

print("\n3. CÁCH MODEL TÍNH")

print("""
Công thức:

f(x) = Wx + b

Trong đó:
- x: vector input (784)
- W: ma trận (10 x 784)
- b: vector bias (10)

=> output:
vector 10 chiều (score cho từng class)
""")


# ============================================================
# 4. VÍ DỤ CỤ THỂ
# ============================================================

print("\n4. VÍ DỤ CỤ THỂ")

import numpy as np

# giả lập ảnh MNIST (784 pixel)
x = np.random.rand(784)

# W: 10 class, mỗi class có 784 weight
W = np.random.rand(10, 784)

# bias
b = np.random.rand(10)

# tính score
scores = np.dot(W, x) + b

print("Shape input:", x.shape)
print("Shape output:", scores.shape)
print("Scores:", scores)
print("Predicted class:", np.argmax(scores))


# ============================================================
# 5. OUTPUT VECTOR NGHĨA LÀ GÌ?
# ============================================================

print("\n5. OUTPUT VECTOR")

print("""
Output là một vector gồm 10 giá trị.

Ví dụ:
[1.2, -0.5, 3.8, 0.1, 2.4, ...]

=> mỗi giá trị là score của 1 class

Giả sử:
index 0 → số 0
index 1 → số 1
...
index 9 → số 9

=> score cao nhất → class đó được chọn
""")


# ============================================================
# 6. QUY TẮC QUYẾT ĐỊNH (ARGMAX)
# ============================================================

print("\n6. QUY TẮC QUYẾT ĐỊNH")

print("""
Model không chọn ngẫu nhiên.

Nó chọn:
argmax(scores)

=> class có score lớn nhất

Ví dụ:
scores = [1.2, 0.5, 3.8, 2.0]

=> class = 2 (vì 3.8 lớn nhất)
""")


# ============================================================
# 7. HIỂU TRỰC QUAN (QUAN TRỌNG)
# ============================================================

print("\n7. HIỂU TRỰC QUAN")

print("""
Bạn tưởng tượng:

- mỗi class có một "bộ lọc" riêng (weight W)
- bộ lọc đó kiểm tra ảnh

Ví dụ:
- bộ lọc số 3 sẽ tìm pattern giống số 3
- bộ lọc số 8 sẽ tìm pattern giống số 8

=> ảnh nào match tốt nhất → score cao nhất

=> class đó thắng
""")


# ============================================================
# 8. VÌ SAO LÀ 784 → 10?
# ============================================================

print("\n8. TẠI SAO 784 → 10?")

print("""
784:
- là số pixel của ảnh

10:
- là số class

=> model đang học:
"pixel pattern → class"

Đây chính là bản chất của classification
""")


# ============================================================
# 9. MỘT CÁCH NHÌN KHÁC (RẤT QUAN TRỌNG)
# ============================================================

print("\n9. CÁCH NHÌN KHÁC")

print("""
Linear Classifier = 10 bộ weight khác nhau

=> mỗi class có:
- 1 vector weight riêng

=> model đang làm:
- so sánh ảnh với từng class

=> class nào giống nhất → chọn
""")


# ============================================================
# 10. HẠN CHẾ (NHẮC LẠI)
# ============================================================

print("\n10. HẠN CHẾ")

print("""
Linear Classifier:
- chỉ học tuyến tính
- không hiểu cấu trúc ảnh

=> không học được:
- hình dạng phức tạp
- quan hệ phi tuyến

=> cần Neural Network
""")


# ============================================================
# 11. TỔNG KẾT
# ============================================================

print("\n11. TỔNG KẾT")

print("""
1. Input: 784 pixel
2. Output: 10 class
3. Công thức: f(x) = Wx + b
4. Output là vector score
5. Chọn class bằng argmax
6. Model so sánh ảnh với từng class
7. Đây là nền tảng để hiểu Neural Network
""")