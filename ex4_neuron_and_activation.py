"""
EX4 - From Linear Classifier to Neuron & Activation Function

Mục tiêu:
1. Hiểu neuron sinh học vs neuron nhân tạo
2. Hiểu neuron = Linear Classifier + Activation
3. Hiểu vì sao cần activation function
4. Nắm các activation phổ biến:
   - Sigmoid
   - Tanh
   - ReLU
   - Leaky ReLU
5. Hiểu bước chuyển từ Linear Model → Neural Network
"""


# ============================================================
# 1. BIOLOGICAL NEURON (NEURON SINH HỌC)
# ============================================================

print("1. BIOLOGICAL NEURON")

print("""
Trong PPT có hình neuron sinh học:

- Nhận tín hiệu từ các neuron khác
- Tổng hợp tín hiệu
- Nếu đủ mạnh → phát tín hiệu ra

Ý tưởng chính:
INPUT -> PROCESS -> OUTPUT

Đây chính là nguồn cảm hứng cho Artificial Neural Network
""")


# ============================================================
# 2. ARTIFICIAL NEURON (NEURON NHÂN TẠO)
# ============================================================

print("\n2. ARTIFICIAL NEURON")

print("""
Neuron nhân tạo được xây dựng dựa trên ý tưởng đó.

Nó gồm 3 bước:

1. Nhận input x
2. Nhân với weight + cộng bias
3. Đi qua activation function

=> Output

Công thức:

z = Wx + b
a = f(z)
""")


# ============================================================
# 3. TỪ LINEAR CLASSIFIER ĐẾN NEURON
# ============================================================

print("\n3. FROM LINEAR CLASSIFIER TO NEURON")

print("""
Linear Classifier:
f(x) = Wx + b

Neuron:
z = Wx + b
a = activation(z)

=> Neuron = Linear + Activation

Đây là bước chuyển cực kỳ quan trọng trong PPT:
- từ model tuyến tính
- sang model phi tuyến (non-linear)
""")


# ============================================================
# 4. TẠI SAO CẦN ACTIVATION FUNCTION?
# ============================================================

print("\n4. TẠI SAO CẦN ACTIVATION FUNCTION?")

print("""
Nếu không có activation:

- nhiều layer Linear ghép lại
- vẫn chỉ là một phép biến đổi tuyến tính

=> model KHÔNG mạnh hơn

Activation giúp:
- tạo tính phi tuyến (non-linearity)
- cho phép model học pattern phức tạp

=> đây là chìa khóa của Deep Learning
""")


# ============================================================
# 5. SIGMOID
# ============================================================

import numpy as np

print("\n5. SIGMOID")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

print("""
Sigmoid:
f(x) = 1 / (1 + e^(-x))

Đầu ra:
(0, 1)

Ý nghĩa:
- thường dùng cho bài toán xác suất
- ví dụ binary classification

Nhược điểm:
- dễ bị vanishing gradient
""")

print("Sigmoid(0) =", sigmoid(0))


# ============================================================
# 6. TANH
# ============================================================

print("\n6. TANH")

def tanh(x):
    return np.tanh(x)

print("""
Tanh:
f(x) ∈ (-1, 1)

Ưu điểm:
- zero-centered (tốt hơn sigmoid)

Nhược điểm:
- vẫn bị vanishing gradient
""")

print("Tanh(0) =", tanh(0))


# ============================================================
# 7. RELU
# ============================================================

print("\n7. RELU")

def relu(x):
    return np.maximum(0, x)

print("""
ReLU:
f(x) = max(0, x)

Ưu điểm:
- đơn giản
- nhanh
- hiệu quả trong thực tế

Nhược điểm:
- chết neuron (x < 0 → luôn 0)
""")

print("ReLU([-2, -1, 0, 1, 2]) =", relu(np.array([-2, -1, 0, 1, 2])))


# ============================================================
# 8. LEAKY RELU
# ============================================================

print("\n8. LEAKY RELU")

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

print("""
Leaky ReLU:
f(x) = x nếu x > 0
     = alpha * x nếu x < 0

Ưu điểm:
- tránh chết neuron

=> cải tiến từ ReLU
""")

print("LeakyReLU([-2, -1, 0, 1, 2]) =", leaky_relu(np.array([-2, -1, 0, 1, 2])))


# ============================================================
# 9. SO SÁNH ACTIVATION
# ============================================================

print("\n9. SO SÁNH ACTIVATION")

print("""
Sigmoid:
- (0,1)
- dùng cho xác suất

Tanh:
- (-1,1)
- tốt hơn sigmoid

ReLU:
- nhanh, phổ biến nhất

Leaky ReLU:
- cải tiến ReLU
""")


# ============================================================
# 10. NEURON HOẠT ĐỘNG NHƯ THẾ NÀO?
# ============================================================

print("\n10. NEURON HOẠT ĐỘNG NHƯ THẾ NÀO?")

x = np.array([1, 2, 3])
W = np.array([0.5, -0.2, 0.1])
b = 0.3

z = np.dot(W, x) + b
a = relu(z)

print("z =", z)
print("a (after activation) =", a)


# ============================================================
# 11. TẠI SAO ĐÂY LÀ BƯỚC QUAN TRỌNG?
# ============================================================

print("\n11. Ý NGHĨA QUAN TRỌNG")

print("""
Linear Classifier:
- chỉ học tuyến tính

Neuron:
- thêm activation → học phi tuyến

=> từ đây mới xây được Neural Network

=> đây là bước:
Machine Learning → Deep Learning
""")


# ============================================================
# 12. KẾT NỐI SANG PHẦN TIẾP
# ============================================================

print("\n12. KẾT NỐI SANG NEURAL NETWORK")

print("""
Trong PPT, sau phần này sẽ chuyển sang:

- Neural Network architecture
- Forward propagation
- Deep Neural Network

=> nhiều neuron ghép lại thành network
""")


# ============================================================
# 13. TỔNG KẾT
# ============================================================

print("\n13. TỔNG KẾT")

print("""
1. Neuron = Linear + Activation
2. Activation tạo tính phi tuyến
3. Không có activation → model vô dụng
4. ReLU là activation phổ biến nhất
5. Đây là nền tảng của Neural Network
""")