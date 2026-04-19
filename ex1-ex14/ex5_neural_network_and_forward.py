"""
EX5 - Neural Network và Forward Propagation

Mục tiêu:
1. Hiểu Neural Network là gì
2. Hiểu cấu trúc của một mạng neural cơ bản
3. Hiểu input layer, hidden layer, output layer
4. Hiểu forward propagation / forward pass / feedforward
5. Hiểu ví dụ MNIST trong neural network
6. Hiểu Deep Neural Network khác gì mạng nông

Tài liệu gốc:
- PPT Deep Learning / Computer Vision của khóa học
"""


# ============================================================
# 1. NEURAL NETWORK LÀ GÌ?
# ============================================================

print("1. NEURAL NETWORK LÀ GÌ?")
print("""
Neural Network là mô hình được tạo thành từ nhiều neuron nhân tạo kết nối với nhau.

Nếu ở bài trước:
- ta chỉ có một neuron đơn lẻ
- neuron đó thực hiện:
    z = Wx + b
    a = activation(z)

thì bây giờ:
- ta ghép nhiều neuron lại
- thành một hệ thống có thể học tốt hơn
- đó chính là Neural Network
""")


# ============================================================
# 2. CẤU TRÚC CƠ BẢN CỦA NEURAL NETWORK
# ============================================================

print("\n2. CẤU TRÚC CƠ BẢN CỦA NEURAL NETWORK")
print("""
Trong PPT, kiến trúc neural network cơ bản gồm 3 loại layer:

1. Input Layer
2. Hidden Layer
3. Output Layer

Đây là cấu trúc nền tảng nhất của neural network.
""")


# ============================================================
# 3. INPUT LAYER
# ============================================================

print("\n3. INPUT LAYER")
print("""
Input Layer là lớp nhận dữ liệu đầu vào.

Ví dụ:
- nếu ảnh MNIST có kích thước 28x28
- sau khi flatten, ta có 784 pixel
- khi đó input layer có thể xem như nhận 784 giá trị

Input layer không phải nơi "học" mạnh nhất,
mà chủ yếu là nơi tiếp nhận dữ liệu để truyền vào mạng.
""")


# ============================================================
# 4. HIDDEN LAYER
# ============================================================

print("\n4. HIDDEN LAYER")
print("""
Hidden Layer là lớp ẩn nằm giữa input và output.

Vai trò:
- học ra các đặc trưng trung gian
- biến đổi dữ liệu từng bước
- làm cho mạng có khả năng học pattern phức tạp

Nếu chỉ có 1 hidden layer:
- mạng còn khá đơn giản

Nếu có nhiều hidden layers:
- ta có Deep Neural Network
""")


# ============================================================
# 5. OUTPUT LAYER
# ============================================================

print("\n5. OUTPUT LAYER")
print("""
Output Layer là lớp cuối cùng của mạng.

Vai trò:
- trả kết quả dự đoán cuối cùng

Ví dụ:
- bài toán MNIST có 10 class
- output layer có thể có 10 neuron
- mỗi neuron tương ứng một class từ 0 đến 9

Mỗi neuron ở output layer sẽ cho ra một score hoặc xác suất.
""")


# ============================================================
# 6. KIẾN TRÚC CỦA MỘT MẠNG ĐƠN GIẢN
# ============================================================

print("\n6. KIẾN TRÚC MỘT MẠNG ĐƠN GIẢN")
print("""
Một mạng neural đơn giản có thể có dạng:

Input -> Hidden -> Output

Ví dụ:
- 784 input features
- 128 hidden units
- 10 output units

Nghĩa là:
- ảnh đi vào
- được biến đổi qua hidden layer
- rồi đưa ra score cho 10 class
""")


# ============================================================
# 7. FORWARD PROPAGATION LÀ GÌ?
# ============================================================

print("\n7. FORWARD PROPAGATION LÀ GÌ?")
print("""
Forward Propagation (hay Forward Pass, Feedforward) là quá trình dữ liệu đi từ đầu vào đến đầu ra.

Hướng đi:
Input Layer -> Hidden Layer(s) -> Output Layer

Ý nghĩa:
- mô hình nhận dữ liệu
- tính toán từng lớp
- cho ra dự đoán cuối cùng

Đây là bước "đi xuôi" của mạng.
""")


# ============================================================
# 8. FEEDFORWARD NEURAL NETWORK
# ============================================================

print("\n8. FEEDFORWARD NEURAL NETWORK")
print("""
Feedforward Neural Network là loại mạng mà dữ liệu chỉ đi theo một hướng:

Từ input -> qua các hidden layers -> ra output

Không có vòng lặp ngược trong cấu trúc mạng.

Đây là dạng mạng cơ bản nhất trong Deep Learning nhập môn.
""")


# ============================================================
# 9. FORWARD PASS Ở MỨC TOÁN HỌC
# ============================================================

print("\n9. FORWARD PASS Ở MỨC TOÁN HỌC")
print("""
Ở mỗi layer, mạng sẽ làm 2 bước:

Bước 1:
z = Wx + b

Bước 2:
a = activation(z)

Sau đó:
- a của layer trước
- trở thành input của layer sau

Quá trình này lặp lại cho đến output layer.
""")


import numpy as np

# ví dụ nhỏ với 1 layer
x = np.array([1.0, 2.0, 3.0])
W = np.array([
    [0.2, 0.1, 0.4],
    [0.5, 0.3, 0.2]
])
b = np.array([0.1, 0.2])

z = np.dot(W, x) + b
a = np.maximum(0, z)  # ReLU

print("Input x =", x)
print("Linear output z =", z)
print("Activated output a =", a)


# ============================================================
# 10. COMPLETE EXAMPLE OF NEURAL NETWORK (Ý TƯỞNG)
# ============================================================

print("\n10. COMPLETE EXAMPLE OF NEURAL NETWORK")
print("""
Trong PPT có ví dụ complete example với ảnh chữ số viết tay.

Ý tưởng là:
- ảnh 28x28 được flatten thành 784 giá trị
- 784 giá trị đó đi vào mạng
- mạng tính toán qua các neuron
- cuối cùng trả ra kết quả dự đoán

Điểm cần nhớ:
- mạng không nhìn "ảnh" như mắt người
- mạng chỉ xử lý vector số
- nhưng qua nhiều phép biến đổi, nó học được pattern của chữ số
""")


# ============================================================
# 11. VÍ DỤ ĐƠN GIẢN 2 LAYER
# ============================================================

print("\n11. VÍ DỤ ĐƠN GIẢN 2 LAYER")

def relu(x):
    return np.maximum(0, x)

# input 4 chiều
x = np.array([0.5, 0.2, 0.1, 0.7])

# layer 1: 4 -> 3
W1 = np.array([
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.1, 0.2, 0.3],
    [0.2, 0.4, 0.1, 0.6]
])
b1 = np.array([0.1, 0.2, 0.3])

# layer 2: 3 -> 2
W2 = np.array([
    [0.2, 0.3, 0.4],
    [0.5, 0.1, 0.2]
])
b2 = np.array([0.1, 0.2])

# forward pass
z1 = np.dot(W1, x) + b1
a1 = relu(z1)

z2 = np.dot(W2, a1) + b2
a2 = relu(z2)

print("z1 =", z1)
print("a1 =", a1)
print("z2 =", z2)
print("a2 =", a2)
print("Predicted class =", np.argmax(a2))


# ============================================================
# 12. NEURAL NETWORK ĐANG HỌC GÌ?
# ============================================================

print("\n12. NEURAL NETWORK ĐANG HỌC GÌ?")
print("""
Một cách trực quan, mạng neural học theo tầng:

- tầng đầu: học pattern đơn giản
- tầng giữa: học tổ hợp pattern
- tầng sâu hơn: học pattern phức tạp hơn

Với ảnh:
- layer đầu có thể học cạnh, nét
- layer sâu hơn học bộ phận
- layer sâu nữa học đối tượng hoàn chỉnh

Đây là lý do mạng sâu mạnh hơn linear classifier đơn thuần.
""")


# ============================================================
# 13. DEEP NEURAL NETWORK LÀ GÌ?
# ============================================================

print("\n13. DEEP NEURAL NETWORK LÀ GÌ?")
print("""
Deep Neural Network là neural network có nhiều hidden layers.

Điểm khác với mạng nông:
- nhiều tầng hơn
- học được biểu diễn phức tạp hơn

Trong PPT, Deep Neural Network được minh họa như quá trình học:
- edges
- combinations of edges
- object models

Đây là ý rất quan trọng:
mạng sâu học từ đặc trưng đơn giản đến đặc trưng trừu tượng hơn.
""")


# ============================================================
# 14. TẠI SAO MẠNG SÂU MẠNH HƠN?
# ============================================================

print("\n14. TẠI SAO MẠNG SÂU MẠNH HƠN?")
print("""
Mạng sâu mạnh hơn vì:
- có nhiều tầng biến đổi dữ liệu
- có thể học các quan hệ phi tuyến phức tạp
- thích hợp với dữ liệu ảnh, âm thanh, văn bản

Nếu chỉ dùng Linear Classifier:
- model bị giới hạn ở quan hệ tuyến tính

Nếu dùng Deep Neural Network:
- model học được cấu trúc phức tạp hơn nhiều
""")


# ============================================================
# 15. LIÊN HỆ VỚI BÀI TOÁN IMAGE CLASSIFICATION
# ============================================================

print("\n15. LIÊN HỆ VỚI IMAGE CLASSIFICATION")
print("""
Trong bài toán image classification:
- ảnh là input
- mạng neural xử lý ảnh qua nhiều layer
- output layer trả ra score / xác suất cho các class

Ví dụ:
- ảnh con chim
- output layer có thể trả về:
  dog, bird, cat
- class có score cao nhất sẽ là dự đoán cuối cùng
""")


# ============================================================
# 16. NHỮNG ĐIỀU DỄ NHẦM KHI HỌC NEURAL NETWORK
# ============================================================

print("\n16. NHỮNG ĐIỀU DỄ NHẦM")
print("""
Nhầm lẫn 1:
Neural Network = chỉ là nhiều công thức
-> Sai
Bản chất là nhiều neuron ghép lại thành hệ thống

Nhầm lẫn 2:
Forward pass là học
-> Chưa đủ
Forward pass chỉ là bước tính đầu ra
Muốn học được còn cần loss + backprop + optimizer

Nhầm lẫn 3:
Càng nhiều layer càng luôn tốt
-> Không hẳn
Mạng sâu mạnh hơn, nhưng cũng khó train hơn
""")


# ============================================================
# 17. KẾT NỐI SANG PHẦN TIẾP THEO
# ============================================================

print("\n17. KẾT NỐI SANG PHẦN TIẾP THEO")
print("""
Sau khi có output từ neural network,
ta cần biết:
- output đó tốt hay xấu?
- dự đoán đó sai bao nhiêu?

Đó là lý do phần tiếp theo trong PPT chuyển sang:
LOSS FUNCTION
""")


# ============================================================
# 18. TỔNG KẾT
# ============================================================

print("\n18. TỔNG KẾT")
print("""
1. Neural Network là tập hợp nhiều neuron kết nối với nhau
2. Cấu trúc cơ bản gồm:
   - Input Layer
   - Hidden Layer
   - Output Layer
3. Forward propagation là quá trình dữ liệu đi từ input đến output
4. Feedforward neural network là mạng đi một chiều từ đầu vào đến đầu ra
5. Deep Neural Network có nhiều hidden layers và học đặc trưng tốt hơn
6. Đây là bước nối giữa neuron đơn lẻ và Deep Learning thật sự
""")