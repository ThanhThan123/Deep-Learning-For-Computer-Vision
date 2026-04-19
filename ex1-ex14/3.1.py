"""
EX3 - Linear Classifier

Nội dung chính:
1. Linear Classifier là gì
2. Score function: f(x, W, b) = Wx + b
3. Vì sao phải flatten ảnh
4. Linear explanation với ví dụ MNIST (784 -> 10 class)
5. Weight visualization
6. Liên hệ giữa Linear Classifier và bước chuyển sang Neural Network

Tài liệu gốc:
- PPT Deep Learning / Computer Vision của khóa học
"""

# ============================================================
# 1. LINEAR CLASSIFIER LÀ GÌ?
# ============================================================

print("1. LINEAR CLASSIFIER LÀ GÌ?")
print("""
Linear Classifier là mô hình phân loại đơn giản dùng để gán nhãn cho dữ liệu đầu vào.


Ý tưởng chính:
- đưa dữ liệu đầu vào x vào mô hình
- mô hình tính ra điểm số (score) cho từng class
- class nào có score lớn nhất sẽ là kết quả dự đoán

Trong Computer Vision, ảnh đầu vào phải được chuyển thành dạng số
để mô hình có thể tính toán.
""")


# ============================================================
# 2. SCORE FUNCTION
# ============================================================

print("\n2. SCORE FUNCTION")
print("""
Trong PPT, công thức score function là:

f(x_i, W, b) = W x_i + b

Trong đó:
- x_i: vector biểu diễn ảnh đầu vào
- W: ma trận trọng số
- b: bias
- f(...): vector điểm số đầu ra

Ý nghĩa:
- mỗi class sẽ nhận được một score riêng
- class có score lớn nhất sẽ được chọn
""")


# ============================================================
# 3. TẠI SAO PHẢI FLATTEN ẢNH?
# ============================================================

print("\n3. TẠI SAO PHẢI FLATTEN ẢNH?")
print("""
Ảnh ban đầu thường có dạng:
- grayscale: (height, width)
- color image: (height, width, channels)

Nhưng Linear Classifier làm việc với vector.

Vì vậy, trước khi đưa ảnh vào mô hình, ta cần:
- trải phẳng ảnh thành một vector 1 chiều

Ví dụ:
- ảnh 28x28 -> vector 784 phần tử
- ảnh 32x32x3 -> vector 3072 phần tử
""")

import numpy as np

image = np.array([
    [1, 2],
    [3, 4]
])

x = image.flatten()

print("Ảnh ban đầu:")
print(image)
print("Sau khi flatten:")
print(x)


# ============================================================
# 4. HIỂU SCORE FUNCTION THEO CÁCH TRỰC QUAN
# ============================================================

print("\n4. HIỂU SCORE FUNCTION THEO CÁCH TRỰC QUAN")
print("""
Có thể hiểu đơn giản như sau:

- mỗi pixel của ảnh sẽ được nhân với một weight tương ứng
- sau đó cộng tất cả các kết quả lại
- rồi cộng thêm bias

=> ta thu được một score cho một class

Nếu có nhiều class:
- mô hình sẽ tính nhiều score
- mỗi score tương ứng với một class khác nhau

Cuối cùng:
- class nào có score lớn nhất -> mô hình chọn class đó
""")


# ============================================================
# 5. HIỂU SCORE FUNCTION THEO CÁCH TOÁN HỌC
# ============================================================

print("\n5. HIỂU SCORE FUNCTION THEO CÁCH TOÁN HỌC")
print("""
Thay vì nhân từng pixel bằng tay, ta dùng phép nhân ma trận:

f(x) = Wx + b

Giả sử:
- x có D phần tử
- có K class

Khi đó:
- W có shape là (K, D)
- b có shape là (K,)
- output có shape là (K,)

Tức là:
- đầu ra là vector K chiều
- mỗi chiều là score của một class
""")


# Ví dụ số nhỏ
x = np.array([1.0, 2.0, 3.0, 4.0])  # input 4 chiều

W = np.array([
    [0.1, 0.2, 0.3, 0.4],   # class 0
    [0.5, 0.6, 0.7, 0.8],   # class 1
    [0.9, 1.0, 1.1, 1.2]    # class 2
])

b = np.array([0.1, 0.2, 0.3])

scores = np.dot(W, x) + b

print("Input x =", x)
print("Scores =", scores)
print("Predicted class =", np.argmax(scores))


# ============================================================
# 6. LINEAR EXPLANATION VỚI MNIST
# ============================================================

print("\n6. LINEAR EXPLANATION VỚI MNIST")
print("""
Trong PPT, phần Linear Classifier explanation dùng ví dụ MNIST.

MNIST là dataset chữ số viết tay:
- ảnh grayscale
- kích thước 28 x 28

Điều này có nghĩa:
- đầu vào có 28 * 28 = 784 pixel
- sau khi flatten, ta có vector 784 chiều

Đầu ra:
- 10 class tương ứng với các số từ 0 đến 9

Vậy:
Input: 784 pixel
Output: 10 class

Đây là mapping rất quan trọng:
784 -> 10
""")


# Ví dụ shape theo MNIST
mnist_input_dim = 28 * 28
num_classes = 10

W_mnist = np.random.randn(num_classes, mnist_input_dim)
b_mnist = np.random.randn(num_classes)
x_mnist = np.random.randn(mnist_input_dim)

scores_mnist = np.dot(W_mnist, x_mnist) + b_mnist

print("MNIST input dim =", mnist_input_dim)
print("MNIST output dim =", scores_mnist.shape[0])
print("Predicted digit =", np.argmax(scores_mnist))


# ============================================================
# 7. OUTPUT VECTOR NGHĨA LÀ GÌ?
# ============================================================

print("\n7. OUTPUT VECTOR NGHĨA LÀ GÌ?")
print("""
Giả sử output là:

[1.2, -0.3, 4.8, 0.6, 2.1, 0.9, -1.0, 1.5, 0.2, 0.4]

Ta hiểu:
- phần tử thứ 0 là score của class 0
- phần tử thứ 1 là score của class 1
- ...
- phần tử thứ 9 là score của class 9

Nếu score lớn nhất nằm ở vị trí 2:
=> mô hình dự đoán ảnh là số 2
""")


# ============================================================
# 8. WEIGHT VISUALIZATION LÀ GÌ?
# ============================================================

print("\n8. WEIGHT VISUALIZATION LÀ GÌ?")
print("""
Trong PPT, sau phần linear explanation là weight visualization.

Ý tưởng:
- mỗi class có một bộ trọng số riêng
- nếu reshape vector trọng số đó về dạng ảnh,
  ta có thể quan sát xem mô hình đang "ưu tiên" pattern nào

Ví dụ với MNIST:
- weight của class 0 thường trông giống số 0
- weight của class 1 thường trông giống số 1

Điều này không có nghĩa ảnh weight sẽ đẹp như ảnh gốc,
nhưng nó cho thấy mô hình đang học những pattern liên quan đến từng class.
""")


# ============================================================
# 9. HAI CÁCH NHÌN WEIGHT VISUALIZATION
# ============================================================

print("\n9. HAI CÁCH NHÌN WEIGHT VISUALIZATION")
print("""
Trong PPT, có thể hiểu weight visualization theo 2 cách:

Cách 1:
- trải phẳng input x để nó tương thích với vector weight w

Cách 2:
- reshape weight w thành dạng hình vuông / dạng ảnh
- để nhìn trực quan hơn

Mục đích chung:
- hiểu model đang học cái gì
- hiểu tại sao class nào đó có score cao
""")


# Ví dụ reshape weight thành ảnh 28x28
w_for_digit_0 = np.random.randn(784)
w_image = w_for_digit_0.reshape(28, 28)

print("Weight vector shape =", w_for_digit_0.shape)
print("Reshaped weight image shape =", w_image.shape)


# ============================================================
# 10. WEIGHT VISUALIZATION VỚI CIFAR
# ============================================================

print("\n10. WEIGHT VISUALIZATION VỚI CIFAR")
print("""
PPT cũng có ví dụ weight visualization trên CIFAR dataset.

Khác với MNIST:
- MNIST là ảnh xám, đơn giản hơn
- CIFAR là ảnh màu, nhiều class hơn, phức tạp hơn

Khi visualize trọng số trên CIFAR:
- ảnh weight sẽ nhiều màu
- khó nhìn trực quan bằng MNIST
- nhưng vẫn giúp ta thấy mô hình đang học pattern màu sắc / cấu trúc nào

Điều này cho thấy:
- dữ liệu càng phức tạp
- weight visualization càng khó đọc trực quan
""")


# ============================================================
# 11. LINEAR CLASSIFIER ĐANG THỰC SỰ HỌC GÌ?
# ============================================================

print("\n11. LINEAR CLASSIFIER ĐANG HỌC GÌ?")
print("""
Linear Classifier đang học:

- với class A:
  pixel nào nên làm tăng score
  pixel nào nên làm giảm score

Tức là mô hình học ra:
- một "mẫu trọng số" cho từng class

Khi input mới đi vào:
- mô hình so sánh input với các mẫu đó
- class nào phù hợp nhất -> score cao nhất
""")


# ============================================================
# 12. GIỚI HẠN CỦA LINEAR CLASSIFIER
# ============================================================

print("\n12. GIỚI HẠN CỦA LINEAR CLASSIFIER")
print("""
Linear Classifier có ưu điểm:
- đơn giản
- dễ hiểu
- là nền tảng tốt để học Deep Learning

Nhưng nó có giới hạn:
- chỉ học được quan hệ tuyến tính
- không tận dụng tốt cấu trúc không gian của ảnh
- khó xử lý các pattern phức tạp

Đó là lý do vì sao PPT chuyển tiếp từ:
Linear Classifier -> Neuron -> Neural Network
""")


# ============================================================
# 13. KẾT NỐI SANG PHẦN TIẾP THEO
# ============================================================

print("\n13. KẾT NỐI SANG PHẦN TIẾP THEO")
print("""
Trong PPT, sau khi học xong:
- score function
- linear explanation
- weight visualization

thì bài giảng chuyển sang:
- Biological Neuron vs Artificial Neuron
- From linear classifier to neuron
- Activation function

Điều đó cho thấy:
Neuron thực chất là bước mở rộng từ Linear Classifier.
""")


# ============================================================
# 14. TỔNG KẾT
# ============================================================

print("\n14. TỔNG KẾT")
print("""
1. Linear Classifier dùng công thức:
   f(x, W, b) = Wx + b

2. Ảnh phải được flatten thành vector trước khi đưa vào mô hình tuyến tính

3. Đầu ra là vector score cho các class

4. Với MNIST:
   - Input: 784 pixel
   - Output: 10 class

5. Weight visualization giúp ta nhìn trực quan model đang học pattern gì

6. Linear Classifier là nền tảng để hiểu neuron và neural network
""")