"""
EX9 - Parameters vs Hyperparameters

Mục tiêu:
1. Hiểu parameter là gì
2. Hiểu hyperparameter là gì
3. Phân biệt rõ 2 khái niệm này
4. Biết ví dụ cụ thể trong Neural Network
5. Hiểu vì sao phải chọn hyperparameter đúng

Tài liệu gốc:
- PPT Deep Learning / Computer Vision
"""


# ============================================================
# 1. TỔNG QUAN
# ============================================================

print("1. TỔNG QUAN")
print("""
Trong Deep Learning, có 2 nhóm rất dễ nhầm:

1. Parameters
2. Hyperparameters

Nếu không phân biệt rõ 2 nhóm này,
bạn sẽ rất dễ rối khi train model.
""")


# ============================================================
# 2. PARAMETERS LÀ GÌ?
# ============================================================

print("\n2. PARAMETERS LÀ GÌ?")
print("""
Parameters là các giá trị mà model tự học từ dữ liệu trong quá trình training.

Ví dụ điển hình:
- weights
- bias

Trong công thức:
z = Wx + b

thì:
- W là parameter
- b là parameter

Điểm quan trọng:
- parameters không do ta gán thủ công cố định
- model sẽ cập nhật chúng qua từng vòng học
""")


# ============================================================
# 3. VÌ SAO PARAMETERS QUAN TRỌNG?
# ============================================================

print("\n3. VÌ SAO PARAMETERS QUAN TRỌNG?")
print("""
Parameters chính là phần "tri thức" mà model học được từ dữ liệu.

Nếu model học tốt:
- weight và bias sẽ được điều chỉnh phù hợp
- từ đó prediction tốt hơn

Nếu parameters chưa tốt:
- model dự đoán kém
- loss cao
""")


# ============================================================
# 4. HYPERPARAMETERS LÀ GÌ?
# ============================================================

print("\n4. HYPERPARAMETERS LÀ GÌ?")
print("""
Hyperparameters là những giá trị do con người đặt ra trước hoặc trong quá trình training.

Khác với parameters:
- hyperparameters không được model tự học trực tiếp
- chúng do người làm mô hình lựa chọn

Ví dụ phổ biến:
- learning rate
- batch size
- số epoch
- số hidden units
- số hidden layers
""")


# ============================================================
# 5. SỰ KHÁC NHAU CỐT LÕI
# ============================================================

print("\n5. SỰ KHÁC NHAU CỐT LÕI")
print("""
Parameters:
- model học từ dữ liệu
- thay đổi liên tục trong training

Hyperparameters:
- do con người quyết định
- dùng để điều khiển cách model học

Nói đơn giản:
- parameters = cái model học được
- hyperparameters = cách ta thiết kế / điều khiển quá trình học
""")


# ============================================================
# 6. VÍ DỤ CỤ THỂ TRONG NEURAL NETWORK
# ============================================================

print("\n6. VÍ DỤ CỤ THỂ TRONG NEURAL NETWORK")
print("""
Giả sử bạn có một mạng neural:

Input: 784
Hidden: 128
Output: 10

Trong đó:

Parameters:
- toàn bộ weights nối giữa các layer
- toàn bộ bias ở các neuron

Hyperparameters:
- số hidden units = 128
- learning rate = 0.001
- batch size = 32
- số epoch = 20
""")


# ============================================================
# 7. WEIGHT VÀ BIAS LÀ PARAMETERS
# ============================================================

print("\n7. WEIGHT VÀ BIAS LÀ PARAMETERS")

import numpy as np

W = np.random.randn(3, 4)
b = np.random.randn(3)

print("W shape =", W.shape)
print("b shape =", b.shape)

print("""
Các giá trị trong W và b:
- ban đầu thường khởi tạo ngẫu nhiên
- sau đó được optimizer cập nhật dần dần

=> đây là parameters
""")


# ============================================================
# 8. LEARNING RATE LÀ HYPERPARAMETER
# ============================================================

print("\n8. LEARNING RATE LÀ HYPERPARAMETER")

learning_rate = 0.001

print("Learning rate =", learning_rate)
print("""
Learning rate là hyperparameter vì:
- model không tự học ra learning rate
- ta phải chọn nó trước

Nếu learning rate quá lớn:
- dễ học không ổn định

Nếu learning rate quá nhỏ:
- học chậm
""")


# ============================================================
# 9. BATCH SIZE LÀ HYPERPARAMETER
# ============================================================

print("\n9. BATCH SIZE LÀ HYPERPARAMETER")

batch_size = 32

print("Batch size =", batch_size)
print("""
Batch size là số lượng mẫu dùng trong một lần tính gradient.

Đây là hyperparameter vì:
- nó do ta chọn
- không phải model tự học

Ví dụ:
- batch size nhỏ: update thường xuyên hơn
- batch size lớn: ổn định hơn nhưng tốn bộ nhớ hơn
""")


# ============================================================
# 10. SỐ EPOCH LÀ HYPERPARAMETER
# ============================================================

print("\n10. SỐ EPOCH LÀ HYPERPARAMETER")

epochs = 20

print("Epochs =", epochs)
print("""
Epoch là số lần model đi qua toàn bộ training dataset.

Đây cũng là hyperparameter vì:
- ta quyết định train bao nhiêu epoch
- model không tự chọn số epoch cho mình
""")


# ============================================================
# 11. SỐ LAYER VÀ SỐ NEURON CŨNG LÀ HYPERPARAMETERS
# ============================================================

print("\n11. KIẾN TRÚC MẠNG CŨNG LÀ HYPERPARAMETER")
print("""
Một số hyperparameters liên quan đến kiến trúc model:

- số hidden layers
- số neuron mỗi layer
- loại activation function
- loại optimizer

Ví dụ:
- dùng 2 hidden layers hay 3 hidden layers?
- mỗi layer có 64 neuron hay 256 neuron?

Đây là những quyết định thiết kế mô hình.
""")


# ============================================================
# 12. TẠI SAO HYPERPARAMETERS QUAN TRỌNG?
# ============================================================

print("\n12. TẠI SAO HYPERPARAMETERS QUAN TRỌNG?")
print("""
Dù model có khả năng học parameters,
nhưng nếu hyperparameters chọn sai,
quá trình học vẫn có thể thất bại.

Ví dụ:
- learning rate quá lớn -> loss dao động
- batch size không phù hợp -> train kém hiệu quả
- số layer quá ít -> model yếu
- số layer quá nhiều -> khó train, dễ overfit
""")


# ============================================================
# 13. PARAMETERS ĐƯỢC HỌC NHƯ THẾ NÀO?
# ============================================================

print("\n13. PARAMETERS ĐƯỢC HỌC NHƯ THẾ NÀO?")
print("""
Quy trình:

1. forward pass
2. tính loss
3. backpropagation tính gradient
4. optimizer cập nhật weights và bias

Qua nhiều vòng lặp:
- parameters dần tốt hơn
- model dần học được pattern trong dữ liệu
""")


# ============================================================
# 14. HYPERPARAMETERS ĐƯỢC CHỌN NHƯ THẾ NÀO?
# ============================================================

print("\n14. HYPERPARAMETERS ĐƯỢC CHỌN NHƯ THẾ NÀO?")
print("""
Hyperparameters thường được chọn bằng:
- kinh nghiệm
- thử nghiệm
- tuning

Ví dụ:
- thử learning rate = 0.1, 0.01, 0.001
- thử batch size = 16, 32, 64
- thử số layer khác nhau

Mục tiêu:
- tìm ra cấu hình train hiệu quả nhất
""")


# ============================================================
# 15. SO SÁNH NHANH
# ============================================================

print("\n15. SO SÁNH NHANH")
print("""
Parameters:
- weights
- bias
- model tự học

Hyperparameters:
- learning rate
- batch size
- epochs
- số layer
- số neuron
- con người chọn
""")


# ============================================================
# 16. NHỮNG NHẦM LẪN HAY GẶP
# ============================================================

print("\n16. NHỮNG NHẦM LẪN HAY GẶP")
print("""
Nhầm lẫn 1:
Learning rate là parameter
-> Sai, đó là hyperparameter

Nhầm lẫn 2:
Weight là hyperparameter
-> Sai, weight là parameter

Nhầm lẫn 3:
Model tự học luôn hyperparameter
-> Sai trong đa số trường hợp cơ bản
""")


# ============================================================
# 17. KẾT NỐI SANG PHẦN TIẾP THEO
# ============================================================

print("\n17. KẾT NỐI SANG PHẦN TIẾP THEO")
print("""
Sau khi hiểu parameters và hyperparameters,
ta sẽ quay lại với dữ liệu đầu vào trong Computer Vision:

- ảnh xám biểu diễn thế nào?
- ảnh màu biểu diễn thế nào?
- preprocessing ra sao?
- đưa ảnh vào PyTorch như thế nào?

Đó là cụm kiến thức tiếp theo trong PPT.
""")


# ============================================================
# 18. TỔNG KẾT
# ============================================================

print("\n18. TỔNG KẾT")
print("""
1. Parameters là giá trị model tự học từ dữ liệu
2. Ví dụ parameter: weights, bias
3. Hyperparameters là giá trị do con người chọn
4. Ví dụ hyperparameter: learning rate, batch size, epochs
5. Parameters quyết định model đã học được gì
6. Hyperparameters quyết định model học như thế nào
""")