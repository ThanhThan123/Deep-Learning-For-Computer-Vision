"""
EX12 - Convolutional Neural Network (CNN)

Mục tiêu:
1. Hiểu CNN là gì
2. Hiểu vì sao CNN ra đời
3. Hiểu hạn chế của Linear / Fully Connected
4. Hiểu convolution hoạt động như thế nào
5. Hiểu cấu trúc CNN cơ bản
6. Hiểu CNN học feature từ ảnh ra sao

Tài liệu gốc:
- PPT Deep Learning / Computer Vision
"""


# ============================================================
# 1. CNN LÀ GÌ?
# ============================================================

print("1. CNN LÀ GÌ?")
print("""
CNN (Convolutional Neural Network) là một loại Neural Network
được thiết kế đặc biệt để xử lý dữ liệu dạng ảnh.

Khác với mạng fully connected:
- CNN tận dụng cấu trúc không gian của ảnh
- giúp học feature tốt hơn

CNN là nền tảng của hầu hết các bài toán:
- image classification
- object detection
- segmentation
""")


# ============================================================
# 2. VÌ SAO CẦN CNN?
# ============================================================

print("\n2. VÌ SAO CẦN CNN?")
print("""
Trong các ex trước:

- ta dùng Linear Classifier hoặc Fully Connected Network
- ảnh bị flatten thành vector

Vấn đề:
- mất thông tin không gian (spatial information)
- pixel gần nhau không còn "liên quan"

Ví dụ:
- pixel (0,0) và (0,1) trong ảnh
- khi flatten → bị tách rời

=> model không hiểu cấu trúc ảnh

CNN giải quyết vấn đề này.
""")


# ============================================================
# 3. CNN GIỮ LẠI CẤU TRÚC ẢNH
# ============================================================

print("\n3. CNN GIỮ LẠI CẤU TRÚC ẢNH")
print("""
CNN không flatten ảnh ngay từ đầu.

Thay vào đó:
- giữ nguyên cấu trúc 2D (hoặc 3D với RGB)
- áp dụng convolution trực tiếp trên ảnh

=> giúp model hiểu:
- cạnh (edges)
- hình dạng
- cấu trúc

Đây là điểm khác biệt quan trọng nhất so với Linear model.
""")


# ============================================================
# 4. CONVOLUTION LÀ GÌ?
# ============================================================

print("\n4. CONVOLUTION LÀ GÌ?")
print("""
Convolution là phép toán dùng một kernel (filter)
trượt qua ảnh để trích xuất đặc trưng.

Ý tưởng:
- lấy một vùng nhỏ của ảnh
- nhân với kernel
- cộng lại → tạo ra giá trị mới

Quá trình này lặp lại trên toàn bộ ảnh.
""")


# ============================================================
# 5. VÍ DỤ CONVOLUTION
# ============================================================

print("\n5. VÍ DỤ CONVOLUTION")

import numpy as np

image = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

kernel = np.array([
    [1, 0],
    [0, -1]
])

# convolution đơn giản
result = image[0:2, 0:2] * kernel
value = np.sum(result)

print("Image:\n", image)
print("Kernel:\n", kernel)
print("Result:\n", result)
print("Convolution output:", value)


# ============================================================
# 6. KERNEL (FILTER) HỌC GÌ?
# ============================================================

print("\n6. KERNEL HỌC GÌ?")
print("""
Kernel không phải cố định.

Trong CNN:
- kernel cũng là parameter
- được học trong quá trình training

Ví dụ:
- kernel có thể học phát hiện cạnh ngang
- kernel khác học cạnh dọc
- kernel khác học texture

=> CNN tự học feature từ dữ liệu
""")


# ============================================================
# 7. FEATURE MAP LÀ GÌ?
# ============================================================

print("\n7. FEATURE MAP LÀ GÌ?")
print("""
Sau khi áp dụng convolution,
ta thu được một ma trận mới gọi là feature map.

Feature map:
- chứa thông tin đã được trích xuất
- giúp model nhận diện pattern

Mỗi kernel → tạo ra 1 feature map

Nhiều kernel → nhiều feature map
""")


# ============================================================
# 8. CNN HỌC NHƯ THẾ NÀO?
# ============================================================

print("\n8. CNN HỌC NHƯ THẾ NÀO?")
print("""
Theo PPT, CNN học theo tầng:

Layer đầu:
- học edge (cạnh)

Layer giữa:
- học combination của edge

Layer sâu:
- học object parts

Layer cuối:
- nhận diện object hoàn chỉnh

Đây là ý rất quan trọng:
CNN học từ đơn giản → phức tạp
""")


# ============================================================
# 9. CẤU TRÚC CNN CƠ BẢN
# ============================================================

print("\n9. CẤU TRÚC CNN CƠ BẢN")
print("""
Một CNN cơ bản gồm:

1. Convolution layer
2. Activation (ReLU)
3. Pooling layer
4. Fully Connected layer

Pipeline:
Image -> Conv -> ReLU -> Pool -> FC -> Output
""")


# ============================================================
# 10. ACTIVATION TRONG CNN
# ============================================================

print("\n10. ACTIVATION TRONG CNN")
print("""
Sau convolution, ta áp dụng activation function (thường là ReLU).

Mục đích:
- tạo tính phi tuyến
- giúp model học pattern phức tạp hơn

ReLU:
f(x) = max(0, x)
""")


# ============================================================
# 11. POOLING LÀ GÌ?
# ============================================================

print("\n11. POOLING LÀ GÌ?")
print("""
Pooling là bước giảm kích thước feature map.

Phổ biến:
- Max pooling

Ví dụ:
- lấy giá trị lớn nhất trong vùng

Mục đích:
- giảm số lượng tham số
- giảm overfitting
- giữ lại thông tin quan trọng
""")


# ============================================================
# 12. FULLY CONNECTED TRONG CNN
# ============================================================

print("\n12. FULLY CONNECTED TRONG CNN")
print("""
Sau các lớp convolution và pooling:

- feature map được flatten
- đưa vào fully connected layer

=> giống Linear Classifier ở ex3

=> cuối cùng:
- output ra class
""")


# ============================================================
# 13. SO SÁNH CNN VS FULLY CONNECTED
# ============================================================

print("\n13. SO SÁNH CNN VS FULLY CONNECTED")
print("""
Fully Connected:
- flatten ảnh
- mất thông tin không gian

CNN:
- giữ cấu trúc ảnh
- học feature theo vùng
- hiệu quả hơn nhiều với ảnh

=> CNN là lựa chọn chuẩn cho Computer Vision
""")


# ============================================================
# 14. TẠI SAO CNN HIỆU QUẢ?
# ============================================================

print("\n14. TẠI SAO CNN HIỆU QUẢ?")
print("""
CNN hiệu quả vì:

1. Local connectivity:
- chỉ nhìn vùng nhỏ của ảnh

2. Shared weights:
- cùng kernel áp dụng toàn ảnh

3. Hierarchical learning:
- học từ đơn giản đến phức tạp

=> giúp model học tốt hơn và ít tham số hơn
""")


# ============================================================
# 15. CNN LIÊN QUAN GÌ ĐẾN CÁC EX TRƯỚC?
# ============================================================

print("\n15. LIÊN HỆ VỚI EX TRƯỚC")
print("""
Ex3:
- Linear Classifier

Ex4:
- Neuron + Activation

Ex5:
- Neural Network

Ex7:
- Optimization

Ex8:
- Backpropagation

=> CNN = Neural Network + cấu trúc đặc biệt cho ảnh
""")


# ============================================================
# 16. NHỮNG NHẦM LẪN HAY GẶP
# ============================================================

print("\n16. NHỮNG NHẦM LẪN HAY GẶP")
print("""
Nhầm lẫn 1:
CNN không cần activation
-> Sai

Nhầm lẫn 2:
CNN chỉ là convolution
-> Sai (còn pooling, FC, activation)

Nhầm lẫn 3:
Flatten ảnh ngay từ đầu
-> Sai (CNN giữ cấu trúc trước)

Nhầm lẫn 4:
Kernel là cố định
-> Sai (kernel được học)
""")


# ============================================================
# 17. KẾT NỐI SANG PHẦN TIẾP
# ============================================================

print("\n17. KẾT NỐI")
print("""
Sau khi hiểu CNN:

PPT sẽ chuyển sang:
- Image Classification thực tế
- Object Detection
- Segmentation

=> CNN là nền tảng cho tất cả các bài toán này
""")


# ============================================================
# 18. TỔNG KẾT
# ============================================================

print("\n18. TỔNG KẾT")
print("""
1. CNN là Neural Network chuyên cho ảnh
2. Không flatten ảnh ngay từ đầu
3. Dùng convolution để trích xuất feature
4. Kernel được học từ dữ liệu
5. CNN học từ edge → pattern → object
6. Là nền tảng của Computer Vision hiện đại
""")