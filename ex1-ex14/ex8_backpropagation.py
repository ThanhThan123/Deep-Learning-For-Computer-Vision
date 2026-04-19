"""
EX8 - Backpropagation

Mục tiêu:
1. Hiểu Backpropagation là gì
2. Hiểu vì sao cần Backpropagation trong Neural Network
3. Hiểu mối liên hệ giữa forward pass, loss và gradient
4. Hiểu cách gradient được truyền ngược từ output về input
5. Nắm được neural network learning process ở mức nền tảng

Tài liệu gốc:
- PPT Deep Learning / Computer Vision
"""


# ============================================================
# 1. BACKPROPAGATION LÀ GÌ?
# ============================================================

print("1. BACKPROPAGATION LÀ GÌ?")
print("""
Backpropagation là thuật toán dùng để tính gradient trong Neural Network.

Hiểu đơn giản:
- Forward pass: đi từ input đến output để tạo prediction
- Tính loss: đo model sai bao nhiêu
- Backpropagation: truyền lỗi ngược từ output về các layer trước
  để biết mỗi weight cần thay đổi thế nào

Nói ngắn gọn:
Backpropagation giúp model biết phải cập nhật tham số ra sao để giảm loss.
""")


# ============================================================
# 2. TẠI SAO CẦN BACKPROPAGATION?
# ============================================================

print("\n2. TẠI SAO CẦN BACKPROPAGATION?")
print("""
Nếu model chỉ có 1 lớp rất đơn giản,
ta có thể tính gradient khá trực tiếp.

Nhưng với Neural Network nhiều layer:
- output phụ thuộc vào hidden layers
- hidden layers phụ thuộc vào input và các weight trước đó

Vì vậy:
- ta không thể cập nhật từng tham số theo kiểu đoán mò
- ta cần một cách có hệ thống để tính gradient cho toàn bộ mạng

Backpropagation chính là lời giải cho bài toán đó.
""")


# ============================================================
# 3. Ý TƯỞNG CỐT LÕI CỦA BACKPROPAGATION
# ============================================================

print("\n3. Ý TƯỞNG CỐT LÕI CỦA BACKPROPAGATION")
print("""
Backpropagation dựa trên Chain Rule (quy tắc dây chuyền) trong đạo hàm.

Ý tưởng:
- output cuối cùng phụ thuộc vào layer trước
- layer trước lại phụ thuộc vào layer trước nữa
- cứ như vậy cho tới input

Nên để tính gradient của loss theo một weight,
ta phải đi ngược qua các lớp và nhân các đạo hàm liên quan.

Đó là lý do gọi là:
Back-propagation = lan truyền ngược
""")


# ============================================================
# 4. QUY TRÌNH HỌC CỦA MẠNG NEURAL
# ============================================================

print("\n4. QUY TRÌNH HỌC CỦA MẠNG NEURAL")
print("""
Quy trình học cơ bản của Neural Network:

Bước 1:
- đưa input vào mạng

Bước 2:
- thực hiện forward pass
- tạo prediction

Bước 3:
- tính loss từ prediction và target

Bước 4:
- dùng backpropagation để tính gradient

Bước 5:
- dùng optimizer để cập nhật weights

Bước 6:
- lặp lại rất nhiều lần

Đây chính là neural network's learning process.
""")


# ============================================================
# 5. FORWARD PASS VÀ BACKWARD PASS
# ============================================================

print("\n5. FORWARD PASS VÀ BACKWARD PASS")
print("""
Forward pass:
- dữ liệu đi từ input -> hidden -> output
- mục tiêu: tạo prediction

Backward pass:
- gradient đi từ output -> hidden -> input
- mục tiêu: tính xem mỗi weight ảnh hưởng đến loss bao nhiêu

Nói cách khác:
- forward để dự đoán
- backward để học
""")


# ============================================================
# 6. VÍ DỤ MẠNG NHỎ
# ============================================================

print("\n6. VÍ DỤ MẠNG NHỎ")

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# input
x = 1.0

# tham số
w1 = 0.5
b1 = 0.1
w2 = 0.8
b2 = 0.2

# forward
z1 = w1 * x + b1
a1 = sigmoid(z1)

z2 = w2 * a1 + b2
y_pred = sigmoid(z2)

y_true = 1.0

# loss đơn giản: squared error
loss = 0.5 * (y_pred - y_true) ** 2

print("z1 =", z1)
print("a1 =", a1)
print("z2 =", z2)
print("y_pred =", y_pred)
print("loss =", loss)


# ============================================================
# 7. BACKPROPAGATION TRONG VÍ DỤ TRÊN
# ============================================================

print("\n7. BACKPROPAGATION TRONG VÍ DỤ TRÊN")
print("""
Ta muốn biết:
- loss thay đổi thế nào theo w2?
- loss thay đổi thế nào theo w1?

Muốn vậy, ta phải đi ngược từ loss.

Ví dụ:
loss -> y_pred -> z2 -> a1 -> z1 -> w1

Đó chính là chuỗi phụ thuộc của các biến.
""")


# ============================================================
# 8. CHAIN RULE (QUY TẮC DÂY CHUYỀN)
# ============================================================

print("\n8. CHAIN RULE")
print("""
Chain Rule là nền tảng toán học của Backpropagation.

Ví dụ:
Nếu loss phụ thuộc vào y
và y phụ thuộc vào x

thì:
d(loss)/dx = d(loss)/dy * dy/dx

Trong Neural Network:
- output phụ thuộc layer trước
- layer trước phụ thuộc layer trước nữa
=> ta nối các đạo hàm lại theo chuỗi
""")


# ============================================================
# 9. TÍNH GRADIENT CHO LAYER CUỐI
# ============================================================

print("\n9. TÍNH GRADIENT CHO LAYER CUỐI")

# đạo hàm loss theo y_pred
dL_dy = y_pred - y_true

# đạo hàm sigmoid tại z2
dy_dz2 = y_pred * (1 - y_pred)

# gradient theo z2
dL_dz2 = dL_dy * dy_dz2

# gradient theo w2 và b2
dL_dw2 = dL_dz2 * a1
dL_db2 = dL_dz2

print("dL/dw2 =", dL_dw2)
print("dL/db2 =", dL_db2)


# ============================================================
# 10. TÍNH GRADIENT CHO LAYER TRƯỚC
# ============================================================

print("\n10. TÍNH GRADIENT CHO LAYER TRƯỚC")

# truyền gradient ngược về a1
dL_da1 = dL_dz2 * w2

# đạo hàm sigmoid tại z1
da1_dz1 = a1 * (1 - a1)

# gradient theo z1
dL_dz1 = dL_da1 * da1_dz1

# gradient theo w1 và b1
dL_dw1 = dL_dz1 * x
dL_db1 = dL_dz1

print("dL/dw1 =", dL_dw1)
print("dL/db1 =", dL_db1)


# ============================================================
# 11. Ý NGHĨA CỦA CÁC GRADIENT
# ============================================================

print("\n11. Ý NGHĨA CỦA CÁC GRADIENT")
print("""
Mỗi gradient cho biết:

- nếu thay đổi weight đó một chút
- thì loss sẽ thay đổi thế nào

Nếu gradient lớn:
- weight đó ảnh hưởng mạnh đến loss

Nếu gradient nhỏ:
- weight đó ảnh hưởng ít hơn

Optimizer sẽ dùng những gradient này để update weights.
""")


# ============================================================
# 12. LOCAL GRADIENT VÀ UPSTREAM GRADIENT
# ============================================================

print("\n12. LOCAL GRADIENT VÀ UPSTREAM GRADIENT")
print("""
Một cách hiểu rất quan trọng trong backpropagation:

Upstream gradient:
- gradient đi từ phía sau truyền về

Local gradient:
- đạo hàm tại node / layer hiện tại

Gradient mới:
= upstream gradient * local gradient

Đây là cách rất trực quan để hiểu backprop:
- layer sau gửi gradient về
- layer hiện tại nhân thêm đạo hàm của mình
- rồi truyền tiếp về layer trước
""")


# ============================================================
# 13. TẠI SAO BACKPROPAGATION HIỆU QUẢ?
# ============================================================

print("\n13. TẠI SAO BACKPROPAGATION HIỆU QUẢ?")
print("""
Nếu tính gradient cho từng weight một cách riêng lẻ,
chi phí sẽ rất lớn.

Backpropagation hiệu quả vì:
- tái sử dụng các kết quả trung gian
- tính gradient có hệ thống
- phù hợp với mạng nhiều layer

Nhờ đó:
- Neural Network mới có thể train thực tế được
""")


# ============================================================
# 14. BACKPROPAGATION KHÔNG PHẢI OPTIMIZER
# ============================================================

print("\n14. BACKPROPAGATION KHÔNG PHẢI OPTIMIZER")
print("""
Đây là chỗ người mới học rất hay nhầm:

Backpropagation:
- dùng để tính gradient

Optimizer:
- dùng gradient đó để cập nhật tham số

Ví dụ:
- backprop cho ra dL/dw
- gradient descent dùng dL/dw để update w

Tức là:
- backprop = tính đạo hàm
- optimizer = cập nhật weights
""")


# ============================================================
# 15. LIÊN HỆ VỚI FORWARD PROPAGATION
# ============================================================

print("\n15. LIÊN HỆ VỚI FORWARD PROPAGATION")
print("""
Forward propagation và Backpropagation luôn đi cùng nhau:

- Forward:
  tính output và loss

- Backward:
  tính gradient từ loss

Không có forward:
- không có prediction
- không có loss

Không có backward:
- model không biết update thế nào

Hai bước này tạo thành vòng học cơ bản của Neural Network.
""")


# ============================================================
# 16. NHỮNG NHẦM LẪN HAY GẶP
# ============================================================

print("\n16. NHỮNG NHẦM LẪN HAY GẶP")
print("""
Nhầm lẫn 1:
Backpropagation là một layer
-> Sai
Nó là thuật toán tính gradient

Nhầm lẫn 2:
Backpropagation là quá trình update weight
-> Không hoàn toàn đúng
Nó tính gradient, còn update là việc của optimizer

Nhầm lẫn 3:
Gradient luôn truyền xuôi
-> Sai
Trong backprop, gradient truyền ngược từ output về input
""")


# ============================================================
# 17. MỐI LIÊN HỆ GIỮA BACKPROPAGATION VÀ LEARNING
# ============================================================

print("\n17. MỐI LIÊN HỆ GIỮA BACKPROPAGATION VÀ LEARNING")
print("""
Model học được vì:

- forward pass tạo prediction
- loss đo prediction sai bao nhiêu
- backpropagation chỉ ra mỗi weight góp phần vào sai số đó thế nào
- optimizer sửa weight
- lặp lại nhiều lần

Qua thời gian:
- loss giảm dần
- prediction tốt hơn
- model học được pattern trong dữ liệu
""")


# ============================================================
# 18. TỔNG KẾT
# ============================================================

print("\n18. TỔNG KẾT")
print("""
1. Backpropagation là thuật toán tính gradient trong mạng nhiều layer
2. Nó dựa trên Chain Rule
3. Forward pass dùng để dự đoán, backward pass dùng để học
4. Gradient được truyền ngược từ output về input
5. Backpropagation không phải optimizer, mà là bước tính gradient
6. Đây là nền tảng để Neural Network học được
""")