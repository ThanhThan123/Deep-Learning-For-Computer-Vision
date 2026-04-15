"""
EX6 - Loss Function

Mục tiêu:
1. Hiểu Loss là gì và tại sao cần
2. Hiểu sự khác nhau giữa prediction và target
3. Nắm các loss phổ biến:
   - MAE
   - MSE
   - Huber
   - Cross-Entropy
4. Hiểu vai trò của loss trong training

Tài liệu gốc:
- PPT Deep Learning / Computer Vision
"""


# ============================================================
# 1. LOSS FUNCTION LÀ GÌ?
# ============================================================

print("1. LOSS FUNCTION LÀ GÌ?")
print("""
Loss Function là hàm đo độ sai lệch giữa:
- dự đoán của model (prediction)
- giá trị thật (target / ground truth)

=> Loss càng nhỏ → model càng tốt
=> Loss càng lớn → model dự đoán càng sai
""")


# ============================================================
# 2. PREDICTION VS TARGET
# ============================================================

print("\n2. PREDICTION VS TARGET")

print("""
Prediction:
- output của model

Target:
- giá trị đúng

Ví dụ:
Prediction = 8
Target = 10

=> model sai 2 đơn vị
=> loss sẽ đo mức độ sai này
""")


# ============================================================
# 3. VÌ SAO CẦN LOSS?
# ============================================================

print("\n3. VÌ SAO CẦN LOSS?")

print("""
Model cần biết:
- nó đang đúng hay sai
- sai bao nhiêu

Loss giúp:
- đánh giá model
- làm tín hiệu để cập nhật trọng số

=> Loss là nền tảng của quá trình học
""")


# ============================================================
# 4. REGRESSION LOSS
# ============================================================

print("\n4. REGRESSION LOSS")

print("""
Regression là bài toán dự đoán giá trị liên tục.

Ví dụ:
- dự đoán giá nhà
- dự đoán nhiệt độ

Trong PPT có 3 loại loss chính:
- MAE
- MSE
- Huber
""")


# ============================================================
# 5. MAE (Mean Absolute Error)
# ============================================================

import numpy as np

print("\n5. MAE")

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

print("""
MAE = trung bình |y_true - y_pred|

Ưu điểm:
- dễ hiểu
- không bị ảnh hưởng quá mạnh bởi outlier

Nhược điểm:
- không nhấn mạnh lỗi lớn
""")

y_true = np.array([10, 20, 30])
y_pred = np.array([12, 18, 25])

print("MAE =", mae(y_true, y_pred))


# ============================================================
# 6. MSE (Mean Squared Error)
# ============================================================

print("\n6. MSE")

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

print("""
MSE = trung bình (y_true - y_pred)^2

Ưu điểm:
- phạt nặng lỗi lớn (do bình phương)

Nhược điểm:
- rất nhạy với outlier
""")

print("MSE =", mse(y_true, y_pred))


# ============================================================
# 7. SO SÁNH MAE VS MSE
# ============================================================

print("\n7. SO SÁNH MAE VS MSE")

print("""
MAE:
- tuyến tính
- ít nhạy với outlier

MSE:
- bình phương lỗi
- nhấn mạnh lỗi lớn

=> chọn tùy bài toán
""")


# ============================================================
# 8. HUBER LOSS
# ============================================================

print("\n8. HUBER LOSS")

def huber(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    condition = np.abs(error) <= delta
    squared = 0.5 * error**2
    linear = delta * (np.abs(error) - 0.5 * delta)
    return np.mean(np.where(condition, squared, linear))

print("""
Huber Loss kết hợp:
- MSE (khi lỗi nhỏ)
- MAE (khi lỗi lớn)

Ưu điểm:
- ổn định hơn MSE
- ít bị outlier hơn
""")

print("Huber =", huber(y_true, y_pred))


# ============================================================
# 9. CLASSIFICATION LOSS
# ============================================================

print("\n9. CLASSIFICATION LOSS")

print("""
Classification là bài toán phân loại.

Output của model:
- thường là xác suất

Loss phổ biến nhất:
- Cross-Entropy
""")


# ============================================================
# 10. CROSS-ENTROPY
# ============================================================

print("\n10. CROSS-ENTROPY")

def cross_entropy(y_true, y_pred):
    epsilon = 1e-9  # tránh log(0)
    return -np.sum(y_true * np.log(y_pred + epsilon))

print("""
Cross-Entropy đo sự khác biệt giữa:
- phân phối xác suất dự đoán
- phân phối xác suất thật

Ví dụ:
y_true = [1, 0, 0]
y_pred = [0.7, 0.2, 0.1]

=> model đo xem dự đoán có gần đúng không
""")


y_true = np.array([1, 0, 0])
y_pred = np.array([0.7, 0.2, 0.1])

print("Cross-Entropy =", cross_entropy(y_true, y_pred))


# ============================================================
# 11. HIỂU TRỰC QUAN CROSS-ENTROPY
# ============================================================

print("\n11. HIỂU TRỰC QUAN")

print("""
Nếu model dự đoán đúng:
- xác suất class đúng cao
=> loss nhỏ

Nếu model sai:
- xác suất class đúng thấp
=> loss lớn

=> mục tiêu:
tăng xác suất đúng → giảm loss
""")


# ============================================================
# 12. LOSS LIÊN QUAN ĐẾN TRAINING NHƯ THẾ NÀO?
# ============================================================

print("\n12. LOSS VÀ TRAINING")

print("""
Quy trình học:

1. Model dự đoán (forward pass)
2. Tính loss
3. Cập nhật weight để giảm loss

=> lặp lại nhiều lần

=> model học dần dần
""")


# ============================================================
# 13. NHỮNG NHẦM LẪN HAY GẶP
# ============================================================

print("\n13. NHỮNG NHẦM LẪN")

print("""
Nhầm lẫn 1:
Loss càng lớn càng tốt
-> Sai (loss phải nhỏ)

Nhầm lẫn 2:
Accuracy = Loss
-> Sai (2 khái niệm khác nhau)

Nhầm lẫn 3:
Loss chỉ dùng để đánh giá
-> Sai (loss còn dùng để training)
""")


# ============================================================
# 14. KẾT NỐI SANG PHẦN TIẾP
# ============================================================

print("\n14. KẾT NỐI")

print("""
Sau khi có loss:

Câu hỏi tiếp theo:
- làm sao giảm loss?

=> PPT chuyển sang:
GRADIENT DESCENT & OPTIMIZATION
""")


# ============================================================
# 15. TỔNG KẾT
# ============================================================

print("\n15. TỔNG KẾT")

print("""
1. Loss đo độ sai giữa prediction và target
2. Regression dùng:
   - MAE
   - MSE
   - Huber
3. Classification dùng:
   - Cross-Entropy
4. Loss là nền tảng của quá trình học
5. Mục tiêu: minimize loss
""")