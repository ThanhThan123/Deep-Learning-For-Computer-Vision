"""
EX5 - Image Classification: Ảnh được biểu diễn như thế nào?

Mục tiêu:
1. Hiểu ảnh trong máy tính thực chất là gì
2. Phân biệt ảnh grayscale và ảnh màu
3. Hiểu shape của ảnh (H, W, C)
4. Hiểu channel-first và channel-last
5. Hiểu vì sao phải normalize pixel
6. Nắm nền tảng để học CNN và Deep Learning

Ghi chú:
- Đây là phần nền tảng quan trọng nhất khi làm việc với ảnh
- Nếu hiểu sai phần này → model sẽ sai từ đầu
"""


# ============================================================
# 1. MÁY TÍNH NHÌN ẢNH NHƯ THẾ NÀO?
# ============================================================

print("1. MÁY TÍNH NHÌN ẢNH NHƯ THẾ NÀO?")
print("""
Con người:
- nhìn ảnh → hiểu ngay đó là mèo, chó, xe...

Máy tính:
- KHÔNG nhìn thấy ảnh như con người
- chỉ thấy một tập hợp các con số

=> Ảnh trong máy tính thực chất là:
   - một ma trận (matrix)
   hoặc
   - một tensor (multi-dimensional array)
""")


# ============================================================
# 2. ẢNH ĐEN TRẮNG (GRAYSCALE)
# ============================================================

print("2. ẢNH GRAYSCALE")
print("""
Ảnh grayscale là ảnh chỉ có 1 kênh màu.

Biểu diễn:
- shape = (height, width)

Ví dụ:
- ảnh 28x28 → shape = (28, 28)

Mỗi pixel có giá trị:
- 0   → màu đen
- 255 → màu trắng

Các giá trị ở giữa → màu xám
""")


# Ví dụ minh họa
import numpy as np

grayscale_image = np.array([
    [0, 50, 100],
    [150, 200, 255]
])

print("\nVí dụ ảnh grayscale:")
print(grayscale_image)


# ============================================================
# 3. ẢNH MÀU (RGB)
# ============================================================

print("\n3. ẢNH MÀU RGB")
print("""
Ảnh màu có 3 kênh:
- Red
- Green
- Blue

Biểu diễn:
- shape = (height, width, channels)

Ví dụ:
- shape = (224, 224, 3)

=> mỗi pixel có 3 giá trị:
   (R, G, B)
""")


# Ví dụ pixel RGB
rgb_pixel = [255, 0, 0]  # màu đỏ
print("\nVí dụ pixel RGB (đỏ):", rgb_pixel)


# ============================================================
# 4. SHAPE CỦA ẢNH
# ============================================================

print("\n4. SHAPE CỦA ẢNH")
print("""
Shape ảnh rất quan trọng trong Deep Learning.

Grayscale:
- (H, W)

RGB:
- (H, W, C)

Ví dụ:
- ảnh 64x64 RGB → (64, 64, 3)

Ý nghĩa:
- H: chiều cao
- W: chiều rộng
- C: số kênh màu
""")


# ============================================================
# 5. CHANNEL-FIRST VS CHANNEL-LAST
# ============================================================

print("\n5. CHANNEL-FIRST VS CHANNEL-LAST")
print("""
Có 2 cách lưu ảnh phổ biến:

1. Channel-last:
   (H, W, C)
   Ví dụ: (224, 224, 3)
   -> dùng trong NumPy, TensorFlow

2. Channel-first:
   (C, H, W)
   Ví dụ: (3, 224, 224)
   -> dùng trong PyTorch

Lưu ý:
- Nếu dùng sai format → model sẽ lỗi
""")


# ============================================================
# 6. VÌ SAO PHẢI NORMALIZE ẢNH?
# ============================================================

print("\n6. NORMALIZATION")
print("""
Pixel gốc:
- nằm trong khoảng [0, 255]

Trong Deep Learning:
- thường chuyển về [0, 1]
  bằng cách chia cho 255

Hoặc:
- [-1, 1]

Mục đích:
- giúp model học ổn định hơn
- giảm gradient quá lớn
- training nhanh hơn
""")


# ví dụ normalize
pixel = 128
normalized_pixel = pixel / 255

print("\nVí dụ normalize:")
print("Pixel gốc:", pixel)
print("Pixel sau normalize:", normalized_pixel)


# ============================================================
# 7. TẠI SAO PHẢI HIỂU ĐIỀU NÀY?
# ============================================================

print("\n7. TẠI SAO PHẢI HIỂU?")
print("""
Nếu không hiểu cách ảnh được biểu diễn:

- bạn sẽ không hiểu model đang học gì
- dễ truyền sai shape → lỗi
- không hiểu vì sao phải normalize
- khó debug model

Đây là nền tảng cho:
- CNN
- Image Classification
- Object Detection
- Segmentation
""")


# ============================================================
# 8. NHỮNG LỖI NGƯỜI MỚI HAY GẶP
# ============================================================

print("\n8. LỖI HAY GẶP")
print("""
Lỗi 1:
- Nhầm (H, W, C) và (C, H, W)

Lỗi 2:
- Quên normalize ảnh

Lỗi 3:
- Không hiểu pixel là gì

Lỗi 4:
- Truyền sai shape vào model

=> Đây là lỗi cực kỳ phổ biến khi mới học Deep Learning
""")


# ============================================================
# 9. TỔNG KẾT
# ============================================================

print("\n9. TỔNG KẾT")
print("""
1. Ảnh = ma trận / tensor số
2. Grayscale: (H, W)
3. RGB: (H, W, 3)
4. PyTorch dùng (C, H, W)
5. Pixel cần normalize
6. Hiểu đúng phần này là nền tảng của Computer Vision
""")