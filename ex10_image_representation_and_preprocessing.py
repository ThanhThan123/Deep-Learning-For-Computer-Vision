"""
EX10 - Image Representation và Data Preprocessing

Mục tiêu:
1. Hiểu ảnh được biểu diễn như thế nào trong Computer Vision
2. Phân biệt ảnh grayscale và ảnh màu
3. Hiểu pixel là gì
4. Hiểu shape của ảnh
5. Hiểu tại sao cần preprocessing
6. Biết các bước preprocessing cơ bản trước khi train model

Tài liệu gốc:
- PPT Deep Learning / Computer Vision
"""


# ============================================================
# 1. TỔNG QUAN
# ============================================================

print("1. TỔNG QUAN")
print("""
Trong Computer Vision, máy tính không nhìn ảnh như con người.

Con người:
- nhìn ảnh và hiểu nội dung trực quan

Máy tính:
- chỉ thấy ảnh như dữ liệu số

Vì vậy:
- trước khi học model
- ta phải hiểu ảnh được biểu diễn dưới dạng nào
- và cần xử lý ảnh ra sao trước khi đưa vào mô hình
""")


# ============================================================
# 2. PIXEL LÀ GÌ?
# ============================================================

print("\n2. PIXEL LÀ GÌ?")
print("""
Pixel là đơn vị nhỏ nhất của ảnh số.

Mỗi ảnh được tạo thành từ rất nhiều pixel.
Mỗi pixel mang thông tin về độ sáng hoặc màu sắc.

Nói đơn giản:
- ảnh là tập hợp các pixel
- máy tính lưu trữ ảnh bằng các giá trị số của pixel đó
""")


# ============================================================
# 3. ẢNH GRAYSCALE ĐƯỢC BIỂU DIỄN NHƯ THẾ NÀO?
# ============================================================

print("\n3. ẢNH GRAYSCALE")
print("""
Theo PPT, ảnh grayscale được biểu diễn dưới dạng ma trận 2 chiều.

Shape:
(height, width)

Ví dụ:
- ảnh 28 x 28
- shape = (28, 28)

Ý nghĩa:
- mỗi phần tử trong ma trận là một pixel
- pixel đó biểu diễn độ sáng

Thường:
- 0 là đen
- 255 là trắng
- các giá trị ở giữa là các mức xám
""")


import numpy as np

gray_image = np.array([
    [0, 50, 100],
    [150, 200, 255]
])

print("Ví dụ ảnh grayscale:")
print(gray_image)
print("Shape:", gray_image.shape)


# ============================================================
# 4. VÌ SAO ẢNH GRAYSCALE LÀ MA TRẬN 2D?
# ============================================================

print("\n4. VÌ SAO ẢNH GRAYSCALE LÀ MA TRẬN 2D?")
print("""
Vì ảnh grayscale chỉ có 1 mức sáng cho mỗi pixel.

Nghĩa là:
- không cần nhiều kênh màu
- chỉ cần 2 chiều:
  + chiều cao
  + chiều rộng

Nói cách khác:
mỗi vị trí (i, j) trong ảnh chỉ cần 1 số duy nhất để mô tả pixel đó.
""")


# ============================================================
# 5. ẢNH MÀU ĐƯỢC BIỂU DIỄN NHƯ THẾ NÀO?
# ============================================================

print("\n5. ẢNH MÀU")
print("""
Theo PPT, ảnh màu được biểu diễn dưới dạng mảng / tensor 3 chiều.

Shape thường là:
(height, width, channels)

Ví dụ:
- ảnh RGB 224 x 224
- shape = (224, 224, 3)

Trong đó:
- height: chiều cao
- width: chiều rộng
- channels: số kênh màu
""")


# ============================================================
# 6. RGB LÀ GÌ?
# ============================================================

print("\n6. RGB LÀ GÌ?")
print("""
RGB là viết tắt của:
- Red
- Green
- Blue

Một pixel màu thường có 3 giá trị:
(R, G, B)

Ví dụ:
- (255, 0, 0)   -> đỏ
- (0, 255, 0)   -> xanh lá
- (0, 0, 255)   -> xanh dương
- (255, 255, 255) -> trắng
- (0, 0, 0)       -> đen
""")

rgb_pixel = np.array([255, 0, 0])
print("Ví dụ pixel RGB màu đỏ:", rgb_pixel)


# ============================================================
# 7. VÌ SAO ẢNH MÀU LÀ 3D?
# ============================================================

print("\n7. VÌ SAO ẢNH MÀU LÀ 3D?")
print("""
Vì với ảnh màu, mỗi pixel không chỉ có 1 giá trị,
mà có nhiều giá trị màu.

Cụ thể với RGB:
- mỗi pixel có 3 số
- nên ngoài height và width,
  ta cần thêm chiều channels

Do đó:
- grayscale -> 2D
- color image -> 3D
""")


# ============================================================
# 8. SHAPE CỦA ẢNH QUAN TRỌNG NHƯ THẾ NÀO?
# ============================================================

print("\n8. SHAPE CỦA ẢNH QUAN TRỌNG NHƯ THẾ NÀO?")
print("""
Shape rất quan trọng trong Deep Learning.

Ví dụ:
- grayscale: (28, 28)
- RGB: (224, 224, 3)

Nếu truyền sai shape:
- model có thể báo lỗi
- hoặc học sai

Nói cách khác:
- hiểu đúng shape là điều bắt buộc khi làm Computer Vision
""")


# ============================================================
# 9. MÁY TÍNH KHÔNG HIỂU ẢNH, CHỈ HIỂU SỐ
# ============================================================

print("\n9. MÁY TÍNH CHỈ HIỂU SỐ")
print("""
Đây là ý rất quan trọng:

Con người nhìn ảnh thấy:
- mèo
- chó
- cây
- xe

Máy tính thì không.

Máy tính chỉ thấy:
- ma trận số
- tensor số

Từ những con số đó,
model mới học ra quy luật để phân loại hoặc phát hiện đối tượng.
""")


# ============================================================
# 10. DATA PREPROCESSING LÀ GÌ?
# ============================================================

print("\n10. DATA PREPROCESSING LÀ GÌ?")
print("""
Data preprocessing là bước xử lý dữ liệu trước khi đưa vào model.

Mục tiêu:
- đưa dữ liệu về dạng phù hợp
- giúp model học ổn định hơn
- giúp training hiệu quả hơn

PPT nhấn mạnh phần preprocessing như một bước cần thiết
trước khi huấn luyện mạng neural.
""")


# ============================================================
# 11. TẠI SAO PHẢI PREPROCESSING?
# ============================================================

print("\n11. TẠI SAO PHẢI PREPROCESSING?")
print("""
Dữ liệu ảnh ngoài thực tế thường:
- kích thước không đồng nhất
- giá trị pixel lớn nhỏ khác nhau
- có nhiễu
- có phân phối dữ liệu chưa phù hợp

Nếu đưa trực tiếp vào model:
- model có thể học kém
- train chậm
- không ổn định

Preprocessing giúp chuẩn hóa dữ liệu trước khi học.
""")


# ============================================================
# 12. CÁC BƯỚC PREPROCESSING CƠ BẢN
# ============================================================

print("\n12. CÁC BƯỚC PREPROCESSING CƠ BẢN")
print("""
Một số bước preprocessing thường gặp:

1. Resize ảnh
- đưa ảnh về cùng kích thước

2. Chuyển kiểu dữ liệu
- ví dụ từ ảnh sang mảng số / tensor

3. Scale / Normalize pixel
- đưa giá trị pixel về khoảng phù hợp

4. Có thể bổ sung:
- augmentation
- standardization
- center dữ liệu

Trong PPT, phần minh họa preprocessing tập trung vào ý tưởng:
- dữ liệu gốc
- zero-centered data
- normalized data
""")


# ============================================================
# 13. NORMALIZATION LÀ GÌ?
# ============================================================

print("\n13. NORMALIZATION LÀ GÌ?")
print("""
Normalization là bước đưa giá trị pixel về khoảng phù hợp hơn.

Ví dụ thường gặp:
- từ [0, 255] -> [0, 1]

Cách làm đơn giản:
pixel_normalized = pixel / 255

Mục đích:
- giảm độ lớn số đầu vào
- giúp model học ổn định hơn
- tối ưu dễ hơn
""")


pixel = 128
pixel_normalized = pixel / 255.0

print("Pixel gốc:", pixel)
print("Pixel sau normalize:", pixel_normalized)


# ============================================================
# 14. ZERO-CENTERED DATA LÀ GÌ?
# ============================================================

print("\n14. ZERO-CENTERED DATA LÀ GÌ?")
print("""
Zero-centered data là dữ liệu được dịch chuyển để có trung bình gần 0.

Ví dụ:
- thay vì chỉ nằm trong [0,1]
- dữ liệu có thể được trừ đi mean
- để phân phối quanh 0

Ý nghĩa:
- giúp việc tối ưu thuận lợi hơn trong nhiều trường hợp
- gradient descent có thể hoạt động ổn định hơn
""")


# Ví dụ minh họa zero-center
x = np.array([1.0, 2.0, 3.0, 4.0])
x_centered = x - np.mean(x)

print("Dữ liệu gốc:", x)
print("Dữ liệu zero-centered:", x_centered)


# ============================================================
# 15. NORMALIZED DATA LÀ GÌ?
# ============================================================

print("\n15. NORMALIZED DATA LÀ GÌ?")
print("""
Normalized data thường là dữ liệu đã được scale về một khoảng chuẩn
hoặc được chuẩn hóa theo mean và standard deviation.

Ví dụ:
x_norm = (x - mean) / std

Mục tiêu:
- dữ liệu giữa các chiều có thang đo phù hợp hơn
- giúp quá trình học hiệu quả hơn
""")


x = np.array([10.0, 20.0, 30.0, 40.0])
x_norm = (x - np.mean(x)) / np.std(x)

print("Dữ liệu gốc:", x)
print("Dữ liệu normalized:", x_norm)


# ============================================================
# 16. SỰ KHÁC NHAU GIỮA RAW DATA, ZERO-CENTERED, NORMALIZED
# ============================================================

print("\n16. RAW DATA VS ZERO-CENTERED VS NORMALIZED")
print("""
Raw data:
- dữ liệu gốc, chưa xử lý

Zero-centered:
- dữ liệu được dịch để trung bình quanh 0

Normalized:
- dữ liệu được scale / chuẩn hóa mạnh hơn để dễ học hơn

Đây là ý quan trọng trong slide preprocessing:
model thường không học tốt nhất trên dữ liệu thô.
""")


# ============================================================
# 17. PREPROCESSING CÓ PHẢI LÀM ĐẸP ẢNH KHÔNG?
# ============================================================

print("\n17. PREPROCESSING CÓ PHẢI LÀM ĐẸP ẢNH KHÔNG?")
print("""
Không.

Preprocessing không phải chủ yếu để ảnh đẹp hơn cho con người nhìn.

Mục tiêu chính là:
- làm dữ liệu phù hợp hơn cho model
- giúp máy học tốt hơn

Đây là khác biệt rất quan trọng giữa:
- xử lý ảnh cho người xem
- xử lý ảnh cho Deep Learning
""")


# ============================================================
# 18. NHỮNG NHẦM LẪN HAY GẶP
# ============================================================

print("\n18. NHỮNG NHẦM LẪN HAY GẶP")
print("""
Nhầm lẫn 1:
Ảnh màu và ảnh xám đều là 2D
-> Sai
Ảnh màu thường là 3D

Nhầm lẫn 2:
Máy tính nhìn ảnh như con người
-> Sai
Máy tính chỉ thấy dữ liệu số

Nhầm lẫn 3:
Normalization là không cần thiết
-> Sai
Đây là bước rất quan trọng trong training

Nhầm lẫn 4:
Preprocessing chỉ để cho đẹp
-> Sai
Mục tiêu chính là phục vụ model
""")


# ============================================================
# 19. KẾT NỐI SANG PHẦN TIẾP THEO
# ============================================================

print("\n19. KẾT NỐI SANG PHẦN TIẾP THEO")
print("""
Sau khi hiểu:
- ảnh được biểu diễn thế nào
- preprocessing là gì

câu hỏi tiếp theo là:
- trong PyTorch, dữ liệu đó được biểu diễn dưới dạng gì?
- đưa ảnh thành tensor ra sao?
- xây dataset như thế nào?

Đó là phần tiếp theo trong PPT.
""")


# ============================================================
# 20. TỔNG KẾT
# ============================================================

print("\n20. TỔNG KẾT")
print("""
1. Ảnh grayscale thường là ma trận 2D
2. Ảnh màu RGB thường là tensor 3D
3. Pixel là đơn vị cơ bản của ảnh số
4. Máy tính chỉ hiểu ảnh dưới dạng số
5. Preprocessing giúp dữ liệu phù hợp hơn với model
6. Zero-centered và normalized data giúp training ổn định hơn
""")