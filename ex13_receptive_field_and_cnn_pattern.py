"""
EX13 - Receptive Field, Dilated Convolution và Common CNN Layer Pattern

Mục tiêu:
1. Hiểu receptive field là gì
2. Hiểu effective receptive field là gì
3. Hiểu các cách tăng receptive field
4. Hiểu dilated / atrous convolution
5. Hiểu common CNN layer pattern trong thực tế
6. Hiểu vì sao nhiều kernel nhỏ thường tốt hơn một kernel lớn

Tài liệu gốc:
- PPT Deep Learning / Computer Vision
"""


# ============================================================
# 1. RECEPTIVE FIELD LÀ GÌ?
# ============================================================

print("1. RECEPTIVE FIELD LÀ GÌ?")
print("""
Receptive field là vùng của ảnh đầu vào mà một neuron hoặc một phần tử
trong feature map "nhìn thấy".

Hiểu đơn giản:
- một điểm ở layer sâu không nhìn toàn bộ ảnh ngay lập tức
- nó chỉ chịu ảnh hưởng bởi một vùng nào đó của ảnh gốc
- vùng đó gọi là receptive field

Ví dụ:
- nếu kernel 3x3 áp dụng lên ảnh
- thì mỗi output ban đầu chỉ nhìn thấy một vùng 3x3 của input
""")


# ============================================================
# 2. TẠI SAO RECEPTIVE FIELD QUAN TRỌNG?
# ============================================================

print("\n2. TẠI SAO RECEPTIVE FIELD QUAN TRỌNG?")
print("""
Nếu receptive field quá nhỏ:
- model chỉ nhìn thấy chi tiết rất cục bộ
- khó hiểu được cấu trúc lớn hơn của object

Nếu receptive field lớn hơn:
- model có thể nhìn được vùng rộng hơn
- hiểu ngữ cảnh tốt hơn
- nhận diện object tốt hơn

Đây là lý do receptive field rất quan trọng trong CNN.
""")


# ============================================================
# 3. RECEPTIVE FIELD TĂNG NHƯ THẾ NÀO KHI MẠNG SÂU HƠN?
# ============================================================

print("\n3. RECEPTIVE FIELD TĂNG NHƯ THẾ NÀO?")
print("""
Khi thêm nhiều convolution layer:
- receptive field của các layer sâu hơn sẽ lớn dần

Ví dụ trực quan:
- layer 1 nhìn vùng nhỏ
- layer 2 nhìn vùng lớn hơn
- layer 3 nhìn vùng lớn hơn nữa

Điều này giúp CNN học:
- layer đầu: edge
- layer giữa: shape / pattern
- layer sâu: object / object parts
""")


# ============================================================
# 4. EFFECTIVE RECEPTIVE FIELD LÀ GÌ?
# ============================================================

print("\n4. EFFECTIVE RECEPTIVE FIELD LÀ GÌ?")
print("""
PPT có phân biệt receptive field và effective receptive field.

Receptive field lý thuyết:
- là toàn bộ vùng input có thể ảnh hưởng đến một output neuron

Effective receptive field:
- là phần thực sự có ảnh hưởng mạnh nhất

Nói dễ hiểu:
- về mặt lý thuyết, neuron có thể nhìn một vùng khá lớn
- nhưng trong thực tế, phần ảnh hưởng mạnh nhất thường tập trung ở vùng trung tâm

=> effective receptive field thường nhỏ hơn cảm giác "toàn vùng lý thuyết"
""")


# ============================================================
# 5. CÁCH TĂNG RECEPTIVE FIELD
# ============================================================

print("\n5. CÁCH TĂNG RECEPTIVE FIELD")
print("""
Theo PPT, có nhiều cách tăng receptive field:

1. Add more Conv layers
   - làm mạng sâu hơn

2. Add pooling layers
   - sub-sampling để mỗi layer sau nhìn vùng lớn hơn

3. Use Dilated Conv
   - tăng vùng nhìn mà không cần kernel quá lớn

4. Use Depth-wise Conv
   - PPT có liệt kê như một lựa chọn trong phần tăng receptive field

Ý quan trọng:
- receptive field không chỉ tăng bằng cách tăng kernel size
- mà còn tăng nhờ kiến trúc mạng
""")  # based directly on PPT wording


# ============================================================
# 6. DILATED / ATROUS CONVOLUTION LÀ GÌ?
# ============================================================

print("\n6. DILATED / ATROUS CONVOLUTION LÀ GÌ?")
print("""
Dilated convolution (Atrous convolution) là convolution có "khoảng cách"
giữa các phần tử trong kernel.

Hiểu đơn giản:
- kernel không quét liền sát từng ô
- mà có khoảng trống ở giữa
- nhờ đó receptive field tăng lên

Ưu điểm:
- nhìn được vùng rộng hơn
- không cần tăng quá nhiều số parameter
- không cần pooling quá sớm

Đây là kỹ thuật rất hữu ích khi muốn:
- giữ độ phân giải feature map
- nhưng vẫn tăng vùng nhìn
""")


# ============================================================
# 7. VÍ DỤ TRỰC QUAN VỀ DILATED CONV
# ============================================================

print("\n7. VÍ DỤ TRỰC QUAN VỀ DILATED CONV")
print("""
Kernel 3x3 bình thường:
- nhìn một vùng 3x3

Kernel 3x3 với dilation:
- vẫn chỉ có 9 phần tử học được
- nhưng trải trên vùng lớn hơn

=> kết quả:
- receptive field tăng
- số parameter không tăng tương ứng như khi dùng kernel lớn thật

Đây là điểm mạnh rất lớn của dilated convolution.
""")


# ============================================================
# 8. COMMON CNN'S LAYER PATTERN (PART 1)
# ============================================================

print("\n8. COMMON CNN'S LAYER PATTERN (PART 1)")
print("""
Theo PPT, pattern phổ biến của CNN là:

1. Một vài block:
   Conv -> (BatchNorm) -> ReLU

2. Max-pooling

3. Lặp lại bước 1 và 2 cho đến khi feature map đủ bé

4. Flatten feature map

5. Một hoặc vài fully connected layer

Đây là dạng CNN rất phổ biến trong các bài toán classification cơ bản.
""")


# ============================================================
# 9. TẠI SAO PATTERN NÀY PHỔ BIẾN?
# ============================================================

print("\n9. TẠI SAO PATTERN NÀY PHỔ BIẾN?")
print("""
Vì mỗi thành phần có vai trò rõ ràng:

- Conv:
  trích xuất feature

- BatchNorm:
  giúp train ổn định hơn

- ReLU:
  tạo phi tuyến

- Max-pooling:
  giảm kích thước, tăng receptive field, giữ thông tin quan trọng

- Flatten + FC:
  biến feature map thành output classification

Đây là pipeline rất tự nhiên cho image classification.
""")


# ============================================================
# 10. COMMON CNN'S LAYER PATTERN (PART 2)
# ============================================================

print("\n10. COMMON CNN'S LAYER PATTERN (PART 2)")
print("""
PPT nhấn mạnh một ý rất quan trọng:

Nhiều Conv layer với filter/kernel bé
tốt hơn 1 Conv layer với filter/kernel lớn
trong nhiều trường hợp.

Ví dụ trong PPT:
- 2 lớp conv 3x3
- so với 1 lớp conv 5x5

Cả hai đều cho receptive field cuối là 5x5,
nhưng 2 lớp 3x3 có những ưu điểm riêng.
""")


# ============================================================
# 11. VÌ SAO 2 LỚP 3x3 THƯỜNG TỐT HƠN 1 LỚP 5x5?
# ============================================================

print("\n11. VÌ SAO 2 LỚP 3x3 THƯỜNG TỐT HƠN 1 LỚP 5x5?")
print("""
Theo PPT:

So sánh:
- 2 conv 3x3
- 1 conv 5x5

Receptive field cuối:
- đều là 5x5

Nhưng 2 conv 3x3 có:
- ít parameter hơn
- nhiều non-linear layer hơn

Ý nghĩa:
- ít parameter hơn -> đỡ tốn tài nguyên hơn
- nhiều non-linearity hơn -> học biểu diễn mạnh hơn

Đây là lý do các CNN hiện đại rất thích stack nhiều kernel nhỏ.
""")


# ============================================================
# 12. SO SÁNH THAM SỐ
# ============================================================

print("\n12. SO SÁNH THAM SỐ")

params_2_conv_3x3 = 2 * 3 * 3
params_1_conv_5x5 = 5 * 5

print("2 conv 3x3 -> số phần tử kernel =", params_2_conv_3x3)
print("1 conv 5x5 -> số phần tử kernel =", params_1_conv_5x5)

print("""
PPT minh họa rất rõ:
- 2 x 3x3 -> 18 đơn vị kernel
- 1 x 5x5 -> 25 đơn vị kernel

=> stack kernel nhỏ có thể tiết kiệm tham số hơn.
""")


# ============================================================
# 13. COMMON CNN'S LAYER PATTERN (PART 3)
# ============================================================

print("\n13. COMMON CNN'S LAYER PATTERN (PART 3)")
print("""
PPT còn đặt ra các câu hỏi thiết kế CNN architecture như:

- Mô hình nên có bao nhiêu layer?
- Mỗi layer nên có parameter như thế nào?
- Convolutional có bao nhiêu channel?
- Kernel size là bao nhiêu?
- Nên dùng activation function gì?
- Pooling kernel size là bao nhiêu?
- Nên flatten feature map ở đâu?
- Nên có bao nhiêu fully connected layer?
- Có nên dùng BatchNorm, Dropout...?

Đây là phần rất quan trọng vì nó cho thấy:
thiết kế CNN là một bài toán kiến trúc,
không chỉ là ghép layer ngẫu nhiên.
""")


# ============================================================
# 14. KHÔNG CÓ MỘT CÔNG THỨC DUY NHẤT CHO MỌI CNN
# ============================================================

print("\n14. KHÔNG CÓ MỘT CÔNG THỨC DUY NHẤT")
print("""
PPT không đưa ra một công thức cứng cho mọi CNN.

Thay vào đó, nó cho thấy:
- có những pattern phổ biến
- có những câu hỏi thiết kế cần cân nhắc

Điều đó có nghĩa:
- xây CNN là quá trình chọn kiến trúc hợp lý
- dựa trên dữ liệu, bài toán và tài nguyên
""")


# ============================================================
# 15. LIÊN HỆ VỚI IMAGE CLASSIFICATION
# ============================================================

print("\n15. LIÊN HỆ VỚI IMAGE CLASSIFICATION")
print("""
Tất cả các ý trên đều rất quan trọng cho classification vì:

- receptive field ảnh hưởng model nhìn được bao nhiêu ngữ cảnh
- dilated conv giúp tăng vùng nhìn
- common layer pattern giúp xây CNN hợp lý
- stack kernel nhỏ giúp model hiệu quả hơn

Nếu thiếu những ý này,
ta chỉ hiểu CNN ở mức rất nông.
""")


# ============================================================
# 16. NHỮNG NHẦM LẪN HAY GẶP
# ============================================================

print("\n16. NHỮNG NHẦM LẪN HAY GẶP")
print("""
Nhầm lẫn 1:
Kernel càng lớn thì luôn tốt hơn
-> Sai

Nhầm lẫn 2:
Receptive field chỉ tăng bằng cách tăng kernel size
-> Sai

Nhầm lẫn 3:
2 conv 3x3 và 1 conv 5x5 là hoàn toàn giống nhau
-> Sai
Chúng có cùng receptive field cuối,
nhưng khác số parameter và số non-linear layers

Nhầm lẫn 4:
CNN architecture chỉ là ghép Conv ngẫu nhiên
-> Sai
Thiết kế CNN cần có chiến lược
""")


# ============================================================
# 17. KẾT NỐI SANG PHẦN TIẾP THEO
# ============================================================

print("\n17. KẾT NỐI SANG PHẦN TIẾP THEO")
print("""
Sau khi hiểu:
- receptive field
- common CNN pattern
- dilated convolution

PPT đi tiếp sang các vấn đề quan trọng khi train CNN sâu:

- vanishing gradients
- exploding gradients
- weight initialization
- gradient clipping

Đó sẽ là Ex14.
""")


# ============================================================
# 18. TỔNG KẾT
# ============================================================

print("\n18. TỔNG KẾT")
print("""
1. Receptive field là vùng input mà một neuron nhìn thấy
2. Effective receptive field là phần thực sự ảnh hưởng mạnh nhất
3. Có nhiều cách tăng receptive field:
   - thêm conv layers
   - thêm pooling
   - dùng dilated conv
4. Dilated convolution giúp tăng vùng nhìn hiệu quả
5. Common CNN pattern thường là:
   Conv -> (BatchNorm) -> ReLU -> Pooling -> ... -> Flatten -> FC
6. Nhiều conv kernel nhỏ thường tốt hơn một kernel lớn trong nhiều trường hợp
""")