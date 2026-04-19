"""
EX4 - Các bài toán chính trong Computer Vision

Mục tiêu:
1. Hiểu 3 bài toán nền tảng trong Computer Vision
2. Phân biệt rõ:
   - Classification
   - Detection
   - Segmentation
3. Biết đầu vào và đầu ra của từng bài toán
4. Biết khi nào nên dùng từng loại bài toán
5. Tránh nhầm lẫn giữa 3 khái niệm này

Ghi chú:
- Đây là phần nền tảng rất quan trọng.
- Nếu hiểu chắc phần này, bạn sẽ dễ học hơn ở các bài CNN, Detection, Segmentation sau này.
"""


# ============================================================
# 1. TỔNG QUAN
# ============================================================

print("1. TỔNG QUAN")
print("""
Trong Computer Vision, có 3 bài toán rất quan trọng:

1. Classification
2. Detection
3. Segmentation

Ba bài toán này khác nhau chủ yếu ở:
- mức độ chi tiết của đầu ra
- lượng thông tin mô hình cần dự đoán
- mục đích sử dụng thực tế
""")


# ============================================================
# 2. CLASSIFICATION
# ============================================================

print("2. CLASSIFICATION")
print("""
Classification là bài toán phân loại ảnh.

Ý tưởng:
- Đưa vào một ảnh
- Mô hình trả về ảnh đó thuộc lớp nào

Ví dụ:
- Đây là mèo
- Đây là chó
- Đây là ô tô
- Đây là máy bay

Đầu vào:
- một ảnh

Đầu ra:
- một nhãn
hoặc
- xác suất của các nhãn

Ví dụ đầu ra:
- cat: 0.82
- dog: 0.15
- mug: 0.02
- hat: 0.01
""")


# ============================================================
# 3. CLASSIFICATION DÙNG KHI NÀO?
# ============================================================

print("3. CLASSIFICATION DÙNG KHI NÀO?")
print("""
Classification phù hợp khi:
- ảnh có một đối tượng chính
- chỉ cần biết ảnh thuộc lớp nào
- không cần biết vật thể nằm ở đâu

Ví dụ:
- ảnh này là mèo hay chó?
- ảnh X-quang này bất thường hay bình thường?
- ảnh lá cây này đang bị bệnh gì?
- ảnh sản phẩm này thuộc loại nào?

Đây thường là bài toán nhập môn dễ hiểu nhất trong Computer Vision.
""")


# ============================================================
# 4. HẠN CHẾ CỦA CLASSIFICATION
# ============================================================

print("4. HẠN CHẾ CỦA CLASSIFICATION")
print("""
Classification chỉ trả lời được:
- ảnh này là gì?

Nó không trả lời được:
- trong ảnh có bao nhiêu vật thể
- vật thể nằm ở đâu
- vật thể có hình dạng cụ thể như thế nào

Ví dụ:
Nếu ảnh có cả mèo và chó,
classification đơn giản thường không đủ để mô tả hết thông tin trong ảnh.
""")


# ============================================================
# 5. DETECTION
# ============================================================

print("5. DETECTION")
print("""
Detection là bài toán phát hiện đối tượng.

Khác với Classification:
- Classification chỉ phân loại ảnh
- Detection vừa phân loại, vừa định vị đối tượng

Nói cách khác:
Detection trả lời 3 câu hỏi:
1. Có những đối tượng nào trong ảnh?
2. Có bao nhiêu đối tượng?
3. Mỗi đối tượng nằm ở đâu?

Đầu ra của Detection thường gồm:
- class name
- bounding box
- confidence score
""")


# ============================================================
# 6. BOUNDING BOX LÀ GÌ?
# ============================================================

print("6. BOUNDING BOX LÀ GÌ?")
print("""
Bounding box là hình chữ nhật bao quanh đối tượng.

Nó thường được biểu diễn bằng:
- tọa độ góc trái trên và góc phải dưới
hoặc
- tâm (x, y), width, height

Ví dụ:
- con mèo ở vị trí này
- con chó ở vị trí kia

Detection không chỉ nói "có mèo",
mà còn chỉ ra mèo đang ở đâu trong ảnh.
""")


# ============================================================
# 7. DETECTION DÙNG KHI NÀO?
# ============================================================

print("7. DETECTION DÙNG KHI NÀO?")
print("""
Detection phù hợp khi:
- trong ảnh có nhiều đối tượng
- cần biết vị trí của từng đối tượng

Ví dụ:
- phát hiện người trong camera an ninh
- phát hiện xe trong giao thông
- phát hiện nón bảo hiểm
- phát hiện polyp trong nội soi
- phát hiện nốt phổi trong CT
- phát hiện lỗi sản phẩm trên dây chuyền

Detection rất phổ biến trong các hệ thống AI ứng dụng thực tế.
""")


# ============================================================
# 8. HẠN CHẾ CỦA DETECTION
# ============================================================

print("8. HẠN CHẾ CỦA DETECTION")
print("""
Detection biết được:
- vật thể là gì
- vật thể ở đâu

Nhưng Detection vẫn còn hạn chế:
- bounding box chỉ là khung chữ nhật
- không bám sát chính xác hình dạng của vật thể

Ví dụ:
- một con mèo có hình dạng cong
- nhưng bounding box vẫn chỉ là hình chữ nhật bao quanh

Nếu cần chính xác đến từng pixel,
Detection là chưa đủ.
""")


# ============================================================
# 9. SEGMENTATION
# ============================================================

print("9. SEGMENTATION")
print("""
Segmentation là bài toán gán nhãn cho từng pixel.

Thay vì chỉ vẽ một khung chữ nhật quanh đối tượng,
mô hình sẽ dự đoán pixel nào thuộc đối tượng nào.

Đầu vào:
- một ảnh

Đầu ra:
- một bản đồ nhãn theo pixel
  (pixel-level prediction)

Ví dụ:
- pixel nào là mèo
- pixel nào là chó
- pixel nào là nền
""")


# ============================================================
# 10. SEGMENTATION DÙNG KHI NÀO?
# ============================================================

print("10. SEGMENTATION DÙNG KHI NÀO?")
print("""
Segmentation phù hợp khi:
- cần độ chính xác vị trí rất cao
- cần biết hình dạng thật của đối tượng
- cần tách riêng đối tượng khỏi nền

Ví dụ:
- tách khối u trong ảnh y khoa
- tách polyp khỏi nền niêm mạc
- tách đường trong xe tự lái
- tách người khỏi background
- tách sản phẩm khỏi nền trong ảnh công nghiệp

Segmentation đặc biệt mạnh ở các bài toán y tế và ảnh cần đo đạc chính xác.
""")


# ============================================================
# 11. VÌ SAO SEGMENTATION KHÓ HƠN?
# ============================================================

print("11. VÌ SAO SEGMENTATION KHÓ HƠN?")
print("""
Segmentation khó hơn Classification và Detection vì:

1. Dự đoán đầu ra chi tiết hơn
   - không chỉ 1 nhãn
   - không chỉ 1 bounding box
   - mà là nhãn cho từng pixel

2. Yêu cầu dữ liệu gán nhãn khó hơn
   - phải vẽ mask chính xác
   - tốn công hơn nhiều so với classification hoặc detection

3. Mô hình thường phức tạp hơn
   - cần giữ lại thông tin không gian tốt
   - cần đầu ra chi tiết
""")


# ============================================================
# 12. SO SÁNH 3 BÀI TOÁN
# ============================================================

print("12. SO SÁNH 3 BÀI TOÁN")
print("""
Classification:
- biết ảnh là gì
- không biết vị trí
- không biết hình dạng chi tiết

Detection:
- biết có đối tượng gì
- biết vị trí bằng bounding box
- chưa biết chính xác từng pixel

Segmentation:
- biết có đối tượng gì
- biết vị trí
- biết chính xác vùng pixel của đối tượng

Mức độ chi tiết tăng dần:
Classification -> Detection -> Segmentation
""")


# ============================================================
# 13. SINGLE OBJECT VÀ MULTIPLE OBJECT
# ============================================================

print("13. SINGLE OBJECT VÀ MULTIPLE OBJECT")
print("""
Classification trong nhập môn thường gắn với single object:
- một ảnh có một đối tượng chính

Detection và Segmentation thường gắn với multiple objects:
- một ảnh có thể chứa nhiều vật thể khác nhau

Ví dụ:
- 1 ảnh có 1 mèo và 1 chó
- detection sẽ vẽ 2 bounding box
- segmentation sẽ tạo mask riêng cho mèo và chó
""")


# ============================================================
# 14. NHÌN CÙNG MỘT ẢNH THEO 3 CÁCH KHÁC NHAU
# ============================================================

print("14. NHÌN CÙNG MỘT ẢNH THEO 3 CÁCH KHÁC NHAU")
print("""
Giả sử có một ảnh chứa:
- 1 con mèo
- 1 con chó

Nếu dùng Classification:
- mô hình có thể chỉ trả về:
  "cat" hoặc "dog"
- nghĩa là chỉ mô tả rất sơ lược

Nếu dùng Detection:
- mô hình trả về:
  + cat ở vị trí A
  + dog ở vị trí B

Nếu dùng Segmentation:
- mô hình trả về:
  + vùng pixel của mèo
  + vùng pixel của chó
  + vùng nền

Đây là cách hiểu trực quan nhất để phân biệt 3 bài toán.
""")


# ============================================================
# 15. TẠI SAO PHẢI CHỌN ĐÚNG BÀI TOÁN?
# ============================================================

print("15. TẠI SAO PHẢI CHỌN ĐÚNG BÀI TOÁN?")
print("""
Nếu chọn sai loại bài toán, kết quả sẽ không đáp ứng được nhu cầu thực tế.

Ví dụ:
- nếu bạn chỉ cần biết ảnh có mèo hay chó
  -> classification là đủ

- nếu bạn cần đếm số người trong ảnh và biết họ ở đâu
  -> phải dùng detection

- nếu bạn cần đo chính xác diện tích tổn thương
  -> phải dùng segmentation

Chọn đúng bài toán giúp:
- mô hình phù hợp hơn
- dữ liệu được chuẩn bị đúng hơn
- kết quả hữu ích hơn trong thực tế
""")


# ============================================================
# 16. ỨNG DỤNG THỰC TẾ CỦA TỪNG BÀI TOÁN
# ============================================================

print("16. ỨNG DỤNG THỰC TẾ CỦA TỪNG BÀI TOÁN")
print("""
Classification:
- phân loại ảnh mèo / chó
- phân loại bệnh / không bệnh
- phân loại sản phẩm

Detection:
- phát hiện người, xe, mũ bảo hiểm
- phát hiện polyp
- phát hiện nốt phổi
- phát hiện lỗi sản phẩm

Segmentation:
- tách khối u
- tách polyp
- tách cơ quan trong ảnh y khoa
- tách vật thể khỏi nền
""")


# ============================================================
# 17. NHỮNG NHẦM LẪN NGƯỜI MỚI HAY GẶP
# ============================================================

print("17. NHỮNG NHẦM LẪN HAY GẶP")
print("""
Nhầm lẫn 1:
Detection và Segmentation là một
-> Sai
Detection dùng bounding box
Segmentation dùng pixel-level mask

Nhầm lẫn 2:
Classification luôn đủ cho mọi bài toán ảnh
-> Sai
Classification chỉ phù hợp khi không cần vị trí đối tượng

Nhầm lẫn 3:
Segmentation lúc nào cũng tốt hơn Detection
-> Không hẳn
Segmentation chi tiết hơn, nhưng:
- khó hơn
- tốn dữ liệu hơn
- tốn tài nguyên hơn

Nên phải chọn theo mục tiêu bài toán.
""")


# ============================================================
# 18. TÓM TẮT KIẾN THỨC CẦN NHỚ
# ============================================================

print("18. TÓM TẮT KIẾN THỨC CẦN NHỚ")
print("""
1. Computer Vision có 3 bài toán cốt lõi:
   - Classification
   - Detection
   - Segmentation

2. Classification:
   - đầu vào: ảnh
   - đầu ra: nhãn

3. Detection:
   - đầu vào: ảnh
   - đầu ra: nhãn + vị trí (bounding box)

4. Segmentation:
   - đầu vào: ảnh
   - đầu ra: nhãn theo từng pixel

5. Mức độ chi tiết:
   Classification < Detection < Segmentation

6. Chọn bài toán đúng quan trọng không kém chọn mô hình đúng
""")