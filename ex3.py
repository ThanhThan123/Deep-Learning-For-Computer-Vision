"""
EX3 - Deep Learning for Computer Vision

Mục tiêu:
1. Hiểu Deep Learning for Computer Vision là gì
2. Hiểu hệ thống Computer Vision hoạt động như thế nào
3. Phân biệt 3 bài toán quan trọng:
   - Classification
   - Detection
   - Segmentation
4. Hiểu máy tính nhìn ảnh khác con người như thế nào
5. Hiểu vai trò của dữ liệu có nhãn (labeled data)

Ghi chú:
- File này tập trung vào tư duy nền tảng trước khi đi sâu vào model.
- Đây là phần rất quan trọng vì gần như toàn bộ các bài sau đều dựa trên nó.
"""


# ============================================================
# 1. DEEP LEARNING FOR COMPUTER VISION LÀ GÌ?
# ============================================================

print("1. DEEP LEARNING FOR COMPUTER VISION")
print("""
Deep Learning for Computer Vision là việc dùng các mô hình Deep Learning
để giúp máy tính hiểu và xử lý dữ liệu hình ảnh hoặc video.

Nói đơn giản:
- Con người nhìn ảnh và hiểu được ảnh có gì
- Máy tính thì không hiểu ảnh theo cách trực quan như con người
- Ta cần mô hình Deep Learning để giúp máy tính học cách "nhìn" ảnh

Computer Vision thường làm các việc như:
- phân loại ảnh
- phát hiện đối tượng
- tách vùng đối tượng
- nhận diện khuôn mặt
- đọc biển số xe
- hỗ trợ chẩn đoán ảnh y khoa
""")


# ============================================================
# 2. COMPUTER VISION HOẠT ĐỘNG NHƯ THẾ NÀO?
# ============================================================

print("2. COMPUTER VISION HOẠT ĐỘNG NHƯ THẾ NÀO?")
print("""
Theo tư duy trong slide, một hệ thống Computer Vision thường đi theo chuỗi:

Input -> Sensing device -> Interpreting device -> Output

Giải thích từng phần:

1. Input
- Là ảnh hoặc video đầu vào

2. Sensing device
- Là thiết bị ghi nhận dữ liệu
- Ví dụ: camera, webcam, camera điện thoại, cảm biến ảnh

3. Interpreting device
- Là máy tính hoặc mô hình AI
- Đây là nơi ảnh được xử lý và phân tích

4. Output
- Là kết quả cuối cùng
- Ví dụ:
  + đây là quả cam
  + đây là con mèo
  + trong ảnh có 2 con chó
  + vùng màu đỏ là khối u
""")


# ============================================================
# 3. VÍ DỤ TRỰC QUAN VỀ CÁCH HỆ THỐNG HIỂU ẢNH
# ============================================================

print("3. VÍ DỤ TRỰC QUAN")
print("""
Ví dụ trong slide:

Quả cam -> Camera -> Máy tính (thiết bị phiên dịch) -> Orange

Điều đó có nghĩa là:
- vật thể ngoài đời thực được camera ghi lại
- camera chỉ chụp được dữ liệu hình ảnh
- máy tính phải dùng mô hình để diễn giải ảnh đó
- sau cùng mới đưa ra nhãn phù hợp

Điểm cần nhớ:
Camera không "hiểu" ảnh.
Máy tính cũng không tự "hiểu" ảnh.
Chỉ khi có mô hình phù hợp, dữ liệu ảnh mới được chuyển thành thông tin có ý nghĩa.
""")


# ============================================================
# 4. KHÁC NHAU GIỮA THỊ GIÁC CON NGƯỜI VÀ MÁY TÍNH
# ============================================================

print("4. THỊ GIÁC CON NGƯỜI VS THỊ GIÁC MÁY")
print("""
Con người nhìn một bức ảnh và gần như nhận ra vật thể rất nhanh.

Ví dụ:
- nhìn thấy mèo
- nhìn thấy chó
- nhìn thấy quả cam

Nhưng máy tính không nhìn theo cách đó.
Máy tính chỉ nhận được một tập hợp các con số.

Với máy tính:
- ảnh là ma trận số
- mỗi pixel là một giá trị
- mô hình học cách biến ma trận số này thành nhãn, vị trí, hoặc mask

Vì vậy, phần khó nhất của Computer Vision là:
làm sao để biến dữ liệu pixel thành tri thức có ý nghĩa.
""")


# ============================================================
# 5. BA BÀI TOÁN CHÍNH TRONG DEEP LEARNING FOR COMPUTER VISION
# ============================================================

print("5. BA BÀI TOÁN CHÍNH")
print("""
Trong phần mở đầu của khóa học, có 3 bài toán rất quan trọng:

1. Classification
2. Detection
3. Segmentation

Đây là 3 nhóm bài toán nền tảng nhất trong Computer Vision hiện đại.
""")


# ============================================================
# 6. CLASSIFICATION
# ============================================================

print("6. CLASSIFICATION")
print("""
Classification là bài toán phân loại ảnh.

Ý tưởng:
- Đưa vào một ảnh
- Mô hình trả về ảnh đó thuộc lớp nào

Ví dụ:
- cat
- dog
- airplane
- car

Đặc điểm:
- thường dùng khi chỉ cần biết ảnh là gì
- không cần xác định vị trí chính xác của vật thể
- thường được xem là bài toán single object trong ngữ cảnh nhập môn

Ví dụ:
- ảnh này là mèo hay chó?
- ảnh X-quang này là bình thường hay bất thường?
- ảnh lá cây này đang bị bệnh gì?
""")


# ============================================================
# 7. DETECTION
# ============================================================

print("7. DETECTION")
print("""
Detection là bài toán phát hiện đối tượng.

Khác với Classification:
- Classification chỉ trả lời: ảnh này là gì
- Detection trả lời:
  + có những đối tượng nào trong ảnh
  + có bao nhiêu đối tượng
  + mỗi đối tượng nằm ở đâu

Kết quả thường gồm:
- tên class
- bounding box
- confidence score

Ví dụ:
- trong ảnh có 1 con mèo và 2 con chó
- mỗi con được vẽ bằng một hình chữ nhật bao quanh

Detection phù hợp với:
- camera giao thông
- phát hiện người trong CCTV
- phát hiện polyp
- phát hiện nốt phổi
- phát hiện sản phẩm lỗi
""")


# ============================================================
# 8. SEGMENTATION
# ============================================================

print("8. SEGMENTATION")
print("""
Segmentation là bài toán chi tiết hơn Detection.

Thay vì chỉ vẽ khung chữ nhật,
Segmentation sẽ dự đoán nhãn cho từng pixel.

Hiểu đơn giản:
- pixel nào thuộc vật thể A
- pixel nào thuộc vật thể B
- pixel nào thuộc nền

Ưu điểm:
- biên đối tượng chính xác hơn
- biết được hình dạng thật của vật thể

Ví dụ:
- tách khối u trong ảnh y khoa
- tách con đường trong xe tự lái
- tách polyp khỏi nền niêm mạc
- tách người ra khỏi background

Segmentation thường được dùng khi:
- cần độ chính xác vị trí cao
- chỉ bounding box là chưa đủ
""")


# ============================================================
# 9. SO SÁNH NHANH 3 BÀI TOÁN
# ============================================================

print("9. SO SÁNH NHANH 3 BÀI TOÁN")
print("""
Classification:
- biết ảnh là gì

Detection:
- biết ảnh có gì
- biết vật thể ở đâu

Segmentation:
- biết ảnh có gì
- biết vật thể ở đâu
- biết chính xác vùng pixel của vật thể

Nói cách khác:
Classification < Detection < Segmentation
về mức độ chi tiết thông tin đầu ra.
""")


# ============================================================
# 10. IMAGE CLASSIFICATION TRONG COMPUTER VISION
# ============================================================

print("10. IMAGE CLASSIFICATION")
print("""
Trong slide minh họa về Image Classification, mô hình nhìn ảnh con mèo
và trả về xác suất cho nhiều lớp khác nhau.

Ví dụ:
- 82% cat
- 15% dog
- 2% hat
- 1% mug

Điều này rất quan trọng vì:
- mô hình không chỉ trả lời một nhãn duy nhất
- nó thường tính score / probability cho tất cả class
- sau đó chọn class có xác suất cao nhất làm dự đoán

Đây chính là tư duy xác suất trong phân loại ảnh.
""")


# ============================================================
# 11. MÁY TÍNH "NHÌN" ẢNH NHƯ THẾ NÀO?
# ============================================================

print("11. MÁY TÍNH NHÌN ẢNH NHƯ THẾ NÀO?")
print("""
Con người nhìn thấy con mèo.
Máy tính thì không.

Máy tính chỉ thấy:
- một ma trận pixel
- hoặc một tensor số

Nghĩa là:
- ảnh thật ngoài đời chỉ là đầu vào trực quan với con người
- còn với mô hình, ảnh phải được chuyển thành số

Ví dụ:
- ảnh grayscale: ma trận 2 chiều
- ảnh màu RGB: tensor 3 chiều

Từ những con số đó, mô hình học ra quy luật để phân loại hoặc phát hiện đối tượng.
""")


# ============================================================
# 12. VÌ SAO DEEP LEARNING PHÙ HỢP VỚI COMPUTER VISION?
# ============================================================

print("12. VÌ SAO DEEP LEARNING PHÙ HỢP VỚI COMPUTER VISION?")
print("""
Dữ liệu ảnh rất phức tạp vì:
- có quá nhiều pixel
- cùng một vật thể có thể thay đổi:
  + góc nhìn
  + ánh sáng
  + màu sắc
  + kích thước
  + nền ảnh

Nếu dùng cách thủ công để thiết kế đặc trưng,
ta sẽ rất khó mô tả hết mọi tình huống.

Deep Learning mạnh ở chỗ:
- tự học đặc trưng từ dữ liệu
- học từ đơn giản đến phức tạp
- phù hợp với dữ liệu ảnh lớn và phức tạp
""")


# ============================================================
# 13. DỮ LIỆU CÓ NHÃN (LABELED DATA) QUAN TRỌNG THẾ NÀO?
# ============================================================

print("13. LABELED DATA QUAN TRỌNG THẾ NÀO?")
print("""
Trong slide có minh họa ảnh mèo và chó kèm nhãn.
Điều này cho thấy vai trò rất quan trọng của labeled data.

Labeled data là dữ liệu đã được gán nhãn đúng.

Ví dụ:
- ảnh này là cat
- ảnh kia là dog

Nếu nhãn sai:
- mô hình sẽ học sai

Nếu dữ liệu ít hoặc không đa dạng:
- mô hình khó tổng quát tốt

Nói cách khác:
chất lượng dữ liệu quan trọng không kém mô hình.
""")


# ============================================================
# 14. DATASET TRONG COMPUTER VISION
# ============================================================

print("14. DATASET")
print("""
Dataset là tập dữ liệu dùng để huấn luyện mô hình.

Một dataset tốt cần:
1. Có đủ số lượng ảnh
2. Có nhãn đúng
3. Đa dạng về:
   - góc chụp
   - ánh sáng
   - background
   - kích thước đối tượng
4. Phản ánh đúng dữ liệu ngoài thực tế

Ví dụ trong slide:
- dataset gồm nhiều ảnh thuộc các class như:
  cat, dog, mug, hat

Điều này giúp mô hình học được sự khác nhau giữa các lớp.
""")


# ============================================================
# 15. BÀI TOÁN SINGLE OBJECT VÀ MULTIPLE OBJECT
# ============================================================

print("15. SINGLE OBJECT VS MULTIPLE OBJECT")
print("""
Single object:
- thường gặp trong bài toán classification nhập môn
- mỗi ảnh chủ yếu gắn với một nhãn chính

Multiple objects:
- thường gặp trong detection và segmentation
- một ảnh có thể chứa nhiều đối tượng khác nhau

Ví dụ:
- 1 ảnh có cả mèo và chó
- mô hình phải nhận ra đủ cả hai
""")


# ============================================================
# 16. ỨNG DỤNG THỰC TẾ CỦA DEEP LEARNING FOR COMPUTER VISION
# ============================================================

print("16. ỨNG DỤNG THỰC TẾ")
print("""
Deep Learning for Computer Vision có rất nhiều ứng dụng thực tế:

1. Y tế
- phát hiện tổn thương
- phát hiện nốt phổi
- tách khối u
- hỗ trợ chẩn đoán hình ảnh

2. Giao thông
- phát hiện xe, người, biển báo
- nhận diện biển số
- hỗ trợ xe tự lái

3. Công nghiệp
- kiểm tra lỗi sản phẩm
- phát hiện dị vật
- giám sát chất lượng

4. Bán lẻ / đời sống
- nhận diện khuôn mặt
- nhận diện hành động
- phân loại ảnh sản phẩm
""")


# ============================================================
# 17. NHỮNG NHẦM LẪN NGƯỜI MỚI HỌC HAY GẶP
# ============================================================

print("17. NHỮNG NHẦM LẪN HAY GẶP")
print("""
Nhầm lẫn 1:
Computer Vision = chỉ có classification
-> Sai, còn detection và segmentation

Nhầm lẫn 2:
Máy tính nhìn ảnh giống con người
-> Sai, máy tính nhìn ảnh dưới dạng số

Nhầm lẫn 3:
Chỉ cần mô hình mạnh là đủ
-> Sai, dữ liệu và nhãn dữ liệu cũng cực kỳ quan trọng

Nhầm lẫn 4:
Detection và Segmentation giống nhau
-> Sai, segmentation chi tiết hơn nhiều vì dự đoán tới từng pixel
""")


# ============================================================
# 18. TỔNG KẾT
# ============================================================

print("18. TỔNG KẾT")
print("""
Sau file Ex3 này, cần nhớ:

1. Deep Learning for Computer Vision là dùng AI để hiểu ảnh / video
2. Hệ thống CV thường đi theo pipeline:
   Input -> Sensing device -> Interpreting device -> Output
3. 3 bài toán cốt lõi:
   - Classification
   - Detection
   - Segmentation
4. Máy tính không nhìn ảnh như con người,
   mà nhìn ảnh như ma trận / tensor số
5. Dữ liệu có nhãn và dataset chất lượng rất quan trọng
6. Deep Learning phù hợp với Computer Vision vì ảnh là dữ liệu phức tạp
""")