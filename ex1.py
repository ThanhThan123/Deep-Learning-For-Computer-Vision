"""
EX1 - Deep Learning và Deep Learning for Computer Vision

Mục tiêu của file này:
1. Hiểu Deep Learning là gì
2. Phân biệt Deep Learning và Machine Learning
3. Hiểu Deep Learning for Computer Vision hoạt động như thế nào
4. Nắm được 3 bài toán chính trong Computer Vision:
   - Classification
   - Detection
   - Segmentation
5. Hiểu cách ảnh được biểu diễn trong Deep Learning
6. Hiểu Linear Classifier và score function
7. Làm quen với Neural Network, CNN, Image Classification, Object Detection
8. Biết một số thuật ngữ nền tảng cần nhớ

Tài liệu tham chiếu:
- PDF khóa học Deep Learning / Computer Vision / NLP của Thang Nguyen - Viet Nguyen
"""

# ============================================================
# 1. DEEP LEARNING LÀ GÌ?
# ============================================================

print("1. DEEP LEARNING")
print("- Deep Learning là một nhánh của Machine Learning.")
print("- Deep Learning dùng các mạng nơ-ron nhiều tầng để học dữ liệu.")
print("- Thay vì con người phải tự nghĩ ra quá nhiều đặc trưng (feature),")
print("  mô hình có thể tự học đặc trưng từ dữ liệu đầu vào.")


# ============================================================
# 2. SỰ KHÁC NHAU GIỮA MACHINE LEARNING VÀ DEEP LEARNING
# ============================================================

print("\n2. MACHINE LEARNING VS DEEP LEARNING")

difference_ml_dl = """
Machine Learning truyền thống thường tách thành 2 bước:
1. Feature Extraction (trích xuất đặc trưng)
2. Classification / Regression (phân loại / dự đoán)

Ví dụ:
- Với ảnh con mèo và con chó, ta có thể phải tự tìm đặc trưng như:
  màu sắc, cạnh, texture, hình dạng tai, mắt, v.v.
- Sau đó mới đưa các đặc trưng đó vào mô hình phân loại.

Deep Learning thì khác:
- Mô hình học trực tiếp từ dữ liệu thô (raw data)
- Gộp cả Feature Extraction và Classification thành một quá trình end-to-end

Ưu điểm của Deep Learning:
- Tự học đặc trưng mạnh hơn
- Khả năng xử lý dữ liệu phức tạp tốt hơn
- Có thể output ra nhiều thông tin hơn
"""

print(difference_ml_dl)


# ============================================================
# 3. DEEP LEARNING FOR COMPUTER VISION
# ============================================================

print("\n3. DEEP LEARNING FOR COMPUTER VISION")

computer_vision_pipeline = """
Một hệ thống thị giác máy tính thường hoạt động theo chuỗi:

Input -> Sensing device -> Interpreting device -> Output

Giải thích:
- Input: ảnh hoặc video đầu vào
- Sensing device: thiết bị ghi nhận dữ liệu, ví dụ camera
- Interpreting device: máy tính / mô hình AI làm nhiệm vụ phân tích
- Output: kết quả cuối cùng

Ví dụ:
Quả cam -> Camera -> Máy tính (phiên dịch) -> Orange
"""

print(computer_vision_pipeline)


# ============================================================
# 4. CÁC BÀI TOÁN CHÍNH TRONG COMPUTER VISION
# ============================================================

print("\n4. CÁC BÀI TOÁN CHÍNH TRONG COMPUTER VISION")

cv_tasks = """
Có 3 bài toán rất quan trọng trong Deep Learning for Computer Vision:

1. Classification
2. Detection
3. Segmentation
"""

print(cv_tasks)


# ---------------------------
# 4.1 Classification
# ---------------------------

classification_text = """
4.1 Classification

Classification là bài toán phân loại ảnh.

Đầu vào:
- Một ảnh

Đầu ra:
- Một nhãn (label) hoặc xác suất cho từng lớp

Ví dụ:
- Ảnh này là mèo
- Ảnh này là chó
- Ảnh này là xe hơi

Đây thường là bài toán single object:
- Chỉ quan tâm nhãn chính của ảnh
- Không cần biết vật thể nằm ở đâu trong ảnh
"""

print(classification_text)


# ---------------------------
# 4.2 Detection
# ---------------------------

detection_text = """
4.2 Detection

Detection là bài toán phát hiện đối tượng trong ảnh.

Đầu vào:
- Một ảnh

Đầu ra:
- Có bao nhiêu đối tượng
- Mỗi đối tượng là gì
- Vị trí của từng đối tượng ở đâu

Detection thường dùng bounding box (hình chữ nhật bao quanh vật thể).

Ví dụ:
- Ảnh có 2 con chó và 1 con mèo
- Mỗi con sẽ được gắn nhãn và vẽ khung vị trí

Đây là bài toán multiple objects.
"""

print(detection_text)


# ---------------------------
# 4.3 Segmentation
# ---------------------------

segmentation_text = """
4.3 Segmentation

Segmentation là bài toán chi tiết hơn detection.

Thay vì chỉ vẽ khung chữ nhật quanh đối tượng,
mô hình sẽ dự đoán nhãn cho từng pixel.

Ví dụ:
- Pixel nào thuộc con mèo
- Pixel nào thuộc con chó
- Pixel nào thuộc nền

Segmentation giúp đường viền ôm sát vật thể hơn detection.

Đây cũng là bài toán multiple objects nếu trong ảnh có nhiều vật thể.
"""

print(segmentation_text)


# ============================================================
# 5. IMAGE CLASSIFICATION - ẢNH ĐƯỢC BIỂU DIỄN NHƯ THẾ NÀO?
# ============================================================

print("\n5. IMAGE CLASSIFICATION - BIỂU DIỄN ẢNH")

image_representation = """
Trong Deep Learning, máy tính không nhìn ảnh như con người.
Máy tính nhìn ảnh dưới dạng ma trận hoặc tensor số.

Thông thường có 2 loại ảnh:
1. Ảnh đen trắng (Grayscale)
2. Ảnh màu (Color Image)
"""

print(image_representation)


# ---------------------------
# 5.1 Ảnh đen trắng
# ---------------------------

grayscale_text = """
5.1 Ảnh đen trắng (Grayscale)

Ảnh grayscale là ảnh chỉ có 1 kênh màu.

Biểu diễn:
- Là mảng 2 chiều: (height, width)

Ví dụ:
- shape = (28, 28)

Giá trị pixel:
- Từ 0 đến 255
- 0 là đen
- 255 là trắng
- Các giá trị ở giữa là các mức xám
"""

print(grayscale_text)


# ---------------------------
# 5.2 Ảnh màu
# ---------------------------

color_image_text = """
5.2 Ảnh màu (Color Image)

Ảnh màu thường có 3 kênh:
- Red
- Green
- Blue

Biểu diễn:
- Là mảng 3 chiều: (height, width, channels)

Ví dụ:
- shape = (224, 224, 3)

Điều đó có nghĩa:
- chiều cao = 224
- chiều rộng = 224
- số kênh màu = 3
"""

print(color_image_text)


# ---------------------------
# 5.3 Format dữ liệu ảnh
# ---------------------------

format_text = """
5.3 Format dữ liệu ảnh

Có 2 kiểu lưu dữ liệu ảnh rất hay gặp:

1. Channel-last:
   (H, W, C)
   Ví dụ: (224, 224, 3)
   Phổ biến trong:
   - NumPy
   - TensorFlow

2. Channel-first:
   (C, H, W)
   Ví dụ: (3, 224, 224)
   Phổ biến trong:
   - PyTorch

Lưu ý:
- Khi làm việc với thư viện Deep Learning, phải để ý format dữ liệu ảnh
- Nếu sai format, mô hình có thể lỗi hoặc chạy sai
"""

print(format_text)


# ---------------------------
# 5.4 Chuẩn hóa pixel
# ---------------------------

normalization_text = """
5.4 Chuẩn hóa pixel

Trong Deep Learning, pixel thường không giữ nguyên khoảng [0, 255].

Thường sẽ đưa về:
- [0, 1]
  bằng cách chia cho 255

hoặc:
- [-1, 1]
  tùy cách tiền xử lý

Mục đích:
- Giúp mô hình học ổn định hơn
- Tăng hiệu quả tối ưu
"""

print(normalization_text)


# ============================================================
# 6. LINEAR CLASSIFIER - SCORE FUNCTION
# ============================================================

print("\n6. LINEAR CLASSIFIER - SCORE FUNCTION")

linear_classifier_text = """
Linear Classifier là mô hình nền tảng để hiểu Neural Network.

Ý tưởng:
- Ảnh đầu vào ban đầu có thể là ma trận 2 chiều hoặc tensor 3 chiều
- Ta cần biến ảnh thành vector
- Sau đó tính điểm (score) cho từng lớp

Công thức:
f(x_i, W, b) = W*x_i + b

Trong đó:
- x_i: vector đầu vào của ảnh thứ i
- W: trọng số (weights)
- b: bias
- f(...): vector điểm số cho các class
"""

print(linear_classifier_text)


# ---------------------------
# 6.1 Flatten ảnh thành vector
# ---------------------------

flatten_text = """
6.1 Flatten ảnh thành vector

Ví dụ ảnh grayscale 28x28:
- Ban đầu shape là (28, 28)
- Sau khi trải phẳng, ta được vector 784 phần tử

Ví dụ:
Input image -> flatten -> vector x_i

Đây là bước rất quan trọng vì mô hình tuyến tính làm việc tốt với vector số.
"""

print(flatten_text)


# ---------------------------
# 6.2 Giải thích trực quan
# ---------------------------

intuitive_text = """
6.2 Cách hiểu trực quan

Ta có thể tưởng tượng:
- Mỗi pixel sẽ được nhân với một weight tương ứng
- Sau đó cộng tất cả lại
- Cuối cùng cộng thêm bias để ra score

Nếu score của class nào lớn nhất,
mô hình sẽ dự đoán ảnh thuộc class đó.
"""

print(intuitive_text)


# ---------------------------
# 6.3 Giải thích theo đại số tuyến tính
# ---------------------------

math_text = """
6.3 Cách hiểu theo đại số tuyến tính

Thay vì nhân từng pixel thủ công,
ta thực hiện phép nhân ma trận:

f(x_i, W, b) = W*x_i + b

Ý nghĩa:
- W chứa trọng số cho từng class
- x_i là vector ảnh
- b là bias
- kết quả là vector score cho tất cả các class

Class có score lớn nhất sẽ là dự đoán cuối cùng.
"""

print(math_text)


# ============================================================
# 7. LINEAR EXPLANATION
# ============================================================

print("\n7. LINEAR EXPLANATION")

linear_explanation = """
Ví dụ với MNIST:
- Input: 784 pixels
- Output: 10 classes (các chữ số từ 0 đến 9)

Nghĩa là:
- Một ảnh chữ số viết tay được biểu diễn thành 784 giá trị pixel
- Mô hình sẽ tính score cho 10 class
- Class nào có score cao nhất sẽ là kết quả dự đoán
"""

print(linear_explanation)


# ============================================================
# 8. LINEAR CLASSIFIER - WEIGHT VISUALIZATION
# ============================================================

print("\n8. WEIGHT VISUALIZATION")

weight_visualization_text = """
Weight visualization giúp ta nhìn trực quan mô hình đang học gì.

Ý tưởng:
- Với mỗi class, mô hình học một bộ trọng số
- Nếu reshape trọng số này về dạng giống ảnh,
  ta có thể hình dung class đó "trông như thế nào"

Ví dụ:
- Nếu input là số 0
- Thì weight của class số 0 nên giống hình số 0

Có 2 cách nghĩ:
1. Trải phẳng ảnh x để khớp với vector w
2. Hoặc reshape w về lại dạng hình vuông để quan sát
"""

print(weight_visualization_text)


# ============================================================
# 9. MỘT VÀI THUẬT NGỮ QUAN TRỌNG
# ============================================================

print("\n9. CÁC THUẬT NGỮ QUAN TRỌNG")

terminology_text = """
Input Layer:
- Lớp nhận dữ liệu đầu vào

Middle Layer / Hidden Layer:
- Lớp ẩn
- Nơi mô hình học ra các biểu diễn trung gian

Output Layer:
- Lớp đầu ra
- Trả về kết quả cuối cùng
"""

print(terminology_text)


# ============================================================
# 10. DATASET
# ============================================================

print("\n10. DATASET")

dataset_text = """
Dataset là tập dữ liệu dùng để huấn luyện mô hình.

Một dataset thường bao gồm:
- nhiều ảnh
- nhãn tương ứng

Ví dụ:
- cat
- dog
- mug
- hat

Dataset càng tốt thì mô hình càng có cơ hội học tốt.

Công cụ hay dùng để đánh nhãn:
- CVAT
  Đây là công cụ phổ biến để annotation dữ liệu ảnh.
"""

print(dataset_text)


# ============================================================
# 11. NEURAL NETWORK
# ============================================================

print("\n11. NEURAL NETWORK")

nn_text = """
Neural Network là cầu nối giữa Machine Learning và Deep Learning.

Cấu trúc cơ bản của một Neural Network gồm 3 loại layer:

1. Input Layer
   - Nhận dữ liệu đầu vào

2. Hidden Layer (Middle Layer)
   - Xử lý thông tin trung gian
   - Học các đặc trưng ẩn

3. Output Layer
   - Trả kết quả dự đoán cuối cùng

Khi số hidden layers nhiều hơn,
mô hình được gọi là Deep Neural Network.
"""

print(nn_text)


# ============================================================
# 12. CONVOLUTIONAL NEURAL NETWORK (CNN)
# ============================================================

print("\n12. CONVOLUTIONAL NEURAL NETWORK (CNN)")

cnn_text = """
CNN là một loại Neural Network rất quan trọng trong Computer Vision.

Tại sao cần CNN?
- Ảnh có cấu trúc không gian
- Pixel gần nhau thường có liên hệ với nhau
- CNN khai thác tốt tính chất này hơn so với việc flatten toàn bộ ảnh ngay từ đầu

CNN thường gồm các thành phần như:
- Convolution layer
- Activation function
- Pooling layer
- Fully connected layer

CNN đặc biệt hiệu quả cho:
- Image Classification
- Object Detection
- Segmentation
"""

print(cnn_text)


# ============================================================
# 13. IMAGE CLASSIFICATION
# ============================================================

print("\n13. IMAGE CLASSIFICATION")

image_classification_text = """
Image Classification là bài toán phân loại ảnh.

Đầu vào:
- Một ảnh

Đầu ra:
- Một class hoặc xác suất cho các class

Ví dụ:
- dog
- cat
- bird

Đây là bài toán nền tảng nhất trong Deep Learning for Computer Vision.
"""

print(image_classification_text)


# ============================================================
# 14. OBJECT DETECTION
# ============================================================

print("\n14. OBJECT DETECTION")

object_detection_text = """
Object Detection là bài toán định vị đối tượng.

Không chỉ phân loại ảnh có gì,
mà còn xác định:
- đối tượng nào xuất hiện
- ở đâu trong ảnh

Kết quả thường bao gồm:
- tên class
- bounding box
- confidence score
"""

print(object_detection_text)


# ============================================================
# 15. OTHER TOPICS
# ============================================================

print("\n15. OTHER TOPICS")

other_topics_text = """
Một số chủ đề mở rộng rất quan trọng trong Computer Vision:

1. Image Segmentation
- Gán nhãn cho từng pixel
- Rất hữu ích trong y tế, tự lái, xử lý ảnh

2. GANs (Generative Adversarial Networks)
- Mô hình sinh dữ liệu
- Có thể tạo ảnh mới, tăng cường dữ liệu, khôi phục ảnh, v.v.
"""

print(other_topics_text)


# ============================================================
# 16. TỔNG KẾT
# ============================================================

print("\n16. TỔNG KẾT")

summary_text = """
Sau file ex1 này, cần nhớ:

1. Deep Learning là nhánh của Machine Learning
2. Deep Learning khác Machine Learning ở chỗ:
   - tự học feature
   - gộp feature extraction và classification thành một pipeline
3. Deep Learning for Computer Vision xử lý ảnh/video bằng AI
4. 3 bài toán chính:
   - Classification
   - Detection
   - Segmentation
5. Ảnh là ma trận / tensor số
6. Linear Classifier dùng công thức:
   f(x, W, b) = W*x + b
7. Neural Network gồm:
   - Input layer
   - Hidden layer
   - Output layer
8. CNN là mô hình rất quan trọng cho xử lý ảnh
"""

print(summary_text)