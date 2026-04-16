"""
EX1 - OVERVIEW DEEP LEARNING FOR COMPUTER VISION

Mục tiêu:
Đây là file tổng quan toàn bộ lộ trình học Deep Learning for Computer Vision.
File này không đi quá sâu vào từng kỹ thuật, mà đóng vai trò "bản đồ tổng thể"
để giúp người học hiểu toàn bộ pipeline kiến thức trước khi đi vào từng ex riêng.

Sau file này, cần nắm được:
1. Deep Learning là gì và khác Machine Learning ra sao
2. Deep Learning for Computer Vision hoạt động như thế nào
3. Các bài toán chính trong Computer Vision
4. Ảnh được biểu diễn như thế nào trong máy tính
5. Model học từ Linear Classifier -> Neural Network -> CNN như thế nào
6. Pipeline train model trong PyTorch
7. Loss, gradient, backpropagation, optimization đóng vai trò gì
8. Các vấn đề khi train deep network
9. CNN nâng cao cho classification
10. Transfer learning, data augmentation và hướng mở rộng sang object detection / segmentation

Tài liệu gốc:
- PPT Deep Learning / Computer Vision của khóa học
"""

print("1. DEEP LEARNING LÀ GÌ?")
print("""
Deep Learning là một nhánh của Machine Learning, sử dụng các mạng neural nhiều tầng
để học trực tiếp từ dữ liệu.

Điểm khác biệt lớn nhất giữa Machine Learning truyền thống và Deep Learning là:

Machine Learning truyền thống:
- thường tách thành 2 bước
  1. Feature Extraction
  2. Classification / Regression

Deep Learning:
- gộp việc học đặc trưng và dự đoán vào cùng một pipeline end-to-end

Điều này giúp Deep Learning đặc biệt mạnh với dữ liệu phức tạp như:
- ảnh
- video
- âm thanh
- văn bản
""")


print("\n2. DEEP LEARNING FOR COMPUTER VISION")
print("""
Deep Learning for Computer Vision là việc dùng mô hình Deep Learning để xử lý ảnh và video.

Pipeline trực quan:
Input -> Sensing Device -> Interpreting Device -> Output

Ví dụ:
Ảnh quả cam -> Camera -> Mô hình AI / máy tính -> Orange

Ý chính:
- Camera chỉ thu dữ liệu ảnh
- Máy tính không hiểu ảnh như con người
- Mô hình Deep Learning phải học cách biến dữ liệu pixel thành thông tin có ý nghĩa
""")


print("\n3. CÁC BÀI TOÁN CHÍNH TRONG COMPUTER VISION")
print("""
Có 3 bài toán cốt lõi trong Computer Vision:

1. Classification
- Nhận một ảnh và dự đoán ảnh đó thuộc class nào
- Ví dụ: cat, dog, car

2. Object Detection
- Không chỉ dự đoán class
- mà còn xác định vật thể nằm ở đâu bằng bounding box

3. Segmentation
- Gán nhãn cho từng pixel
- chi tiết hơn detection
- dùng khi cần biết chính xác vùng của vật thể
""")


print("\n4. ẢNH ĐƯỢC BIỂU DIỄN NHƯ THẾ NÀO TRONG MÁY TÍNH?")
print("""
Máy tính không nhìn ảnh như con người.
Máy tính chỉ nhìn ảnh dưới dạng số.

Ảnh grayscale:
- biểu diễn bằng ma trận 2 chiều
- shape = (height, width)

Ảnh màu RGB:
- biểu diễn bằng tensor 3 chiều
- shape = (height, width, channels)

Ví dụ:
- ảnh 224 x 224 RGB -> (224, 224, 3)

Giá trị pixel thường nằm trong khoảng:
- 0 đến 255

Trước khi đưa vào model, pixel thường được:
- normalize về [0, 1]
- hoặc chuẩn hóa thêm để model học ổn định hơn

Khi làm việc với framework:
- PyTorch thường dùng format (C, H, W)
- TensorFlow / NumPy thường dùng (H, W, C)
""")


print("\n5. LINEAR CLASSIFIER - NỀN TẢNG CỦA MODEL")
print("""
Linear Classifier là model cơ bản nhất để hiểu classification.

Ý tưởng:
- Ảnh đầu vào được flatten thành vector
- Sau đó model tính score cho từng class

Công thức:
f(x, W, b) = Wx + b

Trong đó:
- x: vector đầu vào
- W: weight
- b: bias

Ví dụ với MNIST:
- ảnh 28x28 -> flatten thành 784 phần tử
- output là 10 class (0 đến 9)

Linear Classifier là nền tảng để đi tới:
- neuron
- neural network
- deep learning
""")


print("\n6. TỪ LINEAR CLASSIFIER ĐẾN NEURON")
print("""
Neuron nhân tạo là bước mở rộng từ Linear Classifier.

Linear Classifier:
z = Wx + b

Neuron:
z = Wx + b
a = activation(z)

Tức là:
Neuron = Linear Classifier + Activation Function

Activation function giúp model học được quan hệ phi tuyến.
Nếu không có activation, nhiều layer ghép lại vẫn chỉ giống một phép biến đổi tuyến tính.

Các activation phổ biến:
- Sigmoid
- Tanh
- ReLU
- Leaky ReLU
""")


print("\n7. NEURAL NETWORK VÀ DEEP NEURAL NETWORK")
print("""
Neural Network là tập hợp nhiều neuron kết nối với nhau.

Cấu trúc cơ bản:
- Input Layer
- Hidden Layer
- Output Layer

Forward propagation:
- dữ liệu đi từ input đến output
- model tạo prediction

Deep Neural Network:
- là neural network có nhiều hidden layers hơn
- cho phép học các đặc trưng từ đơn giản đến phức tạp hơn

Ví dụ trong ảnh:
- layer đầu học cạnh
- layer giữa học pattern
- layer sâu học object / object parts
""")


print("\n8. LOSS FUNCTION, GRADIENT, BACKPROPAGATION, OPTIMIZATION")
print("""
Sau khi model tạo prediction, ta cần biết:
- prediction đúng hay sai
- sai bao nhiêu

Đó là vai trò của Loss Function.

Ví dụ:
- Regression: MAE, MSE, Huber
- Classification: Cross-Entropy

Sau khi có loss:
- dùng gradient để biết loss thay đổi ra sao theo từng tham số
- dùng backpropagation để tính gradient trong mạng nhiều layer
- dùng optimizer / gradient descent để cập nhật weights

Pipeline học cơ bản:
1. Forward pass
2. Tính loss
3. Backpropagation
4. Update weights
5. Lặp lại
""")


print("\n9. CÁC VẤN ĐỀ KHI TRAIN DEEP NETWORK")
print("""
Khi train mạng sâu, có một số vấn đề rất quan trọng:

1. Vanishing Gradient
- gradient quá nhỏ
- layer đầu gần như không học được

2. Exploding Gradient
- gradient quá lớn
- training không ổn định

3. Weight Initialization
- cách khởi tạo weight ảnh hưởng mạnh đến training

4. Gradient Clipping
- kỹ thuật giới hạn độ lớn gradient để tránh exploding gradient

Ngoài ra còn có các kỹ thuật hỗ trợ:
- ReLU
- Batch Normalization
- Regularization
""")


print("\n10. CONVOLUTIONAL NEURAL NETWORK (CNN)")
print("""
CNN là loại Neural Network được thiết kế đặc biệt cho dữ liệu ảnh.

Điểm mạnh lớn nhất của CNN:
- không flatten ảnh ngay từ đầu
- giữ lại cấu trúc không gian của ảnh
- học feature trực tiếp trên ảnh

Thành phần phổ biến của CNN:
- Convolution Layer
- Activation Function
- Pooling Layer
- Fully Connected Layer

CNN đặc biệt hiệu quả cho:
- Image Classification
- Object Detection
- Segmentation
""")


print("\n11. CNN NÂNG CAO CHO IMAGE CLASSIFICATION")
print("""
Khi học CNN sâu hơn, có một số khái niệm rất quan trọng:

1. Receptive Field
- vùng input mà một neuron nhìn thấy

2. Effective Receptive Field
- phần thực sự ảnh hưởng mạnh nhất

3. Cách tăng receptive field
- thêm conv layers
- thêm pooling
- dùng dilated / atrous convolution

4. Common CNN layer pattern
- Conv -> BatchNorm -> ReLU -> Pooling -> ... -> Flatten -> FC

5. Nhiều kernel nhỏ thường tốt hơn một kernel lớn
- ví dụ: 2 lớp conv 3x3 thường tốt hơn 1 lớp conv 5x5 trong nhiều trường hợp
""")


print("\n12. PYTORCH DATA PIPELINE")
print("""
Để train model thực tế trong PyTorch, cần nắm pipeline cơ bản:

1. Setup a dataset
2. Define a model
3. Define a loss function
4. Define an optimizer
5. Train
6. Validate
7. Test

Trong PyTorch:
- dữ liệu được biểu diễn bằng tensor
- ảnh cần được chuyển từ file ảnh sang tensor
- dataset có thể dùng built-in datasets hoặc ImageFolder
""")


print("\n13. IMAGE CLASSIFICATION THỰC CHIẾN")
print("""
Khi đi từ lý thuyết sang thực chiến classification, có 3 cụm cực kỳ quan trọng:

1. ImageNet / ILSVRC
- benchmark rất lớn và quan trọng trong lịch sử Deep Learning
- tạo ra nhiều pretrained model mạnh

2. Transfer Learning
- tận dụng model đã train trước trên ImageNet
- có 2 cách phổ biến:
  - dùng CNN như fixed feature extractor
  - fine-tune CNN

3. Data Augmentation
- tạo thêm biến thể của dữ liệu train
- giúp model tổng quát tốt hơn
- ví dụ:
  - flip
  - crop
  - rotate
  - translation
  - scaling

Đây là phần cực kỳ quan trọng trong classification thực tế.
""")


print("\n14. HƯỚNG MỞ RỘNG SAU CLASSIFICATION")
print("""
Sau khi có nền tảng classification, ta mới đi tiếp hợp lý sang:

1. Object Detection
- phát hiện nhiều object
- xác định vị trí bằng bounding box

2. Segmentation
- phân loại theo từng pixel

3. GANs
- sinh dữ liệu mới
- augmentation nâng cao
- image generation / image restoration
""")


print("\n15. TỔNG KẾT TOÀN BỘ EX1")
print("""
EX1 là bản đồ tổng thể của toàn bộ lộ trình Deep Learning for Computer Vision.

Mạch kiến thức đầy đủ là:

Deep Learning
-> Computer Vision tasks
-> Image representation
-> Linear Classifier
-> Neuron
-> Neural Network
-> Loss / Gradient / Backprop / Optimization
-> CNN
-> CNN nâng cao
-> PyTorch pipeline
-> Image Classification thực chiến
-> Object Detection / Segmentation / GANs

Nếu hiểu chắc Ex1 này, người học sẽ không bị rời rạc khi đi vào từng ex chi tiết phía sau.
""")