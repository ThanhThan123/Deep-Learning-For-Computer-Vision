"""
EX11 - PyTorch Tensor, Image-to-Tensor, Dataset và Training Pipeline

Mục tiêu:
1. Hiểu quy trình train một mạng neural trong PyTorch
2. Hiểu tensor trong PyTorch là gì
3. Hiểu cách chuyển ảnh thành tensor
4. Hiểu dataset trong PyTorch
5. Biết built-in datasets và ImageFolder
6. Hiểu vai trò của train / validate / test

Tài liệu gốc:
- PPT Deep Learning / Computer Vision
"""


# ============================================================
# 1. TỔNG QUAN
# ============================================================

print("1. TỔNG QUAN")
print("""
Sau khi hiểu:
- model là gì
- loss là gì
- gradient và backpropagation hoạt động thế nào

thì bước tiếp theo là:
- đưa dữ liệu thật vào hệ thống
- biểu diễn dữ liệu bằng PyTorch
- train model theo một pipeline rõ ràng
""")


# ============================================================
# 2. TRAIN A (CONVOLUTIONAL) NEURAL NETWORK IN PYTORCH
# ============================================================

print("\n2. TRAIN A (CONVOLUTIONAL) NEURAL NETWORK IN PYTORCH")
print("""
Theo PPT, quy trình train một mạng neural / CNN trong PyTorch gồm 7 bước:

Step 1: Setup a dataset
Step 2: Define a model
Step 3: Define a loss function
Step 4: Define an optimizer
Step 5: Train
Step 6: Validate
Step 7: Test
""")


# ============================================================
# 3. STEP 1 - SETUP A DATASET
# ============================================================

print("\n3. STEP 1 - SETUP A DATASET")
print("""
Đầu tiên phải xây dựng dataset từ tập ảnh.

Ý nghĩa:
- đọc dữ liệu từ thư mục / file
- gắn nhãn nếu cần
- đưa dữ liệu về format mà PyTorch xử lý được

Không có dataset đúng:
- model không thể train đúng
- pipeline sẽ không chạy được
""")


# ============================================================
# 4. STEP 2 - DEFINE A MODEL
# ============================================================

print("\n4. STEP 2 - DEFINE A MODEL")
print("""
Sau khi có dữ liệu, ta cần định nghĩa mô hình.

Ví dụ:
- một mạng neural đơn giản
- hoặc một Convolutional Neural Network (CNN)

Mô hình sẽ quy định:
- dữ liệu đi qua các layer thế nào
- output cuối cùng được tạo ra ra sao
""")


# ============================================================
# 5. STEP 3 - DEFINE A LOSS FUNCTION
# ============================================================

print("\n5. STEP 3 - DEFINE A LOSS FUNCTION")
print("""
Tiếp theo là chọn hoặc xây dựng hàm loss.

Loss function dùng để:
- đo model đang sai bao nhiêu
- cung cấp tín hiệu để cập nhật tham số

Ví dụ:
- classification -> CrossEntropyLoss
- regression -> MSELoss, L1Loss, ...
""")


# ============================================================
# 6. STEP 4 - DEFINE AN OPTIMIZER
# ============================================================

print("\n6. STEP 4 - DEFINE AN OPTIMIZER")
print("""
Optimizer dùng để cập nhật weights của model.

Ví dụ phổ biến:
- SGD
- Adam
- RMSprop

Vai trò:
- nhận gradient từ backpropagation
- update parameters để giảm loss
""")


# ============================================================
# 7. STEP 5 - TRAIN
# ============================================================

print("\n7. STEP 5 - TRAIN")
print("""
Train là giai đoạn học trên training data.

Trong bước này:
- model nhìn thấy dữ liệu train
- tạo prediction
- tính loss
- backpropagation
- optimizer update weights

Đây là nơi model thực sự học.
""")


# ============================================================
# 8. STEP 6 - VALIDATE
# ============================================================

print("\n8. STEP 6 - VALIDATE")
print("""
Validate là bước đánh giá model trong quá trình train
bằng validation data.

Mục đích:
- xem model có đang học tốt không
- theo dõi overfitting / underfitting
- so sánh mô hình hoặc hyperparameters

Validation không dùng để update weights.
""")


# ============================================================
# 9. STEP 7 - TEST
# ============================================================

print("\n9. STEP 7 - TEST")
print("""
Test là bước đánh giá cuối cùng với test data.

Mục tiêu:
- đo hiệu năng thật của model trên dữ liệu chưa thấy
- báo cáo kết quả cuối cùng

Test thường được dùng sau khi đã:
- train xong
- chọn được model tốt
- chốt cấu hình
""")


# ============================================================
# 10. PYTORCH TENSOR LÀ GÌ?
# ============================================================

print("\n10. PYTORCH TENSOR LÀ GÌ?")
print("""
Theo PPT:
Tensor in PyTorch is equivalent to Array in NumPy.

Tức là:
- tensor là cấu trúc dữ liệu cơ bản trong PyTorch
- gần giống array của NumPy
- nhưng được thiết kế để hỗ trợ Deep Learning tốt hơn

Hầu hết dữ liệu trong PyTorch đều sẽ được biểu diễn bằng tensor.
""")


# ============================================================
# 11. VÌ SAO TENSOR QUAN TRỌNG?
# ============================================================

print("\n11. VÌ SAO TENSOR QUAN TRỌNG?")
print("""
Tensor quan trọng vì:
- model trong PyTorch nhận input dưới dạng tensor
- weight của model cũng là tensor
- gradient cũng được tính trên tensor

Nói cách khác:
tensor là "ngôn ngữ dữ liệu" của PyTorch.
""")


# ============================================================
# 12. VÍ DỤ TENSOR
# ============================================================

print("\n12. VÍ DỤ TENSOR")

import torch

scalar = torch.tensor(5)
vector = torch.tensor([1, 2, 3])
matrix = torch.tensor([[1, 2], [3, 4]])

print("Scalar:", scalar)
print("Vector:", vector)
print("Matrix:", matrix)


# ============================================================
# 13. IMAGE TO TENSOR
# ============================================================

print("\n13. IMAGE TO TENSOR")
print("""
Theo PPT, ảnh cần được chuyển thành tensor trước khi đưa vào model.

Đây là bước rất quan trọng vì:
- ảnh gốc thường ở dạng file như .jpg, .png
- model không đọc trực tiếp file ảnh như con người
- model cần tensor số

Vì vậy:
Image -> Tensor
là bước bắt buộc trong pipeline Deep Learning với PyTorch.
""")


# ============================================================
# 14. TẠI SAO KHÔNG ĐƯA FILE ẢNH THẲNG VÀO MODEL?
# ============================================================

print("\n14. TẠI SAO KHÔNG ĐƯA FILE ẢNH THẲNG VÀO MODEL?")
print("""
Vì model chỉ xử lý được dữ liệu số.

File ảnh trên đĩa:
- chỉ là một file
- chưa phải tensor phù hợp để tính toán

Muốn dùng ảnh trong PyTorch:
- phải đọc ảnh
- biến nó thành tensor
- có thể kèm normalize / transform
""")


# ============================================================
# 15. DATASET TRONG PYTORCH LÀ GÌ?
# ============================================================

print("\n15. DATASET TRONG PYTORCH LÀ GÌ?")
print("""
Dataset trong PyTorch là cách tổ chức dữ liệu
để model có thể lấy từng mẫu một cách chuẩn hóa.

Một dataset thường chịu trách nhiệm:
- lấy input
- lấy label
- áp dụng transform nếu cần

Nhờ đó:
- việc train / validate / test trở nên nhất quán
- code dễ quản lý hơn
""")


# ============================================================
# 16. BUILT-IN DATASETS
# ============================================================

print("\n16. BUILT-IN DATASETS")
print("""
PPT có phần về built-in datasets trong PyTorch.

Ý nghĩa:
- PyTorch / torchvision đã cung cấp sẵn nhiều dataset phổ biến
- giúp người học và người làm ML dùng nhanh hơn

Ví dụ trong PPT:
- CIFAR dataset

Ưu điểm:
- tiện
- nhanh để thực hành
- không phải tự viết toàn bộ từ đầu
""")


# ============================================================
# 17. CIFAR DATASET
# ============================================================

print("\n17. CIFAR DATASET")
print("""
CIFAR là dataset rất phổ biến trong Computer Vision.

Trong PPT, CIFAR được dùng như ví dụ cho built-in datasets.

Ý nghĩa của việc học CIFAR:
- làm quen với bài toán classification
- làm quen với dataset ảnh màu
- làm quen với pipeline PyTorch
""")


# ============================================================
# 18. IMAGEFOLDER
# ============================================================

print("\n18. IMAGEFOLDER")
print("""
PPT cũng có phần về ImageFolder.

ImageFolder là cách rất phổ biến để tạo dataset từ thư mục ảnh.

Ý tưởng:
- mỗi class nằm trong một thư mục riêng
- PyTorch đọc cấu trúc thư mục đó để tạo dataset

Ví dụ:
data/
    cat/
        a.jpg
        b.jpg
    dog/
        c.jpg
        d.jpg

Ưu điểm:
- đơn giản
- trực quan
- rất phù hợp cho bài toán image classification cơ bản
""")


# ============================================================
# 19. VÍ DỤ TẠO DATASET VỚI IMAGEFOLDER
# ============================================================

print("\n19. VÍ DỤ TẠO DATASET VỚI IMAGEFOLDER")

from torchvision import datasets, transforms

transform = transforms.ToTensor()

# Chỉ minh họa cú pháp, không chạy nếu không có thư mục thật
print("""
Ví dụ PyTorch:

dataset = datasets.ImageFolder(
    root='data/train',
    transform=transforms.ToTensor()
)

Ý nghĩa:
- root: thư mục dữ liệu
- transform: chuyển ảnh thành tensor
""")


# ============================================================
# 20. TRANSFORM TRONG PYTORCH
# ============================================================

print("\n20. TRANSFORM TRONG PYTORCH")
print("""
Transform là các bước xử lý dữ liệu khi load ảnh.

Ví dụ:
- resize
- ToTensor()
- normalize

Transform rất quan trọng vì:
- giúp dữ liệu đầu vào thống nhất
- gắn trực tiếp preprocessing vào pipeline load dữ liệu
""")


# ============================================================
# 21. MỐI LIÊN HỆ GIỮA TENSOR, DATASET VÀ MODEL
# ============================================================

print("\n21. MỐI LIÊN HỆ GIỮA TENSOR, DATASET VÀ MODEL")
print("""
Pipeline có thể hiểu ngắn gọn như sau:

Ảnh trên đĩa
-> Dataset đọc ảnh
-> Transform chuyển thành tensor
-> Tensor đi vào model
-> Model tạo prediction
-> Tính loss
-> Update weights

Đây là cách các phần kiến thức trước giờ ghép lại với nhau.
""")


# ============================================================
# 22. TRAIN / VALIDATE / TEST KHÁC NHAU THẾ NÀO?
# ============================================================

print("\n22. TRAIN / VALIDATE / TEST KHÁC NHAU THẾ NÀO?")
print("""
Train:
- dùng để học
- update weights

Validate:
- dùng để theo dõi trong quá trình train
- không update weights

Test:
- dùng để đánh giá cuối cùng
- không update weights

Đây là cách chia dữ liệu và quy trình rất quan trọng trong Deep Learning.
""")


# ============================================================
# 23. NHỮNG NHẦM LẪN HAY GẶP
# ============================================================

print("\n23. NHỮNG NHẦM LẪN HAY GẶP")
print("""
Nhầm lẫn 1:
Tensor chỉ là ma trận 2D
-> Sai
Tensor có thể có nhiều chiều

Nhầm lẫn 2:
Ảnh đọc từ file đã tự động là tensor
-> Sai
Cần bước chuyển đổi phù hợp

Nhầm lẫn 3:
Validation và Test là một
-> Sai
Hai tập này có mục đích khác nhau

Nhầm lẫn 4:
Dataset chỉ là nơi chứa ảnh
-> Sai
Dataset còn định nghĩa cách đọc và trả dữ liệu
""")


# ============================================================
# 24. KẾT NỐI VỚI CÁC PHẦN TIẾP THEO
# ============================================================

print("\n24. KẾT NỐI VỚI CÁC PHẦN TIẾP THEO")
print("""
Sau khi hiểu pipeline PyTorch cơ bản,
ta đã sẵn sàng để đi sâu hơn vào:

- Convolutional Neural Network (CNN)
- Image Classification thực chiến hơn
- các bài toán vision nâng cao như Detection, Segmentation
""")


# ============================================================
# 25. TỔNG KẾT
# ============================================================

print("\n25. TỔNG KẾT")
print("""
1. Pipeline train model trong PyTorch gồm 7 bước:
   dataset -> model -> loss -> optimizer -> train -> validate -> test
2. Tensor là cấu trúc dữ liệu cơ bản của PyTorch
3. Ảnh cần được chuyển thành tensor trước khi đưa vào model
4. Dataset giúp tổ chức dữ liệu chuẩn hóa
5. Built-in datasets và ImageFolder rất hữu ích cho học và thực hành
6. Train / Validate / Test có vai trò khác nhau rõ ràng
""")