"""
EX7 - Gradient và Optimization

Mục tiêu:
1. Hiểu gradient là gì
2. Hiểu Gradient Descent hoạt động ra sao
3. Hiểu cách cập nhật tham số của model
4. Phân biệt:
   - Batch Gradient Descent
   - Stochastic Gradient Descent (SGD)
   - Mini-batch Gradient Descent
5. Hiểu các khó khăn của Gradient Descent
6. Hiểu Momentum, Adaptive Gradient Descent và RMSProp

Tài liệu gốc:
- PPT Deep Learning / Computer Vision
"""


# ============================================================
# 1. GRADIENT LÀ GÌ?
# ============================================================

print("1. GRADIENT LÀ GÌ?")
print("""
Gradient là vector đạo hàm riêng của hàm loss theo các tham số.

Hiểu đơn giản:
- Gradient cho biết loss đang tăng hay giảm theo hướng nào
- Nó cũng cho biết độ dốc lớn hay nhỏ

Nếu loss giống như một ngọn đồi:
- gradient chỉ hướng đi lên dốc nhanh nhất
- muốn giảm loss, ta phải đi theo hướng ngược lại gradient

Đó là lý do trong tối ưu:
- không đi cùng gradient
- mà đi ngược gradient
""")


# ============================================================
# 2. VÌ SAO CẦN GRADIENT?
# ============================================================

print("\n2. VÌ SAO CẦN GRADIENT?")
print("""
Model có rất nhiều tham số:
- weight
- bias

Muốn model tốt hơn:
- ta cần thay đổi các tham số đó
- nhưng thay đổi như thế nào?

Gradient giúp trả lời:
- tham số nào nên tăng
- tham số nào nên giảm
- giảm/tăng bao nhiêu
""")


# ============================================================
# 3. GRADIENT DESCENT LÀ GÌ?
# ============================================================

print("\n3. GRADIENT DESCENT LÀ GÌ?")
print("""
Gradient Descent là thuật toán tối ưu cơ bản dùng để giảm loss.

Ý tưởng:
- tính gradient của loss
- cập nhật tham số theo hướng ngược gradient
- lặp lại nhiều lần

Mục tiêu:
- tìm bộ tham số tốt hơn
- làm cho loss nhỏ dần
""")


# ============================================================
# 4. CÔNG THỨC CẬP NHẬT THAM SỐ
# ============================================================

print("\n4. CÔNG THỨC CẬP NHẬT THAM SỐ")
print("""
Công thức cơ bản:

w = w - learning_rate * gradient

Trong đó:
- w: tham số hiện tại
- learning_rate: tốc độ học
- gradient: độ dốc của loss theo w

Ý nghĩa:
- nếu gradient dương -> giảm w
- nếu gradient âm -> tăng w

Ta luôn đi theo hướng làm loss giảm.
""")


# ============================================================
# 5. VÍ DỤ NHỎ VỀ CẬP NHẬT THAM SỐ
# ============================================================

print("\n5. VÍ DỤ NHỎ VỀ CẬP NHẬT THAM SỐ")

w = 5.0
gradient = 2.0
learning_rate = 0.1

new_w = w - learning_rate * gradient

print("w cũ =", w)
print("gradient =", gradient)
print("learning_rate =", learning_rate)
print("w mới =", new_w)


# ============================================================
# 6. LEARNING RATE LÀ GÌ?
# ============================================================

print("\n6. LEARNING RATE LÀ GÌ?")
print("""
Learning Rate là độ lớn của mỗi bước cập nhật.

Nếu learning rate quá lớn:
- bước đi quá dài
- dễ vượt qua điểm tốt
- loss có thể dao động hoặc diverge

Nếu learning rate quá nhỏ:
- học rất chậm
- tốn nhiều thời gian

Vì vậy:
- learning rate là hyperparameter cực kỳ quan trọng
""")


# ============================================================
# 7. GRADIENT DESCENT TRONG LINEAR CLASSIFIER VS NEURAL NETWORK
# ============================================================

print("\n7. GRADIENT DESCENT TRONG LINEAR CLASSIFIER VS NEURAL NETWORK")
print("""
Trong PPT có slide so sánh Linear Classifier và Neural Network.

Ý chính:
- Linear Classifier có bề mặt loss đơn giản hơn
- Neural Network có bề mặt loss phức tạp hơn rất nhiều

Điều đó dẫn đến:
- tối ưu Linear Classifier thường dễ hơn
- tối ưu Neural Network khó hơn vì có nhiều điểm gồ ghề, nhiều vùng phức tạp
""")


# ============================================================
# 8. BATCH GRADIENT DESCENT
# ============================================================

print("\n8. BATCH GRADIENT DESCENT")
print("""
Batch Gradient Descent:
- dùng toàn bộ dữ liệu training để tính gradient
- sau đó mới cập nhật tham số

Ưu điểm:
- gradient ổn định hơn
- đi theo hướng tổng quát của toàn bộ dữ liệu

Nhược điểm:
- rất chậm nếu dataset lớn
- tốn bộ nhớ
- mỗi lần update phải đi qua toàn bộ tập dữ liệu
""")


# ============================================================
# 9. STOCHASTIC GRADIENT DESCENT (SGD)
# ============================================================

print("\n9. STOCHASTIC GRADIENT DESCENT (SGD)")
print("""
SGD:
- dùng từng datapoint một để tính gradient
- sau đó cập nhật tham số ngay

Ưu điểm:
- nhanh
- cập nhật thường xuyên
- có thể thoát khỏi một số vùng xấu nhờ tính nhiễu

Nhược điểm:
- gradient rất nhiễu
- loss dao động mạnh
- đường đi không mượt
""")


# ============================================================
# 10. MINI-BATCH GRADIENT DESCENT
# ============================================================

print("\n10. MINI-BATCH GRADIENT DESCENT")
print("""
Mini-batch Gradient Descent:
- dùng một nhóm nhỏ dữ liệu (batch) để tính gradient
- sau đó cập nhật tham số

Đây là cách phổ biến nhất trong Deep Learning hiện đại.

Ưu điểm:
- nhanh hơn batch
- ổn định hơn SGD từng điểm
- tận dụng GPU tốt hơn

Ví dụ:
- batch size = 32
- mỗi lần lấy 32 mẫu để tính gradient
""")


# ============================================================
# 11. SO SÁNH 3 LOẠI GRADIENT DESCENT
# ============================================================

print("\n11. SO SÁNH 3 LOẠI GRADIENT DESCENT")
print("""
Batch GD:
- dùng toàn bộ dữ liệu
- ổn định
- chậm

SGD:
- dùng 1 datapoint
- nhanh
- nhiễu

Mini-batch GD:
- dùng 1 nhóm nhỏ dữ liệu
- cân bằng giữa tốc độ và độ ổn định

=> trong thực tế, mini-batch là lựa chọn phổ biến nhất
""")


# ============================================================
# 12. HOW PARAMETERS ARE UPDATED?
# ============================================================

print("\n12. HOW PARAMETERS ARE UPDATED?")
print("""
Quy trình cập nhật tham số thường là:

Bước 1:
- forward pass
- model dự đoán output

Bước 2:
- tính loss

Bước 3:
- tính gradient

Bước 4:
- cập nhật parameters:
  w = w - lr * grad

Bước 5:
- lặp lại rất nhiều lần

Đây chính là vòng học cơ bản của neural network.
""")


# ============================================================
# 13. CHALLENGE 1 - LOSS FUNCTION LANDSCAPE
# ============================================================

print("\n13. CHALLENGE 1 - LOSS LANDSCAPE")
print("""
Trong PPT, phần challenge nhấn mạnh rằng bề mặt loss không đơn giản.

Loss surface có thể:
- gồ ghề
- nhiều vùng cong
- nhiều điểm không thuận lợi cho tối ưu

Điều này làm cho Gradient Descent không phải lúc nào cũng đi thẳng đến nghiệm tốt nhất.
""")


# ============================================================
# 14. CHALLENGE 2 - LOCAL MINIMA
# ============================================================

print("\n14. CHALLENGE 2 - LOCAL MINIMA")
print("""
Local minima là điểm mà loss nhỏ trong một vùng nhỏ,
nhưng chưa chắc là nhỏ nhất toàn cục.

Hiểu đơn giản:
- bạn đang xuống đồi
- tới một hõm nhỏ
- nhưng đó chưa phải điểm thấp nhất của cả khu vực

Với neural network:
- local minima là một trong các khó khăn khi tối ưu
""")


# ============================================================
# 15. CHALLENGE 3 - SADDLE POINTS
# ============================================================

print("\n15. CHALLENGE 3 - SADDLE POINTS")
print("""
Saddle point là điểm mà:
- theo một hướng thì giống minimum
- theo hướng khác thì lại không phải minimum

Đây là vùng dễ làm gradient rất nhỏ,
khiến việc học chậm lại.

Trong mạng sâu, saddle point thường là vấn đề rất quan trọng.
""")


# ============================================================
# 16. CHALLENGE 4 - RAVINES
# ============================================================

print("\n16. CHALLENGE 4 - RAVINES")
print("""
Ravines là vùng mà:
- theo một chiều thì loss thay đổi rất mạnh
- theo chiều khác thì thay đổi chậm

Kết quả:
- gradient descent dễ dao động qua lại
- đi zig-zag
- học chậm

Đây là lý do cần các optimizer tốt hơn Gradient Descent cơ bản.
""")


# ============================================================
# 17. MOMENTUM
# ============================================================

print("\n17. MOMENTUM")
print("""
Momentum thêm "quán tính" vào quá trình cập nhật.

Ý tưởng:
- không chỉ nhìn gradient hiện tại
- mà còn nhớ hướng đi trước đó

Lợi ích:
- giảm dao động
- đi mượt hơn
- tăng tốc ở những hướng đúng
- đặc biệt hữu ích trong ravines

Hiểu trực quan:
- như một quả bóng đang lăn xuống dốc
- nó có quán tính nên không đổi hướng đột ngột quá mạnh
""")


# ============================================================
# 18. ADAPTIVE GRADIENT DESCENT
# ============================================================

print("\n18. ADAPTIVE GRADIENT DESCENT")
print("""
Trong PPT có chuỗi slide về Adaptive Gradient Descent.

Ý tưởng chung:
- không dùng cùng một learning rate cố định cho mọi tham số
- mỗi tham số có thể có cách cập nhật khác nhau

Lý do:
- có tham số cần bước lớn
- có tham số cần bước nhỏ

Nhờ đó:
- tối ưu linh hoạt hơn
- học tốt hơn trên dữ liệu phức tạp
""")


# ============================================================
# 19. RMSPROP
# ============================================================

print("\n19. RMSPROP")
print("""
RMSProp là một optimizer thích nghi.

Ý tưởng:
- lưu thông tin về độ lớn gradient gần đây
- dùng nó để điều chỉnh learning rate hiệu quả hơn

Ưu điểm:
- giảm dao động
- phù hợp với loss surface phức tạp
- thường train ổn định hơn gradient descent cơ bản

RMSProp là một bước phát triển rất quan trọng trong nhóm optimizer thích nghi.
""")


# ============================================================
# 20. VÍ DỤ TRỰC QUAN NHỎ VỀ GRADIENT DESCENT
# ============================================================

print("\n20. VÍ DỤ TRỰC QUAN NHỎ VỀ GRADIENT DESCENT")

def f(x):
    return x**2

def grad_f(x):
    return 2 * x

x = 5.0
lr = 0.1

for step in range(5):
    gradient = grad_f(x)
    x = x - lr * gradient
    print(f"Step {step+1}: x = {x:.4f}, loss = {f(x):.4f}")


# ============================================================
# 21. NHỮNG NHẦM LẪN HAY GẶP
# ============================================================

print("\n21. NHỮNG NHẦM LẪN HAY GẶP")
print("""
Nhầm lẫn 1:
Gradient Descent luôn tìm ra global minimum
-> Sai

Nhầm lẫn 2:
Learning rate càng lớn càng học nhanh và càng tốt
-> Sai

Nhầm lẫn 3:
SGD luôn tốt hơn Batch hoặc ngược lại
-> Không tuyệt đối
Tùy bài toán và dữ liệu

Nhầm lẫn 4:
Optimizer chỉ là chi tiết nhỏ
-> Sai
Optimizer ảnh hưởng rất mạnh tới việc train có ổn định hay không
""")


# ============================================================
# 22. KẾT NỐI SANG PHẦN TIẾP THEO
# ============================================================

print("\n22. KẾT NỐI SANG PHẦN TIẾP THEO")
print("""
Đến đây ta đã biết:
- loss là gì
- gradient là gì
- optimizer giảm loss thế nào

Câu hỏi tiếp theo là:
- gradient được tính ra sao trong mạng nhiều layer?

Đó là lý do phần tiếp theo trong PPT là:
BACKPROPAGATION
""")


# ============================================================
# 23. TỔNG KẾT
# ============================================================

print("\n23. TỔNG KẾT")
print("""
1. Gradient cho biết hướng và độ lớn thay đổi của loss
2. Gradient Descent cập nhật tham số theo hướng ngược gradient
3. Learning rate quyết định độ dài mỗi bước đi
4. Có 3 kiểu phổ biến:
   - Batch GD
   - SGD
   - Mini-batch GD
5. Gradient Descent gặp khó với:
   - local minima
   - saddle points
   - ravines
6. Momentum, Adaptive Gradient Descent và RMSProp giúp tối ưu tốt hơn
""")