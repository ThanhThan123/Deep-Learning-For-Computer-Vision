import torch
import torch.nn.functional as F

# =========================================================
# 1) Create tensors
# =========================================================

# 1.1 3D tensor of shape 20x30x40 with all values = 0
tensor_3d_zeros = torch.zeros(20, 30, 40)
print("tensor_3d_zeros shape:", tensor_3d_zeros.shape)

# 1.2 1D tensor containing the even numbers between 10 and 100
even_tensor = torch.arange(10, 101, 2)
print("even_tensor:", even_tensor)


# =========================================================
# 2) Sum of elements / rows / columns
# =========================================================

x = torch.rand(4, 6)

sum_all = torch.sum(x)         # tổng tất cả phần tử
sum_columns = torch.sum(x, 0)  # cộng theo cột -> tensor 6 phần tử
sum_rows = torch.sum(x, 1)     # cộng theo hàng -> tensor 4 phần tử

print("\nx =\n", x)
print("Sum of all elements:", sum_all)
print("Sum of columns:", sum_columns)
print("Sum of rows:", sum_rows)


# =========================================================
# 3) Cosine similarity between 2 1D tensors
# =========================================================

x1 = torch.tensor([0.1, 0.3, 2.3, 0.45])
y1 = torch.tensor([0.13, 0.23, 2.33, 0.45])

cos_sim_1d = F.cosine_similarity(x1, y1, dim=0)
print("\nCosine similarity (1D):", cos_sim_1d)


# =========================================================
# 4) Cosine similarity between 2 2D tensors
# =========================================================

x2 = torch.tensor([[ 0.2714, 1.1430, 1.3997, 0.8788],
                   [-2.2268, 1.9799, 1.5682, 0.5850],
                   [ 1.2289, 0.5043, -0.1625, 1.1403]])

y2 = torch.tensor([[-0.3299, 0.6360, -0.2014, 0.5989],
                   [-0.6679, 0.0793, -2.5842, -1.5123],
                   [ 1.1110, -0.1212, 0.0324, 1.1277]])

# cosine similarity theo từng hàng
cos_sim_2d = F.cosine_similarity(x2, y2, dim=1)
print("\nCosine similarity (2D, row-wise):", cos_sim_2d)


# =========================================================
# 5) Reshape tensor
# =========================================================

x3 = torch.tensor([[ 0,  1],
                   [ 2,  3],
                   [ 4,  5],
                   [ 6,  7],
                   [ 8,  9],
                   [10, 11]])

# Make x become 1D tensor
x3_1d = x3.reshape(-1)

# Then make that 1D tensor become 3x4 2D tensor
x3_2d = x3_1d.reshape(3, 4)

print("\nOriginal x3 =\n", x3)
print("x3 as 1D:", x3_1d)
print("x3 as 3x4 =\n", x3_2d)


# =========================================================
# 6) Unsqueeze, resize, concat
# =========================================================

x4 = torch.rand(3, 1080, 1920)
y4 = torch.rand(3, 720, 1280)

# 6.1 Make x become 1x3x1080x1920
x4_4d = x4.unsqueeze(0)

# 6.2 Make y become 1x3x720x1280
y4_4d = y4.unsqueeze(0)

# 6.3 Resize y to have same size as x
# interpolate expects 4D input: [N, C, H, W]
y4_resized = F.interpolate(y4_4d, size=(1080, 1920), mode='bilinear', align_corners=False)

# 6.4 Join them to become 2x3x1080x1920
joined = torch.cat([x4_4d, y4_resized], dim=0)

print("\nx4_4d shape:", x4_4d.shape)
print("y4_4d shape:", y4_4d.shape)
print("y4_resized shape:", y4_resized.shape)
print("joined shape:", joined.shape)