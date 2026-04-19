import pandas as pd
import numpy as np
from sympy.physics.control.control_plots import plt
    
# Lấy ảnh test từ banana
file_path = './full_numpy_bitmap_banana.npy'
test_images = np.load(file_path).astype('float32')
test_image = test_images[100]

categories = ['apple', 'banana', 'bicycle']
scores = []
weights = []

for category in categories:
    file_path=f'./full_numpy_bitmap_{category}.npy'
    images = np.load(file_path).astype('float32')

    #  Ảnh trung bình của class -> đóng vai trò như "weight prototype"
    avg_image = np.mean(images, axis=0)

    weights.append(avg_image)
    # So sánh giữa mẫu và class
    scores.append(test_image @ avg_image)

print(scores)

# print(f' the test_images is most likely {categories[np.argmax(scores)]}')

# Hiển thị weight / average image của từng class
plt.figure(figsize = (10, 4))
for i in range(len(weights)):
    plt.subplot(2, 5, i+1)
    plt.imshow(weights[i].reshape(28, 28))
    plt.axis('off')
    plt.title(categories[i])
plt.show()




