import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 自定义数据集类
class ChestXRayDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        初始化数据集
        :param image_dir: 图像目录路径
        :param transform: 图像预处理操作
        """
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # 遍历目录，收集所有图像路径和标签
        for label in ['NORMAL', 'PNEUMONIA']:
            label_dir = os.path.join(image_dir, label)
            for fname in os.listdir(label_dir):
                self.image_paths.append(os.path.join(label_dir, fname))
                self.labels.append(0 if label == 'NORMAL' else 1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载图像
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# 数据预处理流程
def get_transforms():
    return transforms.Compose([
        # 随机裁剪为224x224
        transforms.RandomResizedCrop(224),
        # 随机旋转-30到+30度
        transforms.RandomRotation(30),
        # 随机水平翻转
        transforms.RandomHorizontalFlip(),
        # 随机调整图像亮度、对比度、饱和度、色调
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        # 随机缩放
        transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),
        # 转换为张量
        transforms.ToTensor(),
        # 归一化处理
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# 设置路径
base_dir = r'C:\Users\32059\PycharmProjects\Medical Treatment\chest_xray'  # 实际路径

train_image_dir = os.path.join(base_dir, 'train')
val_image_dir = os.path.join(base_dir, 'val')
test_image_dir = os.path.join(base_dir, 'test')

# 创建训练集、验证集和测试集数据集
train_dataset = ChestXRayDataset(image_dir=train_image_dir, transform=get_transforms())
val_dataset = ChestXRayDataset(image_dir=val_image_dir, transform=get_transforms())
test_dataset = ChestXRayDataset(image_dir=test_image_dir, transform=get_transforms())

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 可视化预处理后的图像
def visualize_samples():
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    # 选择第一张图像
    img = images[0].numpy().transpose((1, 2, 0))  # 转换为HWC格式
    img = np.clip(img * 0.225 + 0.485, 0, 1)  # 反归一化
    plt.imshow(img)
    plt.title('Sample X-ray Image')
    plt.axis('off')
    plt.show()

# 展示处理后的样本
visualize_samples()

