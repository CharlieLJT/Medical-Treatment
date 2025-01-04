import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        """
        初始化SimpleCNN模型
        :param num_classes: 分类数，默认2类（正常与肺炎）
        """
        super(SimpleCNN, self).__init__()

        # 卷积层 1: 输入通道 3 (RGB)，输出通道 32，卷积核 3x3
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化，2x2

        # 卷积层 2: 输入通道 32，输出通道 64，卷积核 3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 卷积层 3: 输入通道 64，输出通道 128，卷积核 3x3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层
        self.fc1 = nn.Linear(128 * 28 * 28, 512)  #池化后的大小为128 * 28 * 28
        self.fc2 = nn.Linear(512, num_classes)  # 最终输出分类数

        # Dropout层
        self.dropout = nn.Dropout(0.5)  # 50% Dropout概率

    def forward(self, x):
        """
        定义前向传播过程
        :param x: 输入数据
        :return: 输出结果
        """
        # 卷积 + 激活 + 池化
        x = self.pool1(F.relu(self.conv1(x)))  # (B, 32, 112, 112)
        x = self.pool2(F.relu(self.conv2(x)))  # (B, 64, 56, 56)
        x = self.pool3(F.relu(self.conv3(x)))  # (B, 128, 28, 28)

        # 展平操作，转换为全连接层输入
        x = x.view(-1, 128 * 28 * 28)  # 将特征图展平为1D向量，大小应为 (B, 128 * 28 * 28)

        # 全连接层 + Dropout
        x = self.dropout(F.relu(self.fc1(x)))  # (B, 512)
        x = self.fc2(x)  # (B, num_classes)

        return x

# 模型实例化
model = SimpleCNN(num_classes=2)  # 2类：正常与肺炎
print(model)
