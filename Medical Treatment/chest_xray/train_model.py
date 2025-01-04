import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from model import SimpleCNN  # 自己的模型
from data_preprocessing import ChestXRayDataset, get_transforms

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据集
base_dir = r'C:\Users\32059\PycharmProjects\Medical Treatment\chest_xray'  # 实际路径
train_image_dir = os.path.join(base_dir, 'train')
val_image_dir = os.path.join(base_dir, 'val')

train_dataset = ChestXRayDataset(image_dir=train_image_dir, transform=get_transforms())
val_dataset = ChestXRayDataset(image_dir=val_image_dir, transform=get_transforms())

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # 批次大小设置为16
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 初始化模型
model = SimpleCNN(num_classes=2).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 学习率调度器：ReduceLROnPlateau
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)


# 早停实现
def train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=10, patience=5):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)
        scheduler.step(val_loss)  # 更新学习率

        # 记录损失
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # 早停逻辑
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), 'best_model.pth')  # 保存最佳模型
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping...")
                break

    return train_losses, val_losses


# 训练模型
train_losses, val_losses = train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, scheduler)

# 绘制损失曲线
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.show()
