import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
from model import SimpleCNN  #自己的模型
from data_preprocessing import ChestXRayDataset, get_transforms

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载测试集数据
base_dir = r'C:\Users\32059\PycharmProjects\Medical Treatment\chest_xray'  # 实际路径
test_image_dir = os.path.join(base_dir, 'test')

test_dataset = ChestXRayDataset(image_dir=test_image_dir, transform=get_transforms())
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 初始化模型
model = SimpleCNN(num_classes=2).to(device)

# 加载训练过程中保存的最佳模型
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# 模型评估函数
def evaluate_model(model, test_loader):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 预测
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # 存储预测结果和真实标签
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算各类指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')  # binary for 2 classes
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')

    # 输出评估结果
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return accuracy, precision, recall, f1, all_labels, all_preds


# 绘制混淆矩阵
def plot_confusion_matrix(all_labels, all_preds, labels=['Normal', 'Pneumonia']):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

# 评估模型
accuracy, precision, recall, f1, all_labels, all_preds = evaluate_model(model, test_loader)

# 绘制混淆矩阵
plot_confusion_matrix(all_labels, all_preds)

# 保存评估报告
with open('evaluation_report.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")

# 讨论模型的优缺点（可以在文件中手动补充，或通过代码自动生成）
with open('evaluation_report.txt', 'a') as f:
    f.write("\nDiscussion:\n")
    f.write("The model demonstrates a balanced performance with reasonable accuracy, precision, recall, and F1 score. "
            "However, some areas for improvement could be explored, such as tuning the model architecture, using data augmentation, or training with a larger dataset.\n")

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import os
# from model import SimpleCNN  # 假设SimpleCNN是你自己的模型
# from data_preprocessing import ChestXRayDataset, get_transforms  # 假设数据预处理在这两个文件中
#
# # 设置设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # 加载测试集数据
# base_dir = r'C:\Users\32059\PycharmProjects\Medical Treatment\chest_xray'  # 替换为实际路径
# test_image_dir = os.path.join(base_dir, 'test')
#
# test_dataset = ChestXRayDataset(image_dir=test_image_dir, transform=get_transforms())
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
#
# # 初始化模型
# model = SimpleCNN(num_classes=2).to(device)
#
# # 加载训练过程中保存的最佳模型
# model.load_state_dict(torch.load('best_model.pth'))
# model.eval()
#
#
# # 模型评估函数
# def evaluate_model(model, test_loader):
#     all_preds = []
#     all_labels = []
#
#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#
#             # 预测
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#
#             # 存储预测结果和真实标签
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
#
#     # 计算各类指标
#     accuracy = accuracy_score(all_labels, all_preds)
#     precision = precision_score(all_labels, all_preds, average='binary')  # binary for 2 classes
#     recall = recall_score(all_labels, all_preds, average='binary')
#     f1 = f1_score(all_labels, all_preds, average='binary')
#
#     # 输出评估结果
#     print(f"Accuracy: {accuracy:.4f}")
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall: {recall:.4f}")
#     print(f"F1 Score: {f1:.4f}")
#
#     return accuracy, precision, recall, f1
#
#
# # 评估模型
# accuracy, precision, recall, f1 = evaluate_model(model, test_loader)
#
# # 保存评估报告
# with open('evaluation_report.txt', 'w') as f:
#     f.write(f"Accuracy: {accuracy:.4f}\n")
#     f.write(f"Precision: {precision:.4f}\n")
#     f.write(f"Recall: {recall:.4f}\n")
#     f.write(f"F1 Score: {f1:.4f}\n")
#
# # 讨论模型的优缺点（可以在文件中手动补充，或通过代码自动生成）
# with open('evaluation_report.txt', 'a') as f:
#     f.write("\nDiscussion:\n")
#     f.write("The model demonstrates a balanced performance with reasonable accuracy, precision, recall, and F1 score. "
#             "However, some areas for improvement could be explored, such as tuning the model architecture, using data augmentation, or training with a larger dataset.\n")
