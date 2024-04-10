import torch.nn as nn
import torch
from model import src as tf
import math
import torch.nn.functional as F

channel_size = 1
# ——————————配置调用设备————————————
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 数据放到gpu上还是cpu上
print("device", dev)


class TFClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TFClassifier, self).__init__()
        self.input_size = input_size

        # Transformer Encoder
        self.encoder_layer1 = tf.EncoderLayer(d_model=input_size, gamma=0.1, head_size = channel_size,ffn_hidden=2048, drop_prob=0.1,use_XPOS=False)
        self.encoder_layer2 = tf.EncoderLayer(d_model=math.ceil(input_size * 0.25), gamma= 0.3, head_size = channel_size * 2,
                                              ffn_hidden=2048, drop_prob=0.1,use_XPOS=False)
        self.encoder_layer3 = tf.EncoderLayer(d_model=math.ceil(math.ceil(input_size * 0.25) * 0.25), gamma= 0.5, head_size = channel_size * 4,
                                              ffn_hidden=2048,drop_prob=0.1,use_XPOS=False)

        # encoder
        self.conv_downsample1 = nn.Conv1d(channel_size, channel_size * 2, kernel_size=3, stride=4,
                                          padding=1)  # 卷积下采样替代最大池化
        self.conv_downsample2 = nn.Conv1d(channel_size* 2, channel_size * 4, kernel_size=3, stride=4,
                                          padding=1)  # 卷积下采样替代最大池化
        self.conv_downsample3 = nn.Conv1d(channel_size * 4, output_size, kernel_size=3, stride=4,
                                          padding=1)  # 卷积下采样替代最大池化
        self.bn1 = nn.BatchNorm1d(channel_size * 2)  # 添加Batch Normalization
        self.bn2 = nn.BatchNorm1d(channel_size * 4)  # 添加Batch Normalization
        self.bn3 = nn.BatchNorm1d(output_size)  # 添加Batch Normalization

        # Estimation network
        self.estimation_net = nn.Sequential(
            #nn.Linear(output_size*2, output_size),
            nn.Linear(output_size*4, output_size),
        )

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        x = x.unsqueeze(1)

        x_enc1 = self.encoder_layer1(x, src_mask=None)  # transformer特征提取[64,512,20]
        x = self.conv_downsample1(x_enc1)  # 下采样[512,64,59]
        x = self.bn1(x)

        x_enc2 = self.encoder_layer2(x, src_mask=None)  # transformer特征提取[128,512,7]
        x = self.conv_downsample2(x_enc2)  # 卷积下采样减少特征[512,20,128]
        x = self.bn2(x)

        x_enc3 = self.encoder_layer3(x, src_mask=None)  # transformer特征提取[256,512,3]
        x = self.conv_downsample3(x_enc3)  # 卷积下采样减少特征[512,7,256]
        x = self.bn3(x)

        # decoder
        #x = x.permute(0, 2, 1)
        # 计算沿第二个维度的平均值，将其强制降为1维
        #x = x.mean(dim=1, keepdim=True)
        #三维拍扁为2维
        x_dec = x.reshape(x.size(0), -1)
        #x_dec = x.squeeze(1)
        # 分类器
        y_hat = self.estimation_net(x_dec)

        return x_dec, y_hat  # [512,58]

    # 新增：初始化状态的方法
    def init_states(self, batch_size):
        # 初始化每个编码器层的状态
        states = {
            'encoder_layer1': torch.zeros(batch_size, 1, self.input_size).to(dev),
            'encoder_layer2': torch.zeros(batch_size, 1, math.ceil(self.input_size * 0.25)).to(dev),
            'encoder_layer3': torch.zeros(batch_size, 1, math.ceil(math.ceil(self.input_size * 0.25) * 0.25)).to(dev),
        }
        return states

    # 新增：递归前馈方法
    def forward_recurrent(self, x, states):
        """
        x: 当前时间步的输入，形状为(batch_size, 1, input_size)
        states: 包含每一层状态的字典
        """
        # 递归调用每个编码器层
        x = x.unsqueeze(1)
        x, states['encoder_layer1'] = self.encoder_layer1.forward_recurrent(x, states['encoder_layer1'], 1)
        x = self.conv_downsample1(x)  # [512,1,58] -> [512,2,15]
        x = self.bn1(x)
        # x = F.relu(x)  # 假设有激活函数

        x, states['encoder_layer2'] = self.encoder_layer2.forward_recurrent(x, states['encoder_layer2'], 2)
        x = self.conv_downsample2(x)  # [512,2,15] -> [512,4,4]
        x = self.bn2(x)
        # x = F.relu(x)  # 假设有激活函数

        x, states['encoder_layer3'] = self.encoder_layer3.forward_recurrent(x, states['encoder_layer3'], 4)
        x = self.conv_downsample3(x)  # [512,4,4] -> [512, output_size, 1]
        x = self.bn3(x)
        # x = F.relu(x)  # 假设有激活函数

        # decoder
        #x = x.permute(0, 2, 1)
        # 计算沿第二个维度的平均值，将其强制降为1维
        #x = x.mean(dim=1, keepdim=True)
        #x_dec = x.squeeze(1)
        x_dec = x.reshape(x.size(0), -1)
        # 分类器
        y_hat = self.estimation_net(x_dec)

        # 返回最终的输出和更新的状态
        return x_dec, y_hat, states


def tf_loss(y, y_hat):
    diff_loss = nn.CrossEntropyLoss()(y_hat, y)
    # 初始化FocalLoss类
    #focal_loss = Focal_Loss(weight=[1,1,0.7,0.2,0.5,0.1,0.01,0.7,1,2,0,0,0,0,0], gamma=2)
    #diff_loss = focal_loss(y_hat,y)
    return diff_loss


class Focal_Loss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(Focal_Loss, self).__init__()
        self.gamma = gamma
        self.alpha = weight if weight is not None else 1.0

    def forward(self, preds, labels):
        """
        preds: 经softmax的预测结果，形状[batch_size, num_classes]
        labels: 真实类别索引，形状[batch_size]
        """
        # 转换labels为one-hot编码形式
        labels_one_hot = F.one_hot(labels, num_classes=preds.size(1)).float()

        # 计算交叉熵损失
        ce_loss = F.cross_entropy(preds, labels, reduction='none')

        # 计算每个类别的概率
        pt = torch.sum(labels_one_hot * preds, dim=1) + 1e-7
        # 计算Focal Loss
        f_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        return f_loss.mean()