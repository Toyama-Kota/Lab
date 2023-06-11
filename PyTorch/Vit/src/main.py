import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from einops import repeat
from einops.layers.torch import Rearrange

from vit import ViT

# 学習率
LEARNING_RATE = 0.01
# モーメンタム
MOMENTUM = 0.9
# エポック
EPOCH_NUM = 10
# バッチサイズ
BATCH_SIZE = 100
# 画像サイズ
IMAGE_SIZE = 32
# パッチサイズ
PATCH_SIZE = 4
# エンコーダブロック数
ENCODER_LAYER_NUM = 3
# ヘッダー数
HEAD_NUM = 4
# パッチベクトルの長さ
VEC_LENGTH = 256
# MLPベクトルの長さ
MLP_VEC_LENGTH = 256
# クラス数
CLASS_NUM = 10
# クラス
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

epoch_train_loss = 0
epoch_train_acc = 0
epoch_test_loss = 0
epoch_test_acc = 0

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10(
    root = '../data', 
    train = True, 
    download = True, 
    transform = transform
    )

test_set = torchvision.datasets.CIFAR10(
    root='../data', 
    train = False, 
    download = True, 
    transform = transform
    )

train_loader = torch.utils.data.DataLoader(
    train_set, 
    batch_size = BATCH_SIZE, 
    shuffle = True, 
    num_workers = 2
    )

test_loader = torch.utils.data.DataLoader(
    test_set, 
    batch_size = BATCH_SIZE, 
    shuffle = False, 
    num_workers = 2
    )

def train(data, optimizer):
    global epoch_train_loss
    global epoch_train_acc

    inputs, labels = data[0].to(device), data[1].to(device)

    optimizer.zero_grad()

    outputs = net(inputs)
    loss = loss_func(outputs, labels)
    loss.backward()
    optimizer.step()

    epoch_train_loss += loss.item()/len(train_loader)
    acc = (outputs.argmax(dim=1) == labels).float().mean()
    epoch_train_acc += acc/len(train_loader)

    del inputs
    del outputs
    del loss

def test(data):
    global epoch_test_loss
    global epoch_test_acc

    inputs, labels = data[0].to(device), data[1].to(device)
    outputs = net(inputs)

    loss = loss_func(outputs, labels)
    epoch_test_loss += loss.item()/len(test_loader)

    test_acc = (outputs.argmax(dim=1) == labels).float().mean()
    epoch_test_acc += test_acc/len(test_loader)


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ViTモデル
    net = ViT(
        image_size = IMAGE_SIZE,
        patch_size = PATCH_SIZE,
        class_num = CLASS_NUM,
        vec_length = VEC_LENGTH,
        encoder_layer_num = ENCODER_LAYER_NUM,
        head_num = HEAD_NUM,
        mlp_vec_length = MLP_VEC_LENGTH
    ).to(device)

    # 損失関数
    loss_func = nn.CrossEntropyLoss()

    # オプティマイザー
    optimizer = optim.SGD(
        net.parameters(), 
        lr=LEARNING_RATE, 
        momentum=MOMENTUM
        )

    for epoch in range(0, EPOCH_NUM):

        epoch_train_loss = 0
        epoch_train_acc = 0
        epoch_test_loss = 0
        epoch_test_acc = 0

        net.train()
        for data in train_loader:
            train(data, optimizer)
        print(f'Epoch: {epoch+1} train_acc: {epoch_train_acc:.2f} train_loss {epoch_train_loss:.2f}')

        net.eval()
        with torch.no_grad():
            for data in test_loader:
                test(data)
        print(f'Epoch: {epoch+1} test_acc : {epoch_test_acc:.2f}  test_loss  {epoch_test_loss:.2f}')

