import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch
import torch.optim as optim
from torch.autograd import Variable
import os
import natsort
from PIL import Image
import torch.nn.functional as F

tsfms = transforms.Compose([
    transforms.ToTensor()
])


class CustomDataSet(Dataset):
    def __init__(self, noised_dir, clean_dir, transform):
        self.noised_dir = noised_dir
        self.clean_dir = clean_dir
        self.transform = transform
        noised_imgs = os.listdir(noised_dir)
        clean_imgs = os.listdir(clean_dir)

        self.noised_paths = natsort.natsorted(noised_imgs)
        self.clean_paths = natsort.natsorted(clean_imgs)

    def __len__(self):
        assert len(self.noised_paths) == len(self.clean_paths)
        return len(self.noised_paths)

    def __getitem__(self, idx):
        img_noised = os.path.join(self.noised_dir, self.noised_paths[idx])
        img_clean = os.path.join(self.clean_dir, self.clean_paths[idx])

        image_n = Image.open(img_noised)
        image_c = Image.open(img_clean)

        tensor_image_n = self.transform(image_n)
        tensor_image_c = self.transform(image_c)
        return (tensor_image_n, tensor_image_c)


def double_conv(in_c, out_c):
    '''
    convolution -> ReLU -> convolution
    :param in_c: channel_in tensor
    :param out_c: channel_out tensor
    :return: tensor after these 3 operations
    '''
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, padding=1, kernel_size=3),
        nn.ReLU(True),
        nn.BatchNorm2d(out_c),
        nn.Conv2d(out_c, out_c, padding=1, kernel_size=3),
        nn.ReLU(True),
        nn.BatchNorm2d(out_c)
    )
    return conv


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop_out = nn.Dropout(0.15)

        self.down_conv_1 = double_conv(1, 112)
        self.down_conv_2 = double_conv(112, 224)
        self.down_conv_3 = double_conv(224, 448)
        self.down_conv_4 = double_conv(448, 448)

        self.up_conv_1 = double_conv(896, 224)
        self.up_conv_2 = double_conv(448, 112)
        self.up_conv_3 = double_conv(224, 112)

        self.up_1 = nn.ConvTranspose2d(in_channels=448, out_channels=448, kernel_size=2, stride=2)
        self.up_2 = nn.ConvTranspose2d(in_channels=224, out_channels=224, kernel_size=2, stride=2)
        self.up_3 = nn.ConvTranspose2d(in_channels=112, out_channels=112, kernel_size=2, stride=2)

        self.out = nn.Conv2d(in_channels=112, out_channels=1, kernel_size=1, padding=(1, 1))
        self.out_2 = nn.Sigmoid()

    def forward(self, x):
        x1 = self.down_conv_1(x)
        x2 = self.max_pool_2x2(x1)

        x3 = self.down_conv_2(x2)
        x4 = self.max_pool_2x2(x3)

        x5 = self.down_conv_3(x4)
        x6 = self.max_pool_2x2(x5)

        x = self.up_1(x6)
        x = torch.cat((x, x5), dim=1)
        x = self.drop_out(x)
        x = self.up_conv_1(x)

        x = self.up_2(x)
        x = torch.cat((x, x), dim=1)
        x = self.drop_out(x)
        x = self.up_conv_2(x)

        x = self.up_3(x)
        x = torch.cat((x, x), dim=1)
        x = self.drop_out(x)
        x = self.up_conv_3(x)

        x = self.out(x)
        x = self.out_2(x)
        return x


if __name__ == '__main__':
    # Dataset
    dataset = CustomDataSet("X:/dataset/noised", "X:/dataset/clean", transform=tsfms)
    dataset_test = CustomDataSet("X:/dataset/test_noised", "X:/dataset/test_clean", transform=tsfms)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=False,
                              num_workers=4, drop_last=True)

    test_loader = DataLoader(dataset_test, batch_size=16, shuffle=False,
                             num_workers=4, drop_last=True)

    # We check whether cuda is available and choose device accordingly
    if torch.cuda.is_available() == True:
        device = "cuda:0"
    else:
        device = "cpu"

    model = UNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)

    epochs = 1
    l = len(train_loader)
    losslist = list()
    epochloss = 0
    running_loss = 0
    min_loss = 666666666666666

    TRAIN = True
    if TRAIN:
        for epoch in range(epochs):

            print("Entering Epoch: ", epoch)
            for dirty, clean in tqdm((train_loader)):
                # dirty = dirty.view(dirty.size(0), -1).type(torch.FloatTensor)
                # clean = clean.view(clean.size(0), -1).type(torch.FloatTensor)
                dirty, clean = dirty.to(device), clean.to(device)
                # dirty = dirty.unsqueeze(1)
                # dirty = dirty.unsqueeze(2)
                print(dirty.shape)
                # -----------------Forward Pass----------------------
                output = model(dirty)
                loss = criterion(output, clean)
                # -----------------Backward Pass---------------------
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                epochloss += loss.item()

            # -----------------Log-------------------------------
            losslist.append(running_loss / l)

            if loss.item() < min_loss:
                torch.save(model.state_dict(), "X:/dataset/model/model2.pt")
                min_loss = loss.item()

            running_loss = 0
            print("======> epoch: {}/{}, Loss:{}".format(epoch, epochs, loss.item()))

        # plt.plot(range(len(losslist)), losslist)

    # model = torch.load('X:/dataset/model/model2.pt')
    # model.eval()
    f, axes = plt.subplots(6, 3, figsize=(20, 20))
    axes[0, 0].set_title("Original Image")
    axes[0, 1].set_title("Dirty Image")
    axes[0, 2].set_title("Cleaned Image")

    test_imgs = np.random.randint(0, 1126, size=6)
    test_imgs = np.asarray([i for i in range(6)])
    for idx in range((6)):
        print("loop idx")
        dirty = dataset_test[test_imgs[idx]][0]
        clean = dataset_test[test_imgs[idx]][1]
        # dirty = dirty.view(dirty.size(0), -1).type(torch.FloatTensor)
        dirty = dirty.unsqueeze(1)
        dirty = dirty.to(device)
        print(dirty.shape)
        print("Test")
        output = model(dirty)
        print("Test2")
        output = output.view(1, 218, 178)
        output = output.permute(1, 2, 0).squeeze(2)
        output = output.detach().cpu().numpy()

        dirty = dirty.view(1, 218, 178)
        dirty = dirty.permute(1, 2, 0).squeeze(2)
        dirty = dirty.detach().cpu().numpy()

        clean = clean.permute(1, 2, 0).squeeze(2)
        clean = clean.detach().cpu().numpy()

        axes[idx, 0].imshow(clean, cmap="gray")
        axes[idx, 1].imshow(dirty, cmap="gray")
        axes[idx, 2].imshow(output, cmap="gray")
    plt.show()

    # torch.save(model.state_dict(), "X:/dataset/model/model.pt")
