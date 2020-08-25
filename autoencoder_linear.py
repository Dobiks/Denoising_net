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


class denoising_model(nn.Module):
    def __init__(self):
        super(denoising_model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(218*178, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 218*178),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


if __name__ == '__main__':
    #Dataset
    dataset = CustomDataSet("X:/dataset/noised", "X:/dataset/clean", transform=tsfms)
    dataset_test = CustomDataSet("X:/dataset/test_noised", "X:/dataset/test_clean", transform=tsfms)
    train_loader = DataLoader(dataset, batch_size=256, shuffle=False,
                              num_workers=4, drop_last=True)

    test_loader = DataLoader(dataset_test, batch_size=64, shuffle=False,
                              num_workers=4, drop_last=True)

    model = denoising_model()

    # We check whether cuda is available and choose device accordingly
    if torch.cuda.is_available() == True:
        device = "cuda:0"
    else:
        device = "cpu"

    model = denoising_model().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)

    epochs = 200
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
                dirty = dirty.view(dirty.size(0), -1).type(torch.FloatTensor)
                clean = clean.view(clean.size(0), -1).type(torch.FloatTensor)
                dirty, clean = dirty.to(device), clean.to(device)

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
                torch.save(model.state_dict(), "X:/dataset/model/model.pt")
                min_loss = loss.item()

            running_loss = 0
            print("======> epoch: {}/{}, Loss:{}".format(epoch, epochs, loss.item()))

        plt.plot(range(len(losslist)), losslist)

    #model = torch.load('X:/dataset/model/model.pt')
    model.eval()

    f, axes = plt.subplots(6, 3, figsize=(20, 20))
    axes[0, 0].set_title("Original Image")
    axes[0, 1].set_title("Dirty Image")
    axes[0, 2].set_title("Cleaned Image")

    # test_imgs = np.random.randint(0, 1126, size=6)
    test_imgs = np.asarray([i for i in range(6)])
    for idx in range((6)):
        dirty = dataset_test[test_imgs[idx]][0]
        clean = dataset_test[test_imgs[idx]][1]

        dirty = dirty.view(dirty.size(0), -1).type(torch.FloatTensor)
        dirty = dirty.to(device)
        output = model(dirty)

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
