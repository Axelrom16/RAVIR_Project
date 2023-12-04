"""
Main script of the SegRavir Model.
""" 
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchview import draw_graph
from torchviz import make_dot
import matplotlib.pyplot as plt


class ResidualBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int): 
        super().__init__()

        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.batchnorm_1 = nn.BatchNorm2d(out_channels)
        self.batchnorm_2 = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout(0.2)

        self.residual_layer = nn.Identity()
    
    def forward(self, feature):
        # feature: (Batch_Size, In_Channels, Height, Width) 

        residue = feature

        feature = self.conv_1(feature)
        feature = F.relu(feature)
        feature = self.batchnorm_1(feature)
        feature = self.dropout(feature)

        feature = self.conv_2(feature)
        feature = F.relu(feature)
        feature = self.batchnorm_2(feature)
        feature = self.dropout(feature)

        return feature + self.residual_layer(residue)


class SegRAVIRModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.ModuleList([
            # (Batch_Size, 3, Height, Width) --> (Batch_Size, 16, Height / 2, Width / 2)
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            ResidualBlock(16, 16),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),

            # (Batch_Size, 16, Height / 2, Width / 2) --> (Batch_Size, 32, Height / 4, Width / 4)
            ResidualBlock(32, 32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),

            # (Batch_Size, 32, Height / 4, Width / 4) --> (Batch_Size, 64, Height / 8, Width / 8)
            ResidualBlock(64, 64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),

            # (Batch_Size, 64, Height / 8, Width / 8) --> (Batch_Size, 128, Height / 16, Width / 16)
            ResidualBlock(128, 128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        ])

        self.bottleneck = nn.ModuleList([
            # (Batch_Size, 128, Height / 16, Width / 16) --> (Batch_Size, 256, Height / 16, Width / 16)
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256)
        ]) 

        self.decoder = nn.ModuleList([
            # (Batch_Size, 256, Height / 16, Width / 16) --> (Batch_Size, 128, Height / 8, Width / 8)
            nn.ConvTranspose2d(256, 128, stride=2, kernel_size=3, padding=1, output_padding=1),
            ResidualBlock(128, 128),

            # (Batch_Size, 128, Height / 8, Width / 8) --> (Batch_Size, 64, Height / 4, Width / 4)
            nn.ConvTranspose2d(256, 64, stride=2, kernel_size=3, padding=1, output_padding=1),
            ResidualBlock(64, 64),

            # (Batch_Size, 64, Height / 4, Width / 4) --> (Batch_Size, 32, Height / 2, Width / 2)
            nn.ConvTranspose2d(128, 32, stride=2, kernel_size=3, padding=1, output_padding=1),
            ResidualBlock(32, 32),

            # (Batch_Size, 32, Height / 2, Width / 2) --> (Batch_Size, 16, Height, Width)
            nn.ConvTranspose2d(64, 16, stride=2, kernel_size=3, padding=1, output_padding=1),
            ResidualBlock(16, 16),

            # (Batch_Size, 16, Height, Width) --> (Batch_Size, 16, Height, Width)
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),

            # (Batch_Size, 16, Height, Width) --> (Batch_Size, 3, Height, Width)
            nn.Conv2d(16, 3, kernel_size=1)
        ])

    def forward(self, x):
        # x: (Batch_Size, 3, Height, Width)

        skip_encoder = [1, 3, 5, 7]
        skip_connections = []
        i = 0
        for layers in self.encoder:
            x = layers(x)
            if i in skip_encoder:
                skip_connections.append(x)
            i += 1
        
        for layers in self.bottleneck:
            x = layers(x)

        skip_decoder = [2, 4, 6, 8]
        i = 0
        j = 0
        for layers in self.decoder:
            if i in skip_decoder:
                x = torch.cat((x, skip_connections.pop()), dim=1)
                j += 1
            x = layers(x)
            i += 1
        

"""
if __name__ == '__main__':
    model = SegRAVIRModel()
    print(model)

    #model_graph = draw_graph(model, input_size=(1,3,256,256), expand_nested=True)
    #model_graph.visual_graph

    dataset_train = RAVIRDataset(
        data_root='/media/axelrom16/HDD/AI/RAVIR_Project/data/train/training_images',
        segmentation_root='/media/axelrom16/HDD/AI/RAVIR_Project/data/train/training_masks',
        size=256,
        random_crop=False,
        interpolation="bicubic",
        n_labels=3,
        shift_segmentation=False,
        augmentation=True
    )
    train_dataloader = DataLoader(dataset_train, batch_size=8, shuffle=True)
    batch = next(iter(train_dataloader))
    yhat = model(batch["image"].permute(0, 3, 1, 2)) 
    make_dot(yhat).render("rnn_torchviz", format="png")
    # x = torch.randn((1, 3, 256, 256))
    # y = model(x)
    # print(y.shape)

    # dot = make_dot(y, params=dict(model.named_parameters()))
    # dot.render('model', format='png')
"""