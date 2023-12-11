"""
Main script of the SegRavir Model.
""" 
import torch
import torch.nn as nn
from torch.nn import functional as F
import segmentation_models_pytorch as smp


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


class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Convolutional layer for query, key, and value
        self.conv_q = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv_k = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv_v = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Batch normalization for query, key, and value
        self.batchnorm_q = nn.BatchNorm2d(out_channels)
        self.batchnorm_k = nn.BatchNorm2d(out_channels)
        self.batchnorm_v = nn.BatchNorm2d(out_channels)

        # Convolutional layer for output
        self.conv_out = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        # Batch normalization for output
        self.batchnorm_out = nn.BatchNorm2d(out_channels)

        # Dropout layer
        self.dropout = nn.Dropout(0.2)

        # Residual layer
        self.residual_layer = nn.Identity()

    def forward(self, feature):
        # feature: (Batch_Size, In_Channels, Height, Width)

        residue = feature

        # Calculate query, key, and value
        q = self.conv_q(feature)
        k = self.conv_k(feature)
        v = self.conv_v(feature)

        # Batch normalization
        q = self.batchnorm_q(q)
        k = self.batchnorm_k(k)
        v = self.batchnorm_v(v)

        # Self-attention mechanism
        attention_weights = F.softmax(torch.matmul(q, k.transpose(2, 3)) / torch.sqrt(torch.tensor(q.shape[-1]).float()), dim=-1)
        attention_output = torch.matmul(attention_weights, v)

        # Output convolution and batch normalization
        output = self.conv_out(attention_output)
        output = self.batchnorm_out(output)
        output = self.dropout(output)

        return output + self.residual_layer(residue)
    

class CrossAttentionBlock(nn.Module):
    def __init__(self, in_channels_query, in_channels_key, out_channels, final=False):
        super(CrossAttentionBlock, self).__init__()

        # Query, Key, and Value transformations for cross-attention
        self.conv_query = nn.Conv2d(in_channels_query, out_channels, kernel_size=1)
        self.conv_key = nn.Conv2d(in_channels_key, out_channels, kernel_size=1)
        self.conv_value = nn.Conv2d(in_channels_key, out_channels, kernel_size=1)

        # Batch normalization for Query, Key, and Value
        self.batchnorm_query = nn.BatchNorm2d(out_channels)
        self.batchnorm_key = nn.BatchNorm2d(out_channels)
        self.batchnorm_value = nn.BatchNorm2d(out_channels)

        # Output convolution and batch normalization
        if final:
            self.conv_out = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        else:
            self.conv_out = nn.ConvTranspose2d(out_channels, out_channels, stride=2, kernel_size=3, padding=1, output_padding=1)
        self.batchnorm_out = nn.BatchNorm2d(out_channels)

        # Dropout layer
        self.dropout = nn.Dropout(0.2)

        # Residual layer
        if final:  
            self.residual_layer = nn.Conv2d(in_channels_query, out_channels, kernel_size=1)
        else:
            self.residual_layer = nn.ConvTranspose2d(in_channels_query, out_channels, stride=2, kernel_size=3, padding=1, output_padding=1)

    def forward(self, query, key):
        # query: (Batch_Size, In_Channels_Query, Height, Width)
        # key: (Batch_Size, In_Channels_Key, Height, Width)

        # Transformations for cross-attention
        q = self.conv_query(query)
        k = self.conv_key(key)
        v = self.conv_value(key)

        # Batch normalization
        q = self.batchnorm_query(q)
        k = self.batchnorm_key(k)
        v = self.batchnorm_value(v)

        # Cross-attention mechanism
        attention_weights = F.softmax(torch.matmul(q, k.transpose(2, 3)) / torch.sqrt(torch.tensor(q.shape[-1]).float()), dim=-1)
        attention_output = torch.matmul(attention_weights, v)

        # Output convolution and batch normalization
        output = self.conv_out(attention_output)
        output = self.batchnorm_out(output)
        output = self.dropout(output)

        return output + self.residual_layer(query)


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

        self.decoder_mask = nn.ModuleList([
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
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            # (Batch_Size, 16, Height, Width) --> (Batch_Size, 3, Height, Width)
            nn.Conv2d(16, 3, kernel_size=1),
            nn.Softmax(dim=1)
        ])

        self.decoder_reconstruction = nn.ModuleList([
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

            # (Batch_Size, 16, Height, Width) --> (Batch_Size, 3, Height, Width)
            nn.Conv2d(32, 3, kernel_size=1),
            nn.ReLU()
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

        # Copy of the output of the bottleneck
        mask = x
        skip_connections_2 = skip_connections.copy()

        # Main decoder stream 
        skip_decoder = [2, 4, 6, 8]
        i = 0
        for layers in self.decoder_mask:
            if i in skip_decoder:
                mask = torch.cat((mask, skip_connections.pop()), dim=1)
            mask = layers(mask)
            i += 1
        
        # Auxiliary decoder stream
        skip_decoder = [2, 4, 6, 8]
        i = 0
        for layers in self.decoder_reconstruction:
            if i in skip_decoder:
                x = torch.cat((x, skip_connections_2.pop()), dim=1)
            x = layers(x)
            i += 1

        return mask, x


class SegRAVIRAttentionModel(nn.Module):

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

        self.decoder_mask = nn.ModuleList([
            # (Batch_Size, 256, Height / 16, Width / 16) --> (Batch_Size, 128, Height / 8, Width / 8)
            nn.ConvTranspose2d(256, 128, stride=2, kernel_size=3, padding=1, output_padding=1),
            ResidualBlock(128, 128),

            # (Batch_Size, 128, Height / 8, Width / 8) --> (Batch_Size, 64, Height / 4, Width / 4)
            CrossAttentionBlock(128, 128, 64),
            ResidualBlock(64, 64),

            # (Batch_Size, 64, Height / 4, Width / 4) --> (Batch_Size, 32, Height / 2, Width / 2)
            CrossAttentionBlock(64, 64, 32),
            ResidualBlock(32, 32),

            # (Batch_Size, 32, Height / 2, Width / 2) --> (Batch_Size, 16, Height, Width)
            CrossAttentionBlock(32, 32, 16),
            ResidualBlock(16, 16),

            # (Batch_Size, 16, Height, Width) --> (Batch_Size, 16, Height, Width)
            CrossAttentionBlock(16, 16, 16, final=True),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            # (Batch_Size, 16, Height, Width) --> (Batch_Size, 3, Height, Width)
            nn.Conv2d(16, 3, kernel_size=1),
            nn.Softmax(dim=1)
        ])

        self.decoder_reconstruction = nn.ModuleList([
            # (Batch_Size, 256, Height / 16, Width / 16) --> (Batch_Size, 128, Height / 8, Width / 8)
            nn.ConvTranspose2d(256, 128, stride=2, kernel_size=3, padding=1, output_padding=1),
            ResidualBlock(128, 128),

            # (Batch_Size, 128, Height / 8, Width / 8) --> (Batch_Size, 64, Height / 4, Width / 4)
            CrossAttentionBlock(128, 128, 64),
            ResidualBlock(64, 64),

            # (Batch_Size, 64, Height / 4, Width / 4) --> (Batch_Size, 32, Height / 2, Width / 2)
            CrossAttentionBlock(64, 64, 32),
            ResidualBlock(32, 32),

            # (Batch_Size, 32, Height / 2, Width / 2) --> (Batch_Size, 16, Height, Width)
            CrossAttentionBlock(32, 32, 16),
            ResidualBlock(16, 16),

            # (Batch_Size, 16, Height, Width) --> (Batch_Size, 3, Height, Width)
            CrossAttentionBlock(16, 16, 16, final=True),
            nn.Conv2d(16, 3, kernel_size=1),
            nn.ReLU()
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

        # Copy of the output of the bottleneck
        mask = x
        skip_connections_2 = skip_connections.copy()

        # Main decoder stream 
        skip_decoder = [2, 4, 6, 8]
        i = 0
        for layers in self.decoder_mask:
            if i in skip_decoder:
                mask = layers(mask, skip_connections.pop())
            else:
                mask = layers(mask)
            i += 1
        
        # Auxiliary decoder stream
        skip_decoder = [2, 4, 6, 8]
        i = 0
        for layers in self.decoder_reconstruction:
            if i in skip_decoder:
                x = layers(x, skip_connections_2.pop())
            else:
                x = layers(x)
            i += 1

        return mask, x


class Loss_Dice_CE_l2(nn.Module):
    def __init__(self, weight_dice=0.5, weight_ce=0.3, weight_l2=0.2):
        super(Loss_Dice_CE_l2, self).__init__()

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_l2 = weight_l2

        self.dice_loss = smp.losses.DiceLoss(
            mode='multiclass',
            classes=[0, 1, 2]
        )

    def forward(self, logits_mask, pred_img, targets_mask, targets_img):
        # Dice coefficient loss
        dice_loss = self.dice_loss(logits_mask, targets_mask)

        # Cross-entropy loss
        ce_loss = F.cross_entropy(logits_mask, targets_mask)

        # L2 reconstruction loss
        l2_loss = F.mse_loss(pred_img, targets_img)

        # Combine the two losses using the specified weight
        combined_loss = self.weight_dice * dice_loss + self.weight_ce * ce_loss + self.weight_l2 * l2_loss

        return combined_loss