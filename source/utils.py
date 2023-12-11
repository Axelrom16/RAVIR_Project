"""
Script with useful functions. 
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics import Dice, JaccardIndex


def plot_figure_mask(image, mask, figsize=(10, 10)):
    """
    Plot image and mask side by side. 
    """
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].imshow(image)
    ax[0].set_title('Image')
    ax[0].axis('off')
    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title('Mask')
    ax[1].axis('off')
    plt.show() 


def plot_figure_mask_prediction(image, mask, prediction, figsize=(10, 10)):
    """
    Plot image, mask and prediction side by side. 
    """
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    ax[0].imshow(image)
    ax[0].set_title('Image')
    ax[0].axis('off')
    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title('Mask')
    ax[1].axis('off')
    ax[2].imshow(prediction, cmap='gray')
    ax[2].set_title('Predicted Mask')
    ax[2].axis('off')
    plt.show()


def evaluation_jaccard_dice(prediction, ground_truth, n_classes=3):
    """
    Compute Jaccard and Dice scores for each class. 
    """
    jaccard = JaccardIndex(task="multiclass", num_classes=n_classes, average='macro')
    dice = Dice(num_classes=n_classes, average='macro')
    
    jaccard_score = jaccard(prediction, ground_truth)
    dice_score = dice(prediction, ground_truth)

    return jaccard_score, dice_score


def evaluate_model(model, dataloader, device, n_classes=3):
    """
    Evaluate model on a dataloader. 
    """
    model.eval()

    jaccard_scores = []
    dice_scores = []
    with torch.no_grad():
        for batch in dataloader:
            image = batch['image'].to(device)
            mask = batch['mask'].to(device)

            prediction = model(image)
            prediction = torch.argmax(prediction, dim=1)

            prediction = prediction.cpu().numpy()
            mask = mask.cpu().numpy()

            jaccard_score, dice_score = evaluation_jaccard_dice(prediction, mask, n_classes=n_classes)

            jaccard_scores.append(jaccard_score)
            dice_scores.append(dice_score)
            
    return np.mean(jaccard_scores), np.mean(dice_scores)