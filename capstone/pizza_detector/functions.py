# standard imports
import os, glob, fnmatch, pickle, itertools
import pandas as pd
import numpy as np

# image processing imports
import cv2

# classification metrics imports
from sklearn.metrics import roc_curve, roc_auc_score, auc, classification_report, confusion_matrix, accuracy_score, precision_score, recall_score

# plotting imports
import seaborn as sns
import matplotlib
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib import cm as cm

def img_plots(fig_h, path_list, plot_title, label_df=None, x_label=None, variable_df=None):
    sns.set_style("white")
    fig, ax = plt.subplots(1,5,figsize=(16,fig_h))
    
    images_plot = []
    
    for img in path_list[:5]:
        image = cv2.imread(img)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images_plot.append(image_rgb)
    
    for i in range(5):
        plt.subplot(1,5,i+1)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.imshow(images_plot[i])
        if not label_df is None:
            label = 'not pizza' if label_df.values[i] == 0 else 'pizza'
            plt.title(label, size=16)
        if not (x_label is None) and not (variable_df is None):
            plt.xlabel(x_label + "\n%.3f" % variable_df.values[i], size=14)
        plt.suptitle(plot_title, size=18)
    plt.show()  


def transformed_plots(fig_h, path_list, plot_title):
    sns.set_style("white")
    fig, ax = plt.subplots(1,5,figsize=(16,fig_h))
    
    images_plot = []
    
    for img in path_list[:5]:
        image = cv2.imread(img)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images_plot.append(image_rgb)
    
    for i in range(5):
        plt.subplot(1,5,i+1)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.imshow(images_plot[i])
        plt.suptitle(plot_title, size=18)
    plt.show()  
    


def epoch_plot(acc, val_acc, loss, val_loss):
    # A plot of accuracy on the training and validation datasets over training epochs.
    sns.set_style("dark")
    plt.figure(figsize=(20, 6))

    plt.subplot(1,2,1)
    plt.plot(acc)
    plt.plot(val_acc)
    plt.title('Model accuracy', size=18)
    plt.ylabel('Accuracy', size=16)
    plt.xlabel('Epoch', size=16)
    plt.legend(['Train', 'Test'], loc='upper left', fontsize = 14)
    plt.tick_params(axis='both', which='major', labelsize=13)

    # A plot of loss on the training and validation datasets over training epochs.
    plt.subplot(1,2,2)
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Model loss', size=18)
    plt.ylabel('Loss', size=16)
    plt.xlabel('Epoch', size=16)
    plt.legend(['Train', 'Test'], loc='upper left', fontsize = 14)
    plt.tick_params(axis='both', which='major', labelsize=13)

    plt.show()

def corr(df, title):
    sns.set_style("dark")
    fig, ax = plt.subplots(figsize=(12,7))
    mask = np.zeros_like(df.corr(), dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    ax = sns.heatmap(df.corr(), mask=mask, annot=True)
    ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=14, rotation=70)
    ax.set_yticklabels(ax.yaxis.get_ticklabels(), fontsize=14, rotation=0)
    plt.suptitle(title, size=18)
    plt.show()


def color_space_plots(space1, color1, space1_label, space2, color2, space2_label, space3, color3, space3_label, title):
    sns.set_style("dark")
    fig, ax = plt.subplots(1,3,figsize=(18,4))
    sns.despine()

    sns.distplot(space1, bins=100, kde=False, hist_kws={"alpha": 1, "color": color1}, ax=ax[0])
    ax[0].set_xlabel(space1_label, size = 16, )

    sns.distplot(space2, bins=100, kde=False, hist_kws={"alpha": 1, "color": color2}, ax=ax[1])
    ax[1].set_xlabel(space2_label, size = 16)

    sns.distplot(space3, bins=100, kde=False, hist_kws={"alpha": 1, "color": color3}, ax=ax[2])
    ax[2].set_xlabel(space3_label, size = 16)

    plt.suptitle(title, size=18)
    plt.show()



