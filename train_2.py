import matplotlib

matplotlib.use("Agg")

from image_classifier import cnn1, cnn2, cnn3
from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.model_selection import KFold
from torch.optim import Adam, SGD
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--variant", type=int, required=True,
                help="the model to train and save")
args = vars(ap.parse_args())
variants = {1: cnn1, 2: cnn2, 3: cnn3}

# define training hyperparameters
INIT_LR = 1e-3
BATCH_SIZE = 25
EPOCHS = 10
FOLDS = 5

# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose([
    transforms.Resize(125),
    transforms.CenterCrop(100),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# For fold results
results = {}

# Just the train data
dataset = datasets.ImageFolder(
    root="./dataset/train",
    transform=data_transform
)
testData = datasets.ImageFolder(
    root="./dataset/test",
    transform=data_transform
)
testDataLoader = DataLoader(testData, batch_size=BATCH_SIZE)

# Define the K-Fold Cross Validator
kfold = KFold(n_splits=FOLDS, shuffle=True)

# K-fold Cross Validation model evaluation
for fold, (train_ids, validation_ids) in enumerate(kfold.split(dataset)):
    print("-------------------------------")
    print(f'FOLD {fold+1}/{FOLDS}')

    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    validation_subsampler = torch.utils.data.SubsetRandomSampler(validation_ids)

    # Define data loaders for training and validation data in this fold
    trainDataLoader = torch.utils.data.DataLoader(
                      dataset,
                      batch_size=BATCH_SIZE, sampler=train_subsampler)
    valDataLoader = torch.utils.data.DataLoader(
                      dataset,
                      batch_size=BATCH_SIZE, sampler=validation_subsampler)
    # calculate steps per epoch for training and validation set
    trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
    valSteps = len(valDataLoader.dataset) // BATCH_SIZE

    # Initialize the neural network
    model = variants[args["variant"]].CNN(numChannels=3, classes=len(dataset.classes)).to(device)
    # initialize our optimizer and loss function
    opt = Adam(model.parameters(), lr=INIT_LR)
    lossFn = nn.NLLLoss()

    # initialize a dictionary to store training history
    H = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    # loop over our epochs
    for e in range(0, EPOCHS):
        # Print epoch
        print(f'Epoch {e + 1}/{EPOCHS}')

        # set the model in training mode
        model.train()

        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0
        # initialize the number of correct predictions in the training
        # and validation step
        trainCorrect = 0
        valCorrect = 0

        # loop over the training set
        for (x, y) in trainDataLoader:
            # send the input to the device
            (x, y) = (x.to(device), y.to(device))
            # perform a forward pass and calculate the training loss
            pred = model(x)
            loss = lossFn(pred, y)
            # zero out the gradients, perform the backpropagation step,
            # and update the weights
            opt.zero_grad()
            loss.backward()
            opt.step()
            # add the loss to the total training loss so far and
            # calculate the number of correct predictions
            totalTrainLoss += loss
            trainCorrect += (pred.argmax(1) == y).type(
                torch.float).sum().item()

        # evaluate the model per each epoch
        with torch.no_grad():
            # set the model in evaluation mode
            model.eval()
            # loop over the validation set
            for (x, y) in valDataLoader:
                # send the input to the device
                (x, y) = (x.to(device), y.to(device))
                # make the predictions and calculate the validation loss
                pred = model(x)
                totalValLoss += lossFn(pred, y)
                # calculate the number of correct predictions
                valCorrect += (pred.argmax(1) == y).type(
                    torch.float).sum().item()

        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps
        # calculate the training and validation accuracy
        trainCorrect = trainCorrect / len(trainDataLoader.dataset)
        valCorrect = valCorrect / len(valDataLoader.dataset)
        # update our training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["train_acc"].append(trainCorrect)
        H["val_loss"].append(avgValLoss.cpu().detach().numpy())
        H["val_acc"].append(valCorrect)
        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
        print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
            avgTrainLoss, trainCorrect))
        print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
            avgValLoss, valCorrect))

    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()

        # initialize a list to store our predictions
        y_pred = []
        y_true = []

        for inputs, labels in testDataLoader:
            inputs = inputs.to(device)
            outputs = model(inputs)  # Feed Network
            outputs = outputs.argmax(axis=1).cpu().numpy()
            y_pred.extend(outputs)  # Save Prediction

            labels = labels.data.cpu().numpy()
            y_true.extend(labels)  # Save Truth

        # constant for classes
        classes = dataset.classes

        # # Build confusion matrix
        # cf_matrix = confusion_matrix(y_true, y_pred)
        # df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * len(classes), index=[i for i in classes],
        #                      columns=[i for i in classes])
        # plt.figure(figsize=(12, 7))
        # sn.heatmap(df_cm, annot=True)
        # plt.savefig(f'./metrics/cf_matrix_{args["variant"]}.png')

        print(f"FOLD {fold+1} REPORT: ")
        print(classification_report(y_true,
                                    y_pred, target_names=testData.classes))
        # with open(f"./metrics/eval_metrics{args['variant']}.txt", "w") as f:
        #     f.write(classification_report(y_true,
        #                                   y_pred, target_names=testData.classes))
