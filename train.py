import matplotlib

matplotlib.use("Agg")

from image_classifier import cnn1, cnn2, cnn3
from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
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

# define training hyperparameters
INIT_LR = 1e-3
BATCH_SIZE = 25
EPOCHS = 10
# define the train and val splits
TRAIN_SPLIT = 0.9
# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose([
    transforms.Resize(125),
    # smaller edge of image will be matched to 125 i.e, if height > width, then image will be rescaled to (size * height / width, size)
    transforms.CenterCrop(100),  # a square crop (100, 100) is made
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

print("[INFO] loading the face mask detection dataset...")
trainData = datasets.ImageFolder(
    root="./dataset/train",
    transform=data_transform
)
testData = datasets.ImageFolder(
    root="./dataset/test",
    transform=data_transform
)

# calculate the train/validation split
print("[INFO] generating the train/validation split...")
numTrainSamples = int(len(trainData) * TRAIN_SPLIT)
numValSamples = len(trainData) - numTrainSamples
(trainData, valData) = random_split(trainData,
                                    [numTrainSamples, numValSamples],
                                    generator=torch.Generator().manual_seed(42))

# initialize the train, validation, and test data loaders
trainDataLoader = DataLoader(trainData, shuffle=True,
                             batch_size=BATCH_SIZE)
valDataLoader = DataLoader(valData, batch_size=BATCH_SIZE)
testDataLoader = DataLoader(testData, batch_size=BATCH_SIZE)
# calculate steps per epoch for training and validation set
trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
valSteps = len(valDataLoader.dataset) // BATCH_SIZE

# initialize the CNN
print(f"[INFO] initializing the CNN variant {args['variant']}...")

variants = {1: cnn1, 2: cnn2, 3: cnn3}
model = variants[args["variant"]].CNN(numChannels=3, classes=len(trainData.dataset.classes)).to(device)

# initialize our optimizer and loss function
# opt = SGD(model.parameters(), lr=INIT_LR)
# lossFn = nn.CrossEntropyLoss()
opt = Adam(model.parameters(), lr=INIT_LR)
lossFn = nn.NLLLoss()

# initialize a dictionary to store training history
H = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}
# measure how long training is going to take
print("[INFO] training the network...")
startTime = time.time()

# loop over our epochs
for e in range(0, EPOCHS):
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

    # switch off autograd for evaluation
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

# finish measuring how long training took
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
    endTime - startTime))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["train_acc"], label="train_acc")
plt.title("Training Loss/Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(f'./metrics/training{args["variant"]}.png')

# serialize the model to disk
torch.save(model, f'./models/model{args["variant"]}.pth')

# https://christianbernecker.medium.com/how-to-create-a-confusion-matrix-in-pytorch-38d06a7f04b7 [13]
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

# we can now evaluate the network on the test set
print("[INFO] evaluating network...")
# turn off autograd for testing evaluation
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
    classes = trainData.dataset.classes

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * len(classes), index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(f'./metrics/cf_matrix_{args["variant"]}.png')

    print(classification_report(y_true,
                                y_pred, target_names=testData.classes))
    with open(f"./metrics/eval_metrics{args['variant']}.txt", "w") as f:
        f.write(classification_report(y_true,
                                      y_pred, target_names=testData.classes))
