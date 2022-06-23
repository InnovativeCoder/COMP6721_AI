import matplotlib
import pandas as pd

matplotlib.use("Agg")

from image_classifier import cnn1, cnn2, cnn3
from sklearn.metrics import classification_report, confusion_matrix
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
import seaborn as sn

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--variant", type=int, required=True,
                help="the model to train and save")
args = vars(ap.parse_args())
variants = {1: cnn1, 2: cnn2, 3: cnn3}

# define training hyperparameters
INIT_LR = 1e-3
BATCH_SIZE = 25
EPOCHS = 20
FOLDS = 10

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
final_model = None
validation_accuracies = []
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

    """ Training the CNN """
    print(f"Starting training for fold: {fold+1}")
    # set the model in training mode
    model.train()
    # loop over our epochs
    for e in range(0, EPOCHS):
        # loop over the training set
        loss_list = []
        acc_list = []
        correct, total = 0, 0
        for i, (images, labels) in enumerate(trainDataLoader):
            # send the input to the device
            (images, labels) = (images.to(device), labels.to(device))
            # perform a forward pass and calculate the training loss
            outputs = model(images)
            loss = lossFn(outputs, labels)
            loss_list.append(float(loss))
            # zero out the gradients, perform the backpropagation step,
            # and update the weights
            opt.zero_grad()
            loss.backward()
            opt.step()
            # Train accuracy
            total += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            acc_list.append(correct / total)

        print(f'Epoch [{e+1}/{EPOCHS}], Training Loss: {np.average(loss_list):.4f}, Training Accuracy: {(correct / total) * 100:.2f}%')

    # evaluate the model per each fold for the validation dataset
    # set the model in evaluation mode
    model.eval()
    with torch.no_grad():
        correct_labels = []
        predicted_labels = []
        correct = 0
        total = 0
        # loop over the validation set
        for (x, y) in valDataLoader:
            # Save the labels before copying to device
            correct_labels.extend(list(y.numpy()))
            # send the input to the device
            (x, y) = (x.to(device), y.to(device))
            # make the predictions and calculate the validation loss
            pred = model(x)
            _, predicted = torch.max(pred.data, 1)
            # calculate the number of correct predictions
            total += y.size(0)
            correct += (predicted == y).sum().item()
            predicted_labels.extend(list(predicted.cpu().numpy()))

    validation_accuracy = (correct/total) * 100
    print(f"Validation accuracy for the fold {fold + 1}: {validation_accuracy}")

    # Save this validation accuracy for aggregate accuracy across folds
    validation_accuracies.append(validation_accuracy)

    classes = dataset.classes

    # Build confusion matrix
    cf_matrix = confusion_matrix(correct_labels, predicted_labels)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * len(classes), index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(f'./folds_metrics/cf_matrix_{fold + 1}.png')
    print(f"Saved confusion matrix for {fold + 1}")

    with open(f"./folds_metrics/eval_metrics_{fold + 1}.txt", "w") as f:
        f.write(classification_report(correct_labels,
                                      predicted_labels, target_names=classes))

    print(f"Saved evaluation metrics for {fold + 1}")

    # Keep track of the best cnn model
    if validation_accuracy >= np.max(validation_accuracies):
        final_model = model

# serialize the model to disk
torch.save(final_model, f'./models/final_model.pth')

print(f"Average accuracy : {np.average(validation_accuracies)}")

print("TESTING:")
# On the test set
with torch.no_grad():
    # set the model in evaluation mode
    final_model.eval()

    # initialize a list to store our predictions
    y_pred = []
    y_true = []

    for inputs, labels in testDataLoader:
        inputs = inputs.to(device)
        outputs = final_model(inputs)  # Feed Network
        outputs = outputs.argmax(axis=1).cpu().numpy()
        y_pred.extend(outputs)  # Save Prediction

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)  # Save Truth

    # constant for classes
    classes = dataset.classes

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * len(classes), index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(f'./metrics/cf_matrix_final_model.png')
    print(f"Saved test confusion matrix for the final model")
    print(classification_report(y_true,
                                y_pred, target_names=testData.classes))
    with open(f"./metrics/eval_metrics_final_model.txt", "w") as f:
        f.write(classification_report(y_true,
                                      y_pred, target_names=testData.classes))
    print(f"Saved test evaluation metrics for the final model")

#csv.reader("C:\Users\Intel\Downloads\Test_Dataset_Categories", "Excel")

#indeces as class
testMales = [34,35,43,50,69]
testFemales = [66,65,57,50,31]
testYoung = [31,33,24,27,38]
testMiddle = [49,41,57,53,56]
testOld = [20,26,19,20,16]
#each index here should sum to 100

ageList = []
genderList = []

with open('dataset\\test\\Test_Dataset_Categories_List.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, 'excel')
    for row in reader:
        if row[0] == '' or row[1] == "Age":
            continue
        ageList.append(row[1])
        genderList.append(row[2])

malesByClass = [0,0,0,0,0]
femalesByClass = [0,0,0,0,0]
youngByClass = [0,0,0,0,0]
middleByClass = [0,0,0,0,0]
oldByClass = [0,0,0,0,0]
for i in range(len(testData.imgs)):
    if y_pred[i] == y_true[i]:
        outputClass = y_true[i]

        gender = genderList[i]
        age = ageList[i]

        if gender == "Male":
            malesByClass[outputClass] += 1
        else:
            femalesByClass[outputClass] += 1

        if age == "Young":
            youngByClass[outputClass] += 1
        elif age == "Middle":
            middleByClass[outputClass] += 1
        else:
            oldByClass[outputClass] += 1

print("Accuracy per class for Gender and Age biases:")
print("Class - Male - Female - Young - Middle - Old")
for i in range(5):
    print(f"  {i}  -  "
          f"{'{0:.0f}'.format(malesByClass[i]/testMales[i]*100)}%  -  "
          f"{'{0:.0f}'.format(femalesByClass[i]/testFemales[i]*100)}%  -  "
          f"{'{0:.0f}'.format(youngByClass[i]/testYoung[i]*100)}%  -  "
          f"{'{0:.0f}'.format(middleByClass[i]/testMiddle[i]*100)}%  -  "
          f"{'{0:.0f}'.format(oldByClass[i]/testOld[i]*100)}%")