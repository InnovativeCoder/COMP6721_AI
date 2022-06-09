# set the numpy seed for better reproducibility
import numpy as np
from PIL import Image

np.random.seed(42)
# import the necessary packages

from torchvision import transforms
import argparse
import torch

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--variant", type=str, required=True,
                help="variant of the trained PyTorch model")
ap.add_argument("-i", "--image", type=str, required=True,
                help="image to be tested")
args = vars(ap.parse_args())

# set the device we will be using to test the model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

data_transform = transforms.Compose([
    transforms.Resize(125),
    # smaller edge of image will be matched to 125 i.e, if height > width, then image will be rescaled to (size * height / width, size)
    transforms.CenterCrop(100),  # a square crop (100, 100) is made
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

print("[INFO] loading and preprocessing the given image...")
# testData = KMNIST(root="data", train=False, download=True,
#                   transform=ToTensor())
# idxs = np.random.choice(range(0, len(testData)), size=(10,))
# testData = Subset(testData, idxs)
# initialize the test data loader
# testDataLoader = DataLoader(testData, batch_size=1)
# load the model and set it to evaluation mode

image = Image.open(f"{args['image']}")
model = torch.load(f'models/model{args["variant"]}.pth').to(device)
model.eval()

classes = [
    '01_without_mask',
    '02_with_cloth_mask',
    '03_with_surgical_mask',
    '04_with_n95_mask',
    '05_incorrectly_worn_mask'
]

# switch off autograd
with torch.no_grad():
    # loop over the test set

    # grab the original image and ground truth label
    img_data = data_transform(image)
    img = img_data.unsqueeze(0)
    # send the input to the device and make predictions on it
    img = img.to(device)
    pred = model(img)
    # find the class label index with the largest corresponding
    # probability
    idx = pred.argmax(axis=1).cpu().numpy()[0]
    predLabel = classes[idx]

    print("[INFO] predicted label: {}".format(predLabel))
