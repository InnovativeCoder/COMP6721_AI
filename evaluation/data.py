from torch.utils.data import DataLoader
from torchvision import transforms, datasets

data_transform = transforms.Compose([
    transforms.Resize(125),  # smaller edge of image will be matched to 125 i.e, if height > width, then image will be rescaled to (size * height / width, size)
    transforms.CenterCrop(100),  # a square crop (100, 100) is made
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

mask_detection_train_set = datasets.ImageFolder(
    root="../final_dataset/train",
    transform=data_transform
)
train_set_loader = DataLoader(mask_detection_train_set, batch_size=4, shuffle=True, num_workers=4)
