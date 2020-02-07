import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

from FaceLandmarksDataset import FaceLandmarksDataset
from RandomCrop import RandomCrop
from Rescale import Rescale
from ToTensor import ToTensor
from helpers import show_landmarks_batch

face_dataset = FaceLandmarksDataset(
    csv_file='data/faces/face_landmarks.csv',
    root_dir='data/faces',
    transform=transforms.Compose([
        Rescale(256),
        RandomCrop(224),
        ToTensor()
    ])
)

dataloader = DataLoader(face_dataset, batch_size=4, shuffle=True)

print(enumerate(dataloader))
for i_batch, sample_batched in enumerate(dataloader, 0):
    print(i_batch, sample_batched['image'].size(), sample_batched['landmarks'].size())

    if i_batch == 3:
        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break
