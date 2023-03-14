import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch

class DatasetBuilder:
    def __init__(self, directory, img_size, batch_size):
        # Define the transformation to be applied to the images
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        # Create the dataset
        dataset = datasets.ImageFolder(root=directory, transform=transform)

        # Create a data loader
        self.loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        pass

    def get_dataset(self):
        return self.loader