import torch
from AutoEncoder import AutoEncoder
from DatasetBuilder import DatasetBuilder

directory = "C:/Users/allan/Downloads/FacesDatasetDepth"
img_size = 512
base_filters = 32
device = torch.device('cuda')
epochs = 500
batch_size = 2
scaling_factor = 0.18215
learning_rate = 0.0001

dataset_builder = DatasetBuilder(directory=directory, img_size=img_size, batch_size=batch_size)
dataset = dataset_builder.get_dataset()

auto_encoder = AutoEncoder(img_size=img_size, scaling_factor=scaling_factor, device=device, base_filters=base_filters)
auto_encoder.initialize_optimizers_loss(learning_rate=learning_rate)
auto_encoder.train_autoencoder(dataset=dataset, epochs=epochs, device=device)

