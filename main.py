import torch
from DatasetBuilder import DatasetBuilder
from DiffusionModel import DiffusionModel


directory = "C:/Users/allan/Downloads/FacesDatasetDepth"
img_size = 512
latents_size = 64
layers_per_block = 2
base_filters = 64
device = torch.device('cuda')
epochs = 500
batch_size = 4
model_path = "Models/Epoch46.pt"

dataset_builder = DatasetBuilder(directory=directory, img_size=img_size, batch_size=batch_size)
dataset = dataset_builder.get_dataset()

diffusion_model = DiffusionModel()
#diffusion_model.create_model(latents_size=latents_size, layers_per_block=layers_per_block, base_filters=base_filters, device=device)
diffusion_model.load_model(model_path, device=device)
diffusion_model.initialize_opt_loss_function()
diffusion_model.setup_autoencoder(device=device)
diffusion_model.train_model(dataset=dataset, epochs=epochs, device=device, latent_size=latents_size, starting_epoch = 47)

#diffusion_model.generate_image(device=device, num_images=1, img_size=img_size)