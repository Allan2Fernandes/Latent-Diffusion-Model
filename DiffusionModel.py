import numpy
import torch.optim
from diffusers import DDPMScheduler, StableDiffusionPipeline
from diffusers import UNet2DModel
import matplotlib.pyplot as plt

class DiffusionModel:
    def __init__(self):
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='linear')
        pass

    def create_model(self, latents_size, layers_per_block, base_filters, device):
        self.model = UNet2DModel(
            sample_size=latents_size,
            in_channels=4,
            out_channels=4,
            layers_per_block=layers_per_block,
            block_out_channels=(base_filters, base_filters*2, base_filters*4, base_filters*8),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D"
            )
        )
        self.model.to(device=device)
        for param in self.model.parameters():
            print(param.shape)
        pass

    def load_model(self, path, device):
        self.model = torch.load(path)
        self.model.to(device)
        pass

    def get_model(self):
        return self.model

    def setup_autoencoder(self, device):
        model_id = "stabilityai/stable-diffusion-2-1-base"
        self.vae = StableDiffusionPipeline.from_pretrained(model_id).to(torch.device(device)).vae
        self.vae.enable_xformers_memory_efficient_attention()
        self.vae_scaling_factor = self.vae.config.scaling_factor
        pass

    def encode_images_to_latents(self, images):
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample() * self.vae_scaling_factor
            pass
        return latents

    def decode_latents_to_images(self, latents):
        with torch.no_grad():
            latents = latents / self.vae_scaling_factor
            decoded_images = self.vae.decode(latents.detach()).sample
            pass
        return decoded_images

    def initialize_opt_loss_function(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00005)
        self.loss_function = torch.nn.functional.mse_loss

    def train_model(self, dataset, epochs, device, latent_size, starting_epoch=1):
        self.vae.requires_grad_(False)
        num_steps = len(dataset)
        for epoch in range(starting_epoch, epochs+1):
            for step, (image, label) in enumerate(dataset):
                image = image.to(device)
                batch_size = image.shape[0]
                #Reset the gradient every step unless using gradient accumulation
                self.optimizer.zero_grad()
                #Get random timestep
                timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (batch_size,), device=device)
                latents = self.encode_images_to_latents(image)
                #Create noise
                noise = torch.randn(latents.shape).to(device)
                #noise the images using the noise and timestep
                noisy_latents = self.noise_scheduler.add_noise(latents, noise=noise, timesteps=timesteps)
                #Get noise prediction
                noise_prediction = self.model(noisy_latents, timesteps)[0]
                #Get loss between pred and noise
                loss = self.loss_function(noise_prediction, noise)
                #differentiate loss
                loss.backward()
                #Apply gradient step
                self.optimizer.step()
                print("Epoch = {0} ||step = {1}/{3} || loss = {2}".format(epoch, step, loss, num_steps))
                # if (step+1)%500 == 0:
                #     self.show_decoded_latents(latent_size, device)
                pass
            torch.save(self.model, f"Models/Epoch{epoch}.pt")
            self.show_decoded_latents(latent_size, device)
            pass
        pass

    def show_decoded_latents(self, latent_size, device):
        #Create noisy latents
        latents = torch.randn(1, 4, latent_size, latent_size).to(device=device)
        #Denoise the latents
        for step, t in enumerate(self.noise_scheduler.timesteps):
            with torch.no_grad():
                #Get noise prediction
                noise_preds = self.model(latents, t).sample
                pass
            #Using noise pred and the image at the timestep t, calculate the previous sample
            latents = self.noise_scheduler.step(noise_preds, t, latents).prev_sample
            pass
        #Decode the latents
        decoded_images = self.decode_latents_to_images(latents)
        images = torch.permute(decoded_images, (0, 2, 3, 1))
        image = images[0]
        image = image.to('cpu')
        image = numpy.asarray(image)
        image = image * 0.5 + 0.5
        plt.imshow(image)
        plt.show()
        pass


    def generate_image(self, num_images, img_size, device):
        images = torch.randn(num_images, 3, img_size, img_size).to(device=device)
        for step, t in enumerate(self.noise_scheduler.timesteps):
            with torch.no_grad():
                #Get noise prediction
                noise_preds = self.model(images, t).sample
                pass
            #Using noise pred and the image at the timestep t, calculate the previous sample
            images = self.noise_scheduler.step(noise_preds, t, images).prev_sample
            pass
        print(images.shape)
        images = torch.permute(images, (0, 2, 3, 1))
        image = images[0]
        image = image.to('cpu')
        image = numpy.asarray(image)
        image = image*0.5 + 0.5
        plt.imshow(image)
        plt.show()
        return images
