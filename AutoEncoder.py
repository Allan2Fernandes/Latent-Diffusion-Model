import torch
from diffusers import AutoencoderKL
from torch.nn import functional as F
import matplotlib.pyplot as plt

class AutoEncoder():
    def __init__(self, img_size, scaling_factor, device, base_filters):
        self.scaling_factor = scaling_factor
        self.autoencoder = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            block_out_channels=(base_filters, base_filters*2, base_filters*2, base_filters*4),
            down_block_types=(
                'DownEncoderBlock2D',
                'DownEncoderBlock2D',
                'DownEncoderBlock2D',
                'DownEncoderBlock2D'
            ),
            up_block_types=(
                'UpDecoderBlock2D',
                'UpDecoderBlock2D',
                'UpDecoderBlock2D',
                'UpDecoderBlock2D'
            ),
            latent_channels=4,
            act_fn='silu',
            sample_size=img_size,
            scaling_factor=scaling_factor
        )
        self.autoencoder.to(device)
        pass

    def initialize_optimizers_loss(self, learning_rate):
        self.loss = F.mse_loss
        self.optimzer = torch.optim.Adam(self.autoencoder.parameters(), lr=learning_rate)
        pass

    def draw_image(self, image_batch):
        image = image_batch[0]
        image = torch.permute(image, (1,2,0))
        image = image/2 +0.5
        image = image.detach()
        image = image.to('cpu').numpy()
        plt.imshow(image)
        plt.show()
        pass


    def train_autoencoder(self, dataset, epochs, device):
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [4900, 100])

        self.autoencoder.train()
        for epoch in range(1, epochs+1):
            num_steps = len(dataset)
            for index, image_batch in enumerate(dataset):

                image_batch = image_batch[0]
                image_batch = image_batch.to(device = device)
                self.optimzer.zero_grad()
                labels = image_batch
                latents = self.scaling_factor*self.autoencoder.encode(image_batch).latent_dist.sample()
                outputs = self.autoencoder.decode(latents/self.scaling_factor).sample
                loss = self.loss(outputs, labels)
                if index%100 == 0:
                    self.draw_image(outputs)
                pass
                loss.backward()
                self.optimzer.step()
                print("Epoch = {0} || Step = {1}/{2} || Loss = {3}".format(epoch, index, num_steps, loss))


            pass



