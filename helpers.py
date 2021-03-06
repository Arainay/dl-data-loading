from torchvision import utils
import matplotlib.pyplot as plt


def show_landmarks_batch(sample_batched):
    images_batch, landmarks_batch = sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(
            landmarks_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
            landmarks_batch[i, :, 1].numpy() + grid_border_size,
            s=10,
            marker='.',
            c='r'
        )
        plt.title('Batch from dataloader')
