import numpy as np


class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        height, weight = image.shape[:2]
        new_height, new_weight = self.output_size

        top = np.random.randint(0, height - new_height)
        left = np.random.randint(0, weight - new_weight)

        image = image[top: top + new_height, left: left + new_weight]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}
