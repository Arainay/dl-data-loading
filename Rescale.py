from skimage import transform


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        height, weight = image.shape[:2]
        if isinstance(self.output_size, int):
            if height > weight:
                new_height, new_weight = self.output_size * height / weight, self.output_size
            else:
                new_height, new_weight = self.output_size, self.output_size * weight / height

        else:
            new_height, new_weight = self.output_size

        new_height, new_weight = int(new_height), int(new_weight)

        img = transform.resize(image, (new_height, new_weight))
        landmarks = landmarks * [new_weight / weight, new_height / height]

        return {'image': img, 'landmarks': landmarks}
