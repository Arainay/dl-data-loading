from torch import from_numpy


class ToTensor(object):
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        image = image.transpose((2, 0, 1))
        return {'image': from_numpy(image), 'landmarks': from_numpy(landmarks)}
