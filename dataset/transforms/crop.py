import numpy as np
import cv2

TOP_LEFT_COORD = [21, 2]
BOTTOM_RIGHT_COORD = [148, 129]

class CropAnd2GRAY(object):
    """
    Crop the image to eliminate borders in a sample.
    Grayscale the image to reduce parameters.
    """
    def __call__(self, sample):
        image, correlations = sample['image'], sample['correlations']

        image = image[TOP_LEFT_COORD[1]: BOTTOM_RIGHT_COORD[1],
                      TOP_LEFT_COORD[0]: BOTTOM_RIGHT_COORD[0],
                      :]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.array(image)/255
        image = np.expand_dims(image, axis=2)

        # cv2.imshow('vis', image)
        # cv2.waitKey(0)

        return {'image': image, 'correlations': correlations}