import numpy as np

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as an H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
            return img[:, ::-1]
        return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NDArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        if len(img.shape) == 2:
            h, w = img.shape
            padded_img = np.zeros((h+2*self.padding, w+2*self.padding))
            padded_img[self.padding:self.padding+h, self.padding:self.padding+w] = img
            start_x = self.padding + shift_x
            start_y = self.padding + shift_y
            return padded_img[start_x:start_x+h, start_y:start_y+w]
        else:
            h, w, c = img.shape
            padded_img = np.zeros((h+2*self.padding, w+2*self.padding, c))
            padded_img[self.padding:self.padding+h, self.padding:self.padding+w, :] = img
            start_x = self.padding + shift_x
            start_y = self.padding + shift_y
            return padded_img[start_x:start_x+h, start_y:start_y+w, :]
        ### END YOUR SOLUTION
