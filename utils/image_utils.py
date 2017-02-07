__author__ = 'edwardcannon'


from numpy import *
from PIL import Image
import requests
from io import BytesIO
from pylab import *
from numpy import *
from scipy.ndimage import filters
import scipy.misc


class ImageDownloader(object):
    """
    Downloads images from internet
    """
    def __init__(self):
        self.images = []

    @staticmethod
    def download_image(url):
        """
        Downloads image given url
        :param url: url to extract image
        """
        downloaded_image = requests.get(url)
        return array(Image.open(BytesIO(downloaded_image.content)))


class ImageCleaner(object):
    """
    Denoises set of images
    """
    def __init__(self):
        pass

    def denoise(self, im, u_init, tolerance=0.1,
                tau=0.125, tv_weight=100):
        """
        Applies Rudin-Osher-Fatemi denoising model to images
        :param im: Input image
        :param u_init: initial guess for u, the denoised image
        :param tolerance: stop criterion
        :param tau: step lenght
        :param tv_weight: TV-regularisation term
        :return:
        """
        m, n = im.shape[0:2]
        u = u_init
        px = im
        py = im
        error = 1

        while (error > tolerance):
            u_old = u
            #gradient of primal var
            grad_ux = roll(u, -1, axis=1)-u
            grad_uy = roll(u, -1, axis=0)-u

            #update dual var
            px_new = px + (tau/tv_weight)*grad_ux
            py_new = py + (tau/tv_weight)*grad_uy
            norm_new = maximum(1, sqrt(px_new**2+py_new**2))

            px = px_new/norm_new
            py = py_new/norm_new
            rx_px = roll(px, 1, axis=1)
            ry_py = roll(py, 1, axis=0)

            div_p = (px-rx_px)+(py-ry_py)
            u = im + tv_weight*div_p

            error = linalg.norm(u-u_old)/sqrt(n*m)
        return u, im-u

class KnnClassifier(object):
    """
    Uses K-NN classifier to
    categorize an image
    """
    def __init__(self, labels, samples):
        self.labels = labels
        self.samples = samples

    def classify(self, image, k=3):
        """
        Classifies
        :param image: image pixels
        :param k: Number of neighbours
        :return:
        """
        dist = array([self.__euclidean_distance(image, train_sample) for train_sample in self.samples])
        ndx = dist.argsort()
        votes = {}
        for i in range(k):
            label = self.labels[ndx[i]]
            votes.setdefault(label, 0)
            votes[label] += 1
        return max(votes)

    def __euclidean_distance(self, img1, img2):
        """
        Calculates Euclidean distance between 2 images
        :param img1: Image 1 array
        :param img2: Image 2 array
        :return:
        """
        return sqrt(sum((img1-img2)**2))





if __name__ == '__main__':
    pass