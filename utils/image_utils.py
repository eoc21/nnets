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

    def download_image(self, url):
        downloaded_image = requests.get(url)
        im = array(Image.open(BytesIO(downloaded_image.content)))
        imshow(im)
        show()



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

if __name__ == '__main__':
    pass