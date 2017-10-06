import pywt
import numpy
from PIL import Image, ImageEnhance

import os

import numpy as np
import pywt
from PIL import Image
from scipy.fftpack import dct
from scipy.fftpack import idct

class Watermarker:

    def print_image_from_array(self,image_array, name):
        image_array_copy = image_array.astype("uint8")
        img = Image.fromarray(image_array_copy)
        img.convert('RGB').save(name)

    def process_coefficients(self,imArray, model, level):
        coeffs = pywt.wavedec2(data=imArray, wavelet=model, level=level)
        coeffs_H = list(coeffs)

        return coeffs_H

    def apply_dct(self,image_array):
        img_bits = 24
        print image_array.shape
        x, y,z = image_array.shape
        all_subdct = np.empty((x, y,z))
        for i in range(0, x, img_bits):
            for j in range(0, y, img_bits):
                for k in range(0, z, img_bits):
                    subpixels = image_array[i:i + img_bits, j:j + img_bits, k:k + img_bits]
                    subdct = dct(dct(subpixels.T, norm="ortho").T, norm="ortho")
                    all_subdct[i:i + img_bits, j:j + img_bits, k:k + img_bits] = subdct
        return all_subdct

    def embed_watermark(self, watermark_array, orig_image):
        watermark_array_size = watermark_array[0].__len__()
        watermark_flat = watermark_array.ravel()
        print watermark_flat,'flat array=================before watermark'
        ind = 0
        img_bits = 24
        x1, y1, z1 = orig_image.shape
        for x in range(0, x1, img_bits):
            for y in range(0, y1, img_bits):
                for z in range(0, z1, img_bits):
                    if ind < watermark_flat.__len__():
                        subdct = orig_image[x:x + img_bits, y:y + img_bits, z:z + img_bits]
                        subdct[5][5][0] = watermark_flat[ind]
                        orig_image[x:x + img_bits, y:y + img_bits, z:z + img_bits] = subdct
                        ind += 1
        return orig_image

    def inverse_dct(self, all_subdct):
        x, y, z = all_subdct.shape
        all_subidct = np.empty((x, y, z))
        img_bits = 24
        for i in range(0, x, img_bits):
            for j in range(0, y, img_bits):
                for k in range(0, z, img_bits):
                    subidct = idct(idct(all_subdct[i:i + img_bits, j:j + img_bits, k:k + img_bits].T, norm="ortho").T, norm="ortho")
                    all_subidct[i:i + img_bits, j:j + img_bits, k:k + img_bits] = subidct

        return all_subidct

    def get_watermark(self,dct_watermarked_coeff, watermark_size):
        # watermark = [[0 for x in range(watermark_size)] for y in range(watermark_size)]

        subwatermarks = []
        img_bits = 24
        for x in range(0, dct_watermarked_coeff.__len__(), img_bits):
            for y in range(0, dct_watermarked_coeff.__len__(), img_bits):
                coeff_slice = dct_watermarked_coeff[x:x + img_bits, y:y + img_bits]
                subwatermarks.append(coeff_slice[4][4])

        watermark = np.array(subwatermarks).reshape(1, 507)

        return watermark

    def recover_watermark(self,image_array, model='haar', level=1):

        coeffs_watermarked_image = self.process_coefficients(image_array, model, level=level)
        dct_watermarked_coeff = self.apply_dct(coeffs_watermarked_image[0])

        watermark_array = self.get_watermark(dct_watermarked_coeff, 128)

        # watermark_array *= 255;
        watermark_array = np.uint8(watermark_array)

        # Save result
        img = Image.fromarray(watermark_array)
        img.save('recovered_watermark.jpg')

    def spike(self):
        model = 'haar'
        level = 0
        original_image = Image.open('1.jpg')
        watermark_image = Image.open('logo.png')
        original_image_array = numpy.asarray(original_image)
        watermark_image_array = numpy.asarray(watermark_image)
        print original_image_array
        print '#######'
        coeffs_image = self.process_coefficients(original_image_array, model, level)
        self.print_image_from_array(coeffs_image[0], 'LL_after_DWT.jpg')

        dct_array = self.apply_dct(coeffs_image[0])
        self.print_image_from_array(dct_array, 'LL_after_DCT.jpg')

        dct_array = self.embed_watermark(watermark_image_array, dct_array)
        self.print_image_from_array(dct_array, 'LL_after_embeding.jpg')

        coeffs_image[0] = self.inverse_dct(dct_array)
        self.print_image_from_array(coeffs_image[0], 'LL_after_IDCT.jpg')

        # reconstruction
        image_array_H = pywt.waverec2(coeffs_image, model)
        self.print_image_from_array(image_array_H, 'image_with_watermark.jpg')

        # recover images
        # image_array = convert_image('cropped_image.jpg', 2048)
        self.recover_watermark(image_array = image_array_H, model=model, level = level)

    def __main__(self):
        print "Starting..."
        self.spike()
        # dwt_image = pywt.dwtn(original_image_array, 'db2')
        # shape=dwt_image[0].shape
        # dwt_image_flatten = dwt_image[0].flatten()
        #
        # water_mark_string = "People's archive of rural India"
        # water_marked_array = numpy.fromstring(water_mark_string, numpy.int8)
        #
        # encrypted_array = ndimage.convolve(dwt_image_flatten, water_marked_array, mode='constant', cval=0.0)
        #
        # dwt_encrypted_array = encrypted_array.reshape((950, 1400, 3))
        #
        # dwt_retransform_db3 = pywt.waverecn([dwt_encrypted_array], 'db3')
        # dwt_image_db3 = Image.fromarray(dwt_retransform_db3, 'RGB')
        # dwt_image_db3.save('watermarker/dwt_image_db3.jpg')
        # water_mark_string = "People's archive of rural India"
        # water_string_to_ascii_array = numpy.asarray(water_mark_string,dtype=numpy.int8)
        # ndimage.convolve(Arrays, water_marked_array, mode='constant', cval=0.0)
        # nd_array_encrypted = encrypted_array.reshape((950, 1400, 3))


# water=Watermarker()
# water.__main__()

def reduceOpacity(im, opacity):
    """Returns an image with reduced opacity."""
    assert opacity >= 0 and opacity <= 1
    if im.mode != 'RGBA':
        im = im.convert('RGBA')
    else:
        im = im.copy()
    alpha = im.split()[3]
    alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
    im.putalpha(alpha)
    return im

def watermark(im, mark, position, opacity=1):
    """Adds a watermark to an image."""
    if opacity < 1:
        mark = reduceOpacity(mark, opacity)
    if im.mode != 'RGBA':
        im = im.convert('RGBA')
    # create a transparent layer the size of the image and draw the
    # watermark in that layer.
    layer = Image.new('RGBA', im.size, (0,0,0,0))
    if position == 'tile':
        for y in range(0, im.size[1], mark.size[1]):
            for x in range(0, im.size[0], mark.size[0]):
                layer.paste(mark, (x, y))
    elif position == 'scale':
        # scale, but preserve the aspect ratio
        ratio = min(float(im.size[0]) / mark.size[0], float(im.size[1]) / mark.size[1])
        w = int(mark.size[0] * ratio)
        h = int(mark.size[1] * ratio)
        mark = mark.resize((w, h))
        layer.paste(mark, ((im.size[0] - w) / 2, (im.size[1] - h) / 2))
    else:
        layer.paste(mark, position)
    # composite the watermark with the layer
    return Image.composite(layer, im, layer)

img = Image.open('1.jpg')

mark1 = Image.open('test.jpg')
mark2 = Image.open('test.jpg')

img = watermark(img, mark1, (img.size[0]-mark1.size[0]-5, img.size[1]-mark1.size[1]-5), 1)
img = watermark(img, mark2, 'scale', 1)