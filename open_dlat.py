from os import listdir
from os.path import isfile, join
from numpy import load
import tensorflow as tf

import PIL
import dnnlib
import dnnlib.tflib as tflib
import pickle


data = load('out/dlatents0.npz')
lst = data.files
for item in lst:
    print(item)
    print(data[item].shape)

# mypath = "images/"
# onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
# print(onlyfiles)

print(data[item])

network_pkl = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/ffhq.pkl"

# If downloads fails, due to 'Google Drive download quota exceeded' you can try downloading manually from your own Google Drive account
# network_pkl = "/content/drive/My Drive/GAN/stylegan2-ffhq-config-f.pkl"

tflib.init_tf({'rnd.np_random_seed': 303})
print('Loading networks from "%s"...' % network_pkl)
with dnnlib.util.open_url(network_pkl) as fp:
        _G, _D, Gs = pickle.load(fp)
# self._images_float_expr



image_float_expr = tf.cast(Gs.components.synthesis.get_output_for(data[item]), tf.float32)
images_uint8_expr = tflib.convert_images_to_uint8(image_float_expr, nchw_to_nhwc=True)
PIL.Image.fromarray(tflib.run(images_uint8_expr)[0], 'RGB').save(f'out/proj.png')
