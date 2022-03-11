import torch
import torchvision.transforms as T
from PIL import Image

# constants

CHANNELS_TO_MODE = {
    1 : 'L',
    3 : 'RGB',
    4 : 'RGBA'
}

def seek_all_images(img, channels = 3):
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]

    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1

# tensor of shape (frame, channels, height, width) -> gif

def video_tensor_to_gif(tensor, path, duration = 80, loop = 0, optimize = True):
    images = map(T.ToPILImage(), tensor.unbind(0))
    first_img, *rest_imgs = images
    first_img.save(path, save_all = True, append_images = rest_imgs, duration = duration, loop = loop, optimize = optimize)
    return images

# gif -> (frame, channels, height, width) tensor

def gif_to_tensor(path, channels = 3):
    img = Image.open(path)
    tensors = tuple(map(T.ToTensor(), seek_all_images(img, channels = channels)))
    return torch.stack(tensors, dim = 0)
