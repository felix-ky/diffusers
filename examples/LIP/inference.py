import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.utils import save_image
import  numpy as np
import random
import os
import argparse

transforms.Resize
# generate random masks
def random_mask(im_shape, id_file, mask_root):
    image_transforms_resize_and_crop = transforms.Compose(
        [
            transforms.Resize((im_shape[0] // 4 * 2 + im_shape[0] % 4, im_shape[1] // 4 * 2 + im_shape[1] % 4), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Pad((im_shape[1] // 4, im_shape[0] // 4, im_shape[1] // 4, im_shape[0] // 4), fill=0),
            transforms.CenterCrop((im_shape[0], im_shape[1])),
        ]
    )
    with open(id_file, 'r') as file:
        names = file.read().splitlines()
    name = names[random.randint(0, len(names) - 1)]
    mask = Image.open(mask_root + name + '.png')
    mask = image_transforms_resize_and_crop(mask)
    
    mask = transforms.ToTensor()(mask.convert("L"))
    mask[mask > 0] = 1
    mask = transforms.ToPILImage('L')(mask)
    
    return mask

def difference_grid(image, mask, inpaint):
    w, h = image.size
    grid = Image.new("RGB", size=(w * 3, h))
    grid.paste(image, box=(0 * w, 0 * h))
    grid.paste(mask, box=(1 * w, 0 * h))
    grid.paste(inpaint, box=(2 * w, 0 * h))
    return grid

temp_transforms = transforms.Compose(
    transforms=[
        transforms.Resize(size=512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(size=512),
    ]
)

parser = argparse.ArgumentParser(description="How to inpaint.")
parser.add_argument(
    "--save_root",
    type=str,
    default=None,
    help="A root to store the root."
)
parser.add_argument(
    "--id_file",
    type=str,
    default=None,
    help="A file help find masks."
)
parser.add_argument(
    "--mask_root",
    type=str,
    default=None,
    required=True,
    help="A folder containing the masks.",
)
parser.add_argument(
    "--model_folder",
    type=str,
    default=None,
    required=True,
    help="The weight aftert the training of inpainting.",
)
parser.add_argument(
    "--prompt",
    type=str,
    default=None,
    required=True,
    help="Description about the background.",
)
args = parser.parse_args()

model_folder = os.path.join(args.save_root, args.model_folder)
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model_folder,
    revision="fp16",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

prompt = args.prompt

if __name__ == "__main__":
    if not os.path.exists(args.save_root): os.mkdir(args.save_root)
    id_file = os.path.join(args.save_root, "./img/id_file.txt")
    image_path = os.path.join(args.save_root, "./img/origin/")
    mask_path = os.path.join(args.save_root, "./img/mask/")
    inpaint_path = os.path.join(args.save_root, "./img/inpaint/")
    difference_path = os.path.join(args.save_root, "./img/difference/")
    # if not os.path.exists(image_path): os.mkdir(image_path)
    if not os.path.exists(mask_path): os.mkdir(mask_path)
    if not os.path.exists(inpaint_path): os.mkdir(inpaint_path)
    if not os.path.exists(difference_path): os.mkdir(difference_path)
    with open(id_file, 'r') as file:
        names = file.readlines()
        names = map(lambda a: a.strip(), names)
    for name in names:
        image = Image.open(os.path.join(image_path, name + '.png'))
        image = temp_transforms(img=image)
        mask = random_mask((image.height, image.width), args.id_file, args.mask_root)
        mask.save(os.path.join(mask_path, name + ".png"))
        inpaint = pipe(prompt=prompt, image=image, mask_image=mask).images[0]
        inpaint.save(os.path.join(inpaint_path, name + '.png'))
        difference = difference_grid(image=image, mask=mask, inpaint=inpaint)
        difference.save(os.path.join(difference_path, name + '.png'))