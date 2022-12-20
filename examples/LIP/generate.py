from diffusers import StableDiffusionPipeline
import torch
import os
import argparse

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16)
pipe.to("cuda")

num_images = 8
parser = argparse.ArgumentParser(description="How to generate background.")
parser.add_argument(
    "--save_root",
    type=str,
    default=None,
    help="A root to store the root."
)
parser.add_argument(
    "--background",
    type=str,
    default=None,
    required=True,
    help="Description about the background.",
)
args = parser.parse_args()
prompt = [args.background] * num_images
images = pipe(prompt=prompt).images

save_path = os.path.join(args.save_root, "./img/origin/")
id_file_path = os.path.join(args.save_root, "./img/id_file.txt")
ids = []

if __name__ == "__main__":
    if not os.path.exists(args.save_root): os.mkdir(args.save_root)
    if not os.path.exists(save_path): os.mkdir(save_path)
    for index, image in enumerate(images):
        image.save(os.path.join(save_path, str(index) + '.png'))
        ids.append(str(index) + '\n')
        
    with open(id_file_path, "w") as file:
        file.writelines(ids)