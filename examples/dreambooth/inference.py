from diffusers import StableDiffusionPipeline
import torch

# model_id = "outputs/dog_single"
# model_id = "outputs/cat_single"
model_id = "outputs/multi"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

prompt = "a dog and a cat"
# prompt = "A photo of yhb cat in a bucket"
# prompt = "A photo of a yhb cat fighting a dog"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save("two-dog.png")