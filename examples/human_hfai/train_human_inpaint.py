import argparse
import hashlib
import itertools
import math
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image, ImageDraw
from torchvision.prototype import transforms  # TODO: migrate to pytorch 2.0 for easy transform impl
# from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import time

import hfai

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

ip = os.environ['MASTER_IP']
port = os.environ['MASTER_PORT']
world_size = int(os.environ['WORLD_SIZE'])  # 机器个数
rank = int(os.environ['RANK'])  # 当前机器编号
local_rank = int(os.environ['LOCAL_RANK'])
gpus = torch.cuda.device_count()  # 每台机器的GPU个数

def prepare_mask_and_masked_image(image, mask):
    """
    args:
        image: PIL image
        mask: torch tensor

    TODO: reuse instance images to avoid redundancy
    """
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    # mask = np.array(mask.convert("L"))
    # mask = mask.astype(np.float32) / 255.0
    mask = mask[None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    # mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    return mask, masked_image


# generate random masks
def random_mask(im_shape, ratio=1, mask_full_image=False):
    mask = Image.new("L", im_shape, 0)
    draw = ImageDraw.Draw(mask)
    size = (random.randint(0, int(im_shape[0] * ratio)), random.randint(0, int(im_shape[1] * ratio)))
    # use this to always mask the whole image
    if mask_full_image:
        size = (int(im_shape[0] * ratio), int(im_shape[1] * ratio))
    limits = (im_shape[0] - size[0] // 2, im_shape[1] - size[1] // 2)
    center = (random.randint(size[0] // 2, limits[0]), random.randint(size[1] // 2, limits[1]))
    draw_type = random.randint(0, 1)
    if draw_type == 0 or mask_full_image:
        draw.rectangle(
            (center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2),
            fill=255,
        )
    else:
        draw.ellipse(
            (center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2),
            fill=255,
        )

    return mask


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default="a photo of a coco person",
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If not have enough images, additional images will be"
            " sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder")
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.instance_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")

    return args


class HumanDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        # tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        # self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images


        self.class_data_root = None

        self.image_transforms_resize_and_crop = transforms.Compose(
            [
                transforms.RandomShortestSize(size+3), # TODO: avoid min size failure
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            ]
        )

        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.mask_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        # get mask image
        mask_path = self.instance_images_path[index % self.num_instance_images]
        mask = Image.open(mask_path)

        # TODO: now the instance path is hard coded to be under masks parent dir, should change
        image_path = str(mask_path).replace("masks", "images")
        instance_image = Image.open(image_path)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        
        instance_image, mask = self.image_transforms_resize_and_crop(instance_image, mask)

        example["PIL_images"] = instance_image
        example["masks"] = self.mask_transforms(mask)
        example["instance_images"] = self.image_transforms(instance_image)
        
        # pre load the fix prompt to avoid loading clip
        # example["instance_prompt_ids"] = self.tokenizer(
        #     self.instance_prompt,
        #     padding="do_not_pad",
        #     truncation=True,
        #     max_length=self.tokenizer.model_max_length,
        # ).input_ids

        return example


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def main():
    print(f"{rank}: start")
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    dist.init_process_group(backend='nccl',
                            init_method=f'tcp://{ip}:{port}',
                            world_size=world_size,
                            rank=rank)
    torch.cuda.set_device(local_rank)
    print("{}: ".format(dist.get_rank()) + "finish distributed")

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    # Handle the repository creation
    if dist.get_rank() == 0:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

    # Load models and create wrapper for stable diffusion
    print("{}: ".format(dist.get_rank()) + "start to load models")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder").to("cuda")
    text_encoder = DistributedDataParallel(text_encoder, device_ids=[local_rank])
    print("{}: ".format(dist.get_rank()) + "finish text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae").to("cuda")
    vae = DistributedDataParallel(vae, device_ids=[local_rank])
    print("{}: ".format(dist.get_rank()) + "finish vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet").to("cuda")
    unet = DistributedDataParallel(unet, device_ids=[local_rank])
    print("{}: ".format(dist.get_rank()) + "finish unet")

    vae.requires_grad_(False)
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * world_size
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # disable text_encoder
    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder else unet.parameters()
        # itertools.chain(unet.parameters())
    )
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    print(f"{rank}: finish initialize models, optimizer, schedulers")
    
    train_dataset = HumanDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        # tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
    )
    train_sampler = DistributedSampler(train_dataset)

    def collate_fn(examples):
        # prompt input disabled
        # input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if args.with_prior_preservation:
            # input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]
            pior_pil = [example["class_PIL_images"] for example in examples]

        masks = []
        masked_images = []

        # TODO: move this to the dataloader for potential speedup
        # or at least batchify this (loop is slow)
        for example in examples:
            pil_image = example["PIL_images"]
            mask = example["masks"]
            # generate a random mask
            # mask = random_mask(pil_image.size, 1, False)
            # prepare mask and masked image
            mask, masked_image = prepare_mask_and_masked_image(pil_image, mask)

            masks.append(mask)
            masked_images.append(masked_image)

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        # input_ids = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids
        masks = torch.stack(masks)
        masked_images = torch.stack(masked_images)
        # prompt input disabled
        # batch = {"input_ids": input_ids, "pixel_values": pixel_values, "masks": masks, "masked_images": masked_images}
        batch = {"pixel_values": pixel_values, "masks": masks, "masked_images": masked_images}
        return batch

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=True
    )
    print(f"{rank}: finish initialize data")

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.

    # Text encoder disabled for now
    # if not args.train_text_encoder:
    #    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if dist.get_rank() == 0:
        tracker = SummaryWriter(log_dir=logging_dir, comment="human_inpaint")

    # pre load the text encoder result and release text_encoder
    input_ids = tokenizer(
         args.instance_prompt,
         padding="do_not_pad",
         truncation=True,
         max_length=tokenizer.model_max_length,
    ).input_ids
    # make this batch size one. TODO: repeat batch size times if cannot broadcast
    input_ids = [input_ids] * args.train_batch_size
    input_ids = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids
    # input_ids: 
    # tensor([[49406,   320,  1125,   539,   320, 13316,  2533, 49407]],
    #        device='cuda:0')
    encoder_hidden_states = text_encoder(input_ids)[0].cuda().to(dtype=weight_dtype)
    # it seems that text_encoder is not loaded into gpu and we have to save it for the pipeline
    # del text_encoder

    # Train!
    total_batch_size = args.train_batch_size * world_size * args.gradient_accumulation_steps

    if dist.get_rank() == 0:
        print("***** Running training *****")
        print(f"  Num examples = {len(train_dataset)}")
        print(f"  Num batches each epoch = {len(train_dataloader)}")
        print(f"  Num Epochs = {args.num_train_epochs}")
        print(f"  Instantaneous batch size per device = {args.train_batch_size}")
        print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        print(f"  Total optimization steps = {args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=dist.get_rank() != 0)
        progress_bar.set_description("Steps")
    global_step = 0

    print(f"{rank}: start training")
    for epoch in range(args.num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            # Convert images to latent space

            latents = vae.module.encode(batch["pixel_values"].to(dtype=weight_dtype).to("cuda")).latent_dist.sample()
            latents = latents * 0.18215

            # Convert masked images to latent space
            masked_latents = vae.module.encode(
                batch["masked_images"].reshape(batch["pixel_values"].shape).to(dtype=weight_dtype).to("cuda")
            ).latent_dist.sample()
            masked_latents = masked_latents * 0.18215

            masks = batch["masks"].to("cuda")
            # resize the mask to latents shape as we concatenate the mask to the latents
            mask = torch.stack(
                [
                    torch.nn.functional.interpolate(mask, size=(args.resolution // 8, args.resolution // 8))
                    for mask in masks
                ]
            )
            mask = mask.reshape(-1, 1, args.resolution // 8, args.resolution // 8).to(dtype=weight_dtype)

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents, device="cuda")
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # concatenate the noised latents with the mask and the masked latents
            latent_model_input = torch.cat([noisy_latents, mask, masked_latents], dim=1)

            # Use precomputed prompt
            # Get the text embedding for conditioning
            # encoder_hidden_states = text_encoder(batch["input_ids"])[0]

            # Predict the noise residual
            noise_pred = unet(latent_model_input, timesteps, encoder_hidden_states).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            if args.with_prior_preservation:
                # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
                noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
                target, target_prior = torch.chunk(target, 2, dim=0)

                # Compute instance loss
                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none").mean([1, 2, 3]).mean()

                # Compute prior loss
                prior_loss = F.mse_loss(noise_pred_prior.float(), target_prior.float(), reduction="mean")

                # Add the prior loss to the instance loss.
                loss = loss + args.prior_loss_weight * prior_loss
            else:
                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

            # if accelerator.sync_gradients:
            #     params_to_clip = (
            #         itertools.chain(unet.parameters(), text_encoder.parameters())
            #         if args.train_text_encoder
            #         else unet.parameters()
            #     )
            #     accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if dist.get_rank() == 0:
                progress_bar.update(1)
            global_step += 1

            if dist.get_rank() == 0:
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                print(f"step {global_step}: ", logs)
                tracker.add_scalars("densepose_512",logs, global_step=global_step)

            if global_step >= args.max_train_steps:
                break
        
            if dist.get_rank() == 0 and hfai.receive_suspend_command():
                    # pipeline = StableDiffusionPipeline.from_pretrained(
                    #     args.pretrained_model_name_or_path,
                    #     unet=unet,
                    #     text_encoder=text_encoder,
                    # )
                    if not os.path.exists(os.path.join(args.output_dir, './latest/')): os.mkdir(os.path.join(args.output_dir, './latest/'))
                    # pipeline.save_pretrained(os.path.join(args.output_dir, './latest/'))
                    torch.save(text_encoder.state_dict(), os.path.join(os.path.join(args.output_dir, './latest/'), './text_encoder.ckpt'))
                    torch.save(vae.state_dict(), os.path.join(os.path.join(args.output_dir, './latest/'), './vae.ckpt'))
                    torch.save(unet.state_dict(), os.path.join(os.path.join(args.output_dir, './latest/'), './unet.ckpt'))
                    time.sleep(5)
                    hfai.go_suspend()

        # print("wait for rank 0")
        dist.barrier()

    print(f"{local_rank}: finish train")
    
    # Create the pipeline using using the trained modules and save it.
    if dist.get_rank() == 0:
        # pipeline = StableDiffusionPipeline.from_pretrained(
        #     args.pretrained_model_name_or_path,
        #     unet=unet,
        #     text_encoder=text_encoder,
        # )
        # pipeline.save_pretrained(args.output_dir)
        if not os.path.exists(os.path.join(args.output_dir, './final/')): os.mkdir(os.path.join(args.output_dir, './final/'))
        # pipeline.save_pretrained(os.path.join(args.output_dir, './latest/'))
        torch.save(text_encoder.state_dict(), os.path.join(os.path.join(args.output_dir, './final/'), './text_encoder.ckpt'))
        torch.save(vae.state_dict(), os.path.join(os.path.join(args.output_dir, './final/'), './vae.ckpt'))
        torch.save(unet.state_dict(), os.path.join(os.path.join(args.output_dir, './final/'), './unet.ckpt'))
        
def test_dataloader():
    train_dataset = HumanDataset(
        instance_data_root="data/densepose/masks",
        instance_prompt="a photo of a coco person",
        class_data_root=None,
        class_prompt=None,
        tokenizer=None,
        size=512,
        center_crop=False,
    )
    for i in range(10):
        a = train_dataset[i]
        pil_image = a['PIL_images']
        mask = a['masks']
        mask, masked_image = prepare_mask_and_masked_image(pil_image, mask)
        pil_image.save('i_{}.png'.format(i))
        Image.fromarray(mask[0][0].bool().numpy()).save('m_{}.png'.format(i))
        Image.fromarray(((masked_image[0].numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)).save('mi_{}.png'.format(i))

if __name__ == "__main__":
    #test_dataloader()
    main()
