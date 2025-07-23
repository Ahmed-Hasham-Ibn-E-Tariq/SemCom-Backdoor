import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset, random_split
from tqdm.auto import tqdm
import torch
import argparse
import os
from tqdm import tqdm
from torchvision import transforms
from torchvision import datasets, transforms, models
import time
import sys
import random
from colorama import Fore
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from torchvision import transforms as tfms
from .ddim import sample, invert
# 修改以下导入语句
from .channel import Channel, get_channel
from transformers import pipeline
from .dataset import ImageDataset
import pyiqa
import csv

def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Communication with Stable Diffusion')
    parser.add_argument("--trigger", type=str, default=None, help="path to trigger .npy")
    
    # Dataset parameters
    parser.add_argument('--dataset_dir', type=str, default='/home/ubuntu/code/SeCom/SD_SemCom/kodak_images/', 
                        help='Directory containing test images')
    
    # Model parameters
    parser.add_argument('--model_id', type=str, default='runwayml/stable-diffusion-v1-5', 
                        help='Model ID for Stable Diffusion')
    parser.add_argument('--captioner', action='store_true', 
                            help='Enable image captioning (default: False)')
    parser.add_argument('--captioner_model', type=str, default='Salesforce/blip-image-captioning-large',
                        help='Model for image captioning')
    
    # Communication parameters
    parser.add_argument('--snr_db', type=float, nargs='+', default=[5],
                        help='List of SNR values in dB')

    # 添加信道类型参数
    parser.add_argument('--channel_type', type=str, default='AWGN',
                        help='Channel type: AWGN or Rayleigh')
    
    # Diffusion parameters
    parser.add_argument('--T1', type=int, default=0,
                        help='Forward Step at Tx')
    parser.add_argument('--T2', type=int, default=10,
                        help='Forward Step at Rx')
    parser.add_argument('--T3', type=int, nargs='+', default=[15],
                        help='Backward Step at Rx')
    parser.add_argument('--total_steps', type=int, default=50,
                        help='Number of total sampling steps')
    parser.add_argument('--guidance_scale', type=float, default=6.0,
                        help='Guidance scale for classifier-free guidance')
    
    # Device parameters
    parser.add_argument('--device_id', type=int, default=1,
                        help='CUDA device ID')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./outputs_jyy/imgs/',
                        help='Directory to save output images')
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Setup device
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load models
    pipe = StableDiffusionPipeline.from_pretrained(args.model_id).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    if args.captioner:
        captioner = pipeline("image-to-text", model=args.captioner_model, device=device)
        
    
    # Initialize quality metrics
    psnr_metric = pyiqa.create_metric('psnr', device=device)
    lpips_metric = pyiqa.create_metric('lpips', device=device)
    ssim_metric = pyiqa.create_metric('ms_ssim', device=device)
    
    # Setup transformation
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Process each SNR value
    for i, snr_db in enumerate(args.snr_db):
        noise_channel = Channel(args.channel_type.lower()).to(device)
        output_dir = os.path.join(args.output_dir, f"T1_{args.T1}_T2_{args.T2}_T3_{args.T3}_{args.channel_type}_{snr_db}db")
        os.makedirs(output_dir, exist_ok=True)
        
        # Get the corresponding start step for reconstruction
        T3 = args.T3[i] if i < len(args.T3) else args.T3[-1]
        
        # Metrics storage
        metrics = {
            'psnr': [],
            'lpips': [],
            'ssim': []
        }
        
        # Process each image in the dataset
        for img_name in os.listdir(args.dataset_dir):
            if not (img_name.endswith('.jpg') or img_name.endswith('.png')):
                continue
                
            # Load the image
            input_image = Image.open(os.path.join(args.dataset_dir, img_name)).convert("RGB")
            
            # Apply transformations
            test_image = transform(input_image).to(device)
            
            # Get image caption
            text_prompt = "High quality, High res, 8k"
            if args.captioner:
                result = captioner(input_image)
                text_prompt += result[0]['generated_text']
            
            print(f"Processing image: {img_name}")
            print(f"Caption: {text_prompt}")
            
            # Encode image to latent space
            with torch.no_grad():
                latent = pipe.vae.encode(test_image.unsqueeze(0) * 2 - 1)
                latents = 0.18215 * latent.latent_dist.sample()  # Scale latent variables
                
                # 自定义一个封装函数来适配新的 Channel 类接口
                def apply_channel(x):
                    # 为每个样本准备相同的 SNR 参数
                    chan_param = torch.tensor([snr_db]).to(device)
                    # 应用信道并返回第一个输出（信道输出）
                    output, _, _ = noise_channel(x, chan_param)
                    return output
                if args.trigger:
                    trig = torch.from_numpy(np.load(args.trigger)).to(device)

    # back‑door trigger hook
    # -------- back‑door trigger hook --------
                    def inject(_, inp):
                        x = inp[0]                                # latent  B × 4 × H × W
                        trg = trig
        # resize trigger if latent H/W differ
                        if trg.shape[-2:] != x.shape[-2:]:
                            trg = torch.nn.functional.interpolate(
                                trg.unsqueeze(0),
                                size=x.shape[-2:],                # latent (H, W)
                                mode="bilinear",
                                align_corners=False
                            ).squeeze(0)
                        return (x + trg,) + inp[1:]

                    unet = pipe.unet if 'pipe' in locals() else sampler.model
                    unet.register_forward_pre_hook(inject)
                    print("✓ Trigger hook registered")

    # -------- end trigger hook --------


                # VAE
                if args.T1==0 and args.T2==0 and T3==0:
                    # Apply noise channel to latents
                    latents = apply_channel(latents)
                    # VAE Decode
                    #latents = latents / 0.18215
                    sampled_images = pipe.decode_latents(latents)
                    sampled_images=pipe.numpy_to_pil(sampled_images)
                    
                elif args.T1==0 and args.T2>0 and T3>=0:
                    # Apply noise channel to latents
                    inverted_latents = invert(pipe, latents, text_prompt,
                                            num_inference_steps=args.total_steps, 
                                            T1=args.T1,
                                            T2=args.T2,
                                            noise_channel=apply_channel,
                                            device=device)
                    # Get specific step latents
                    start_latents = inverted_latents[-1][None]
                    # Generate images from the noisy latents
                    sampled_images = sample(pipe, text_prompt,
                                       start_step=args.total_steps-T3,
                                       start_latents=start_latents, 
                                       guidance_scale=args.guidance_scale, 
                                       num_inference_steps=args.total_steps,
                                       device=device)

                    
                elif args.T1>0 and args.T2>0 and T3>0:
                    inverted_latents = invert(pipe, latents, text_prompt, 
                                            num_inference_steps=args.total_steps, 
                                            T1=args.T1,
                                            T2=args.T2,
                                            noise_channel=apply_channel,
                                            device=device)
                
                    # Get specific step latents
                    start_latents = inverted_latents[-1][None]
                    # Generate images from the noisy latents
                    sampled_images = sample(pipe, text_prompt, 
                                       start_step=args.total_steps-T3,
                                       start_latents=start_latents, 
                                       guidance_scale=args.guidance_scale, 
                                       num_inference_steps=args.total_steps,
                                       device=device)
                    
                elif args.T1>0 and args.T2==0 and T3>0:
                    inverted_latents = invert(pipe, latents, text_prompt, 
                                            num_inference_steps=args.total_steps, 
                                            T1=args.T1,
                                            T2=args.T2,
                                            noise_channel=apply_channel,
                                            device=device)
                    # Get specific step latents
                    start_latents = inverted_latents[-1][None]
                    # channel
                    start_latents = apply_channel(start_latents)
                    # Generate images from the noisy latents
                    sampled_images = sample(pipe, text_prompt, 
                                       start_step=args.total_steps-T3,
                                       start_latents=start_latents, 
                                       guidance_scale=args.guidance_scale, 
                                       num_inference_steps=args.total_steps,
                                       device=device)
                # Save the generated image
                output_path = os.path.join(output_dir, img_name)
                sampled_images[0].save(output_path)
                
        #         # Calculate metrics
        #         input_tensor = transform(input_image).unsqueeze(0).to(device)
        #         output_tensor = transform(sampled_images[0]).unsqueeze(0).to(device)
                
        #         psnr_value = psnr_metric(input_tensor, output_tensor).item()
        #         lpips_value = lpips_metric(input_tensor, output_tensor).item()
        #         ssim_value = ssim_metric(input_tensor, output_tensor).item()
                
        #         metrics['psnr'].append(psnr_value)
        #         metrics['lpips'].append(lpips_value)
        #         metrics['ssim'].append(ssim_value)
                
        #         print(f"PSNR: {psnr_value:.2f}, LPIPS: {lpips_value:.4f}, SSIM: {ssim_value:.4f}")
        
        # # Calculate and save average metrics
        # avg_metrics = {k: sum(v)/len(v) for k, v in metrics.items()}
        # print(f"\nAverage metrics for {args.channel_type} SNR {snr_db}dB:")
        # print(f"PSNR: {avg_metrics['psnr']:.2f}, LPIPS: {avg_metrics['lpips']:.4f}, SSIM: {avg_metrics['ssim']:.4f}")
        
        # 构造保存路径
        # csv_file = os.path.join('./outputs_jyy/', f'snr_{snr_db}db.csv')

# # 检查文件是否存在，如果不存在就写入表头
#         write_header = not os.path.exists(csv_file)

#         with open(csv_file, mode='a', newline='') as f:
#             writer = csv.writer(f)
    
#             if write_header:
#                 writer.writerow(['T1', 'T2', 'T3', 'PSNR', 'LPIPS', 'SSIM'])  # 写入表头

#     # 假设 T1, T2, T3 是你当前运行的三个变量
#             writer.writerow([args.T1, args.T2, args.T3,
#                             avg_metrics['psnr'],
#                             avg_metrics['lpips'],
#                             avg_metrics['ssim']])

if __name__ == "__main__":
    main()
