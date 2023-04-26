from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


model = create_model('./models/step_2_train.yaml').cpu()
model.load_state_dict(load_state_dict('./models/step_2_trained.pth', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

def process(control, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    with torch.no_grad():
        #control = resize_image(control, image_resolution)
        H, W, C = control.shape

        # detected_map = apply_canny(img, low_threshold, high_threshold)
        # detected_map = HWC3(detected_map)

        # print(img.shape)
        # print(detected_map.shape)

        # control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = control.cuda()
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return results

# set basic parameters
strength = 1.0 #label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01
guess_mode = False #gr.Checkbox(label='Guess Mode', value=False)

ddim_steps = 50 #gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
scale = 9.0 #gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
seed = 1234 # gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
eta = 0.0 # gr.Number(label="eta (DDIM)", value=0.0)
a_prompt = 'best quality, extremely detailed'#gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality' #gr.Textbox(label="Negative Prompt",value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
num_samples = 1 #gr.Slider(label="Number of Samples", minimum=1, maximum=10, value=1, step=1)
resolution = 512 

from step_2_dataset import MyDataset

dataset = MyDataset()
for i in range(5):
    item = dataset[i]
    prompt = item['txt']
    control = item['hint']

    sample_result = process(control, prompt, a_prompt, n_prompt, num_samples, resolution, ddim_steps, guess_mode, strength, scale, seed, eta)[0]
    #sample_result = cv2.cvtColor(sample_result, cv2.COLOR_BGR2RGB)
    print(sample_result.shape)
    cv2.imshow('sample_result', sample_result)
    cv2.waitKey(0)
    cv2.imwrite('sample_result_'+str(i)+'.jpg', sample_result)