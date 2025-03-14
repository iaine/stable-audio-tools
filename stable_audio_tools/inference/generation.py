import numpy as np
import torch 
import typing as tp
import math 
from torchaudio import transforms as T

import traceback

from .utils import prepare_audio
from .sampling import sample, sample_k, sample_rf
from ..data.utils import PadCrop

import json
from datetime import datetime

def generate_diffusion_uncond(
        model,
        steps: int = 250,
        batch_size: int = 1,
        sample_size: int = 2097152,
        seed: int = -1,
        device: str = "cuda",
        init_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        init_noise_level: float = 1.0,
        return_latents = False,
        **sampler_kwargs
        ) -> torch.Tensor:
    
    # The length of the output in audio samples 
    audio_sample_size = sample_size

    # If this is latent diffusion, change sample_size instead to the downsampled latent size
    if model.pretransform is not None:
        sample_size = sample_size // model.pretransform.downsampling_ratio
    trace = {}
    trace["initseed"]={"data":seed, "time": datetime.now()}       
    # Seed
    # The user can explicitly set the seed to deterministically generate the same output. Otherwise, use a random seed.
    seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1, dtype=np.uint32)
    trace["seed"]={"data":seed, "time": datetime.now()}
    torch.manual_seed(seed)
    # Define the initial noise immediately after setting the seed
    noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)
    trace["initnoise"]={"data":noise, "time": datetime.now()}
    if init_audio is not None:
        # The user supplied some initial audio (for inpainting or variation). Let us prepare the input audio.
        in_sr, init_audio = init_audio

        io_channels = model.io_channels

        # For latent models, set the io_channels to the autoencoder's io_channels
        if model.pretransform is not None:
            io_channels = model.pretransform.io_channels

        # Prepare the initial audio for use by the model
        init_audio = prepare_audio(init_audio, in_sr=in_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)

        # For latent models, encode the initial audio into latents
        if model.pretransform is not None:
            init_audio = model.pretransform.encode(init_audio)

        init_audio = init_audio.repeat(batch_size, 1, 1)
    else:
        # The user did not supply any initial audio for inpainting or variation. Generate new output from scratch. 
        init_audio = None
        init_noise_level = None

    # Inpainting mask

    if init_audio is not None:
        # variations
        sampler_kwargs["sigma_max"] = init_noise_level
        mask = None 
    else:
        mask = None
    trace["mask"]={"data":mask, "time": datetime.now()}
    # Now the generative AI part:

    diff_objective = model.diffusion_objective

    if diff_objective == "v":    
        # k-diffusion denoising process go!
        sampled = sample_k(model.model, noise, init_audio, mask, steps, **sampler_kwargs, device=device)
    elif diff_objective == "rectified_flow":
        sampled = sample_rf(model.model, noise, init_data=init_audio, steps=steps, **sampler_kwargs, device=device)
    trace["diffusion_objective"]={"data":diff_objective, "time": datetime.now()}
    trace["sampled"]={"data":sampled, "time": datetime.now()}

    # Denoising process done. 
    # If this is latent diffusion, decode latents back into audio
    if model.pretransform is not None and not return_latents:
        sampled = model.pretransform.decode(sampled)

    # Return audio
    return sampled


def generate_diffusion_cond(
        model,
        steps: int = 250,
        cfg_scale=6,
        conditioning: dict = None,
        conditioning_tensors: tp.Optional[dict] = None,
        negative_conditioning: dict = None,
        negative_conditioning_tensors: tp.Optional[dict] = None,
        batch_size: int = 1,
        sample_size: int = 2097152,
        sample_rate: int = 48000,
        seed: int = -1,
        device: str = "cuda",
        init_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        init_noise_level: float = 1.0,
        mask_args: dict = None,
        return_latents = False,
        **sampler_kwargs
        ) -> torch.Tensor: 
    """
    Generate audio from a prompt using a diffusion model.
    
    Args:
        model: The diffusion model to use for generation.
        steps: The number of diffusion steps to use.
        cfg_scale: Classifier-free guidance scale 
        conditioning: A dictionary of conditioning parameters to use for generation.
        conditioning_tensors: A dictionary of precomputed conditioning tensors to use for generation.
        batch_size: The batch size to use for generation.
        sample_size: The length of the audio to generate, in samples.
        sample_rate: The sample rate of the audio to generate (Deprecated, now pulled from the model directly)
        seed: The random seed to use for generation, or -1 to use a random seed.
        device: The device to use for generation.
        init_audio: A tuple of (sample_rate, audio) to use as the initial audio for generation.
        init_noise_level: The noise level to use when generating from an initial audio sample.
        return_latents: Whether to return the latents used for generation instead of the decoded audio.
        **sampler_kwargs: Additional keyword arguments to pass to the sampler.    
    """
    software = {}
    keys = []
    trace = {}
    trace["initseed"]={"data":seed, "time": datetime.timestamp()} 
    trace["cfg_scale"]={"data":cfg_scale, "time": datetime.timestamp()}
    trace["diffusion_steps"]={"data":steps, "time": datetime.timestamp()}
    trace["model"]={"data":model, "time": datetime.timestamp()}
    
    
    #print("Inside {} {} {} {}".format(filename, line, procname, text))
    # The length of the output in audio samples 
    stack = traceback.extract_stack()
    (filename, line, procname, text) = stack[-1]
    audio_sample_size = sample_size
    keys.append({"name": "audio_sample_size", "fname": filename, "line": line, "text": text, "keys": "", "types": type(audio_sample_size)})
    trace["initsample_size"]={"data":audio_sample_size, "time": datetime.timestamp()} 

    # If this is latent diffusion, change sample_size instead to the downsampled latent size
    if model.pretransform is not None:
        sample_size = sample_size // model.pretransform.downsampling_ratio
    
    # Seed
    # The user can explicitly set the seed to deterministically generate the same output. Otherwise, use a random seed.
    seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1, dtype=np.uint32)
    trace["seed"]={"data":seed, "time": datetime.timestamp()}

    torch.manual_seed(seed)
    stack = traceback.extract_stack()
    (filename, line, procname, text) = stack[-1]
    keys.append({"name": "seed", "fname": filename, "line": line, "text": text, "keys": "", "types": type(seed)})
    trace["batch_size"]={"data":batch_size, "time": datetime.timestamp()}
    trace["io_channels"]={"data":model.io_channels, "time": datetime.timestamp()}
    trace["sample_size"]={"data":sample_size, "time": datetime.timestamp()}
    
    # Define the initial noise immediately after setting the seed
    noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)
    trace["noise"]={"data":noise, "time": datetime.timestamp()}
    stack = traceback.extract_stack()
    (filename, line, procname, text) = stack[-1]
    keys.append({"name": "noise", "fname": filename, "line": line, "text": text, "keys": "", "types": type(noise)})
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False

    # Conditioning
    assert conditioning is not None or conditioning_tensors is not None, "Must provide either conditioning or conditioning_tensors"
    if conditioning_tensors is None:
        conditioning_tensors = model.conditioner(conditioning, device)
    trace["conditioning_tensors"]={"data":conditioning_tensors, "time": datetime.timestamp()}
    stack = traceback.extract_stack()
    (filename, line, procname, text) = stack[-1]
    keys.append({"name": "conditioning_tensors", "fname": filename, "line": line, "text": text, "keys": [conditioning_tensors.keys()], "types": [type(k) for k in conditioning_tensors.keys() ]})

    conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)
    trace["conditioning_inputs"]={"data":conditioning_inputs, "time": datetime.timestamp()}
    stack = traceback.extract_stack()
    (filename, line, procname, text) = stack[-1]
    keys.append({"name": "conditioning_tensors", "fname": filename, "line": line, "text": text, "keys": [conditioning_inputs.keys()], "types": [type(k) for k in conditioning_inputs.keys() ]})
    if negative_conditioning is not None or negative_conditioning_tensors is not None:
        
        if negative_conditioning_tensors is None:
            negative_conditioning_tensors = model.conditioner(negative_conditioning, device)
            
        negative_conditioning_tensors = model.get_conditioning_inputs(negative_conditioning_tensors, negative=True)
    else:
        negative_conditioning_tensors = {}
    trace["negativeconditioning_tensors"]={"data":negative_conditioning_tensors, "time": datetime.timestamp()}
    stack = traceback.extract_stack()
    (filename, line, procname, text) = stack[-1]
    keys.append({"name": "negative_conditioning", "fname": filename, "line": line, "text": text, "keys": [negative_conditioning.keys()], "types": [type(k) for k in negative_conditioning.keys() ]})
    stack = traceback.extract_stack()
    (filename, line, procname, text) = stack[-1]
    keys.append({"name": "negative_conditioning_tensors", "fname": filename, "line": line, "text": text, "keys": [negative_conditioning_tensors.keys()], "types": [type(k) for k in negative_conditioning_tensors.keys() ]})

    if init_audio is not None:
        # The user supplied some initial audio (for inpainting or variation). Let us prepare the input audio.
        in_sr, init_audio = init_audio

        io_channels = model.io_channels

        # For latent models, set the io_channels to the autoencoder's io_channels
        if model.pretransform is not None:
            io_channels = model.pretransform.io_channels

        # Prepare the initial audio for use by the model
        init_audio = prepare_audio(init_audio, in_sr=in_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)

        # For latent models, encode the initial audio into latents
        if model.pretransform is not None:
            init_audio = model.pretransform.encode(init_audio)

        init_audio = init_audio.repeat(batch_size, 1, 1)
    else:
        # The user did not supply any initial audio for inpainting or variation. Generate new output from scratch. 
        init_audio = None
        init_noise_level = None
        mask_args = None
    
    # Inpainting mask
    if init_audio is not None and mask_args is not None:
        # Cut and paste init_audio according to cropfrom, pastefrom, pasteto
        # This is helpful for forward and reverse outpainting
        cropfrom = math.floor(mask_args["cropfrom"]/100.0 * sample_size)
        pastefrom = math.floor(mask_args["pastefrom"]/100.0 * sample_size)
        pasteto = math.ceil(mask_args["pasteto"]/100.0 * sample_size)
        assert pastefrom < pasteto, "Paste From should be less than Paste To"
        croplen = pasteto - pastefrom
        if cropfrom + croplen > sample_size:
            croplen = sample_size - cropfrom 
        cropto = cropfrom + croplen
        pasteto = pastefrom + croplen
        cutpaste = init_audio.new_zeros(init_audio.shape)
        cutpaste[:, :, pastefrom:pasteto] = init_audio[:,:,cropfrom:cropto]
        #print(cropfrom, cropto, pastefrom, pasteto)
        init_audio = cutpaste
        # Build a soft mask (list of floats 0 to 1, the size of the latent) from the given args
        mask = build_mask(sample_size, mask_args)
        mask = mask.to(device)
    elif init_audio is not None and mask_args is None:
        # variations
        sampler_kwargs["sigma_max"] = init_noise_level
        mask = None 
    else:
        mask = None

    model_dtype = next(model.model.parameters()).dtype

    trace["model_dtype"]={"data":model_dtype, "time": datetime.timestamp()}
    stack = traceback.extract_stack()
    (filename, line, procname, text) = stack[-1]
    keys.append({"name": "model_dtype", "fname": filename, "line": line, "text": text, "keys": "", "types": type[model_dtype]})
 
    noise = noise.type(model_dtype)

    trace["noise"]={"data":noise, "time": datetime.timestamp()}
    stack = traceback.extract_stack()
    (filename, line, procname, text) = stack[-1]
    keys.append({"name": "noise", "fname": filename, "line": line, "text": text, "keys": "", "types": type[noise]})
    conditioning_inputs = {k: v.type(model_dtype) if v is not None else v for k, v in conditioning_inputs.items()}
    trace["conditioning_inputs1"]={"data":conditioning_inputs, "time": datetime.timestamp()}

    # Now the generative AI part:
    # k-diffusion denoising process go!

    diff_objective = model.diffusion_objective
    trace["diff_objective"]={"data":diff_objective, "time": datetime.timestamp()}
    stack = traceback.extract_stack()
    (filename, line, procname, text) = stack[-1]
    keys.append({"name": "diff_objective", "fname": filename, "line": line, "text": text, "keys": "", "types": type[diff_objective]})

    if diff_objective == "v":    
        # k-diffusion denoising process go!
        sampled = sample_k(model.model, noise, init_audio, mask, steps, **sampler_kwargs, **conditioning_inputs, **negative_conditioning_tensors, cfg_scale=cfg_scale, batch_cfg=True, rescale_cfg=True, device=device)
    elif diff_objective == "rectified_flow":

        if "sigma_min" in sampler_kwargs:
            del sampler_kwargs["sigma_min"]

        if "sampler_type" in sampler_kwargs:
            del sampler_kwargs["sampler_type"]

        sampled = sample_rf(model.model, noise, init_data=init_audio, steps=steps, **sampler_kwargs, **conditioning_inputs, **negative_conditioning_tensors, cfg_scale=cfg_scale, batch_cfg=True, rescale_cfg=True, device=device)

    trace["sampled"]={"data":sampled, "time": datetime.timestamp()}
    stack = traceback.extract_stack()
    (filename, line, procname, text) = stack[-1]
    keys.append({"name": "sampled", "fname": filename, "line": line, "text": text, "keys": "", "types": type(sampled)})

    # v-diffusion: 
    #sampled = sample(model.model, noise, steps, 0, **conditioning_tensors, embedding_scale=cfg_scale)
    del noise
    del conditioning_tensors
    del conditioning_inputs
    torch.cuda.empty_cache()
    # Denoising process done. 
    # If this is latent diffusion, decode latents back into audio
    if model.pretransform is not None and not return_latents:
        #cast sampled latents to pretransform dtype
        sampled = sampled.to(next(model.pretransform.parameters()).dtype)
        sampled = model.pretransform.decode(sampled)

    trace["sampled2"]={"data":sampled, "time": datetime.timestamp()}
    stack = traceback.extract_stack()
    (filename, line, procname, text) = stack[-1]
    keys.append({"name": "sampled", "fname": filename, "line": line, "text": text, "keys": sampled, "types": sampled})
    print("keys")
    k = json.dumps(keys)
    print(k)
    with open("iainemsley/keys.json", "w") as fh: fh.write(k)
    print("tracing")
    t = json.dumps(trace)
    with open("iainemsley/trace.json", "w") as fh: fh.write(t)
    print(json.dumps(t))
    # Return audio
    return sampled

# builds a softmask given the parameters
# returns array of values 0 to 1, size sample_size, where 0 means noise / fresh generation, 1 means keep the input audio, 
# and anything between is a mixture of old/new
# ideally 0.5 is half/half mixture but i haven't figured this out yet
def build_mask(sample_size, mask_args):
    maskstart = math.floor(mask_args["maskstart"]/100.0 * sample_size)
    maskend = math.ceil(mask_args["maskend"]/100.0 * sample_size)
    softnessL = round(mask_args["softnessL"]/100.0 * sample_size)
    softnessR = round(mask_args["softnessR"]/100.0 * sample_size)
    marination = mask_args["marination"]
    # use hann windows for softening the transition (i don't know if this is correct)
    hannL = torch.hann_window(softnessL*2, periodic=False)[:softnessL]
    hannR = torch.hann_window(softnessR*2, periodic=False)[softnessR:]
    # build the mask. 
    mask = torch.zeros((sample_size))
    mask[maskstart:maskend] = 1
    mask[maskstart:maskstart+softnessL] = hannL
    mask[maskend-softnessR:maskend] = hannR
    # marination finishes the inpainting early in the denoising schedule, and lets audio get changed in the final rounds
    if marination > 0:        
        mask = mask * (1-marination) 
    #print(mask)
    return mask
