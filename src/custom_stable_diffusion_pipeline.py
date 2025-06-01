from diffusers import StableDiffusionPipeline, StableDiffusionPipelineOutput
from diffusers.utils import logging
import torch
from typing import List, Optional, Union, Dict, Any
import numpy as np

logger = logging.get_logger(__name__)

class CustomStableDiffusionPipeline(StableDiffusionPipeline):
    """Custom Stable Diffusion pipeline with support for fault injection and intermediate output saving."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info("Initialized CustomStableDiffusionPipeline")

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
        num_images_per_prompt: int,
        do_classifier_free_guidance: bool,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
    ):
        """
        Encodes input prompts, returning additional negative prompt layer outputs.

        Args:
            prompt (str or List[str]): Input prompt(s) for image generation.
            device (torch.device): Device to perform computation on.
            num_images_per_prompt (int): Number of images to generate per prompt.
            do_classifier_free_guidance (bool): Whether to use classifier-free guidance.
            negative_prompt (str or List[str], optional): Negative prompt(s).
            prompt_embeds (torch.FloatTensor, optional): Pre-computed prompt embeddings.
            negative_prompt_embeds (torch.FloatTensor, optional): Pre-computed negative prompt embeddings.
            lora_scale (float, optional): LoRA scale for fine-tuning.

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
                - Negative prompt layer outputs.
                - Prompt embeddings.
                - Negative prompt embeddings.
        """
        prompt_embeds, negative_prompt_embeds = super().encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
        )
        my_encoder_layer_outputs_nega = negative_prompt_embeds.clone() if negative_prompt_embeds is not None else None
        return my_encoder_layer_outputs_nega, prompt_embeds, negative_prompt_embeds

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
        latents: Optional[torch.FloatTensor] = None,
        user_fixed_latents: bool = False,
    ):
        """
        Prepares latent variables with optional fixed seed.

        Args:
            batch_size (int): Batch size for generation.
            num_channels_latents (int): Number of latent channels.
            height (int): Height of the output image.
            width (int): Width of the output image.
            dtype (torch.dtype): Data type for latents.
            device (torch.device): Device for computation.
            generator (torch.Generator or List[torch.Generator], optional): Random number generator.
            latents (torch.FloatTensor, optional): Pre-computed latents.
            user_fixed_latents (bool): If True, uses a fixed seed (0) for reproducibility.

        Returns:
            torch.FloatTensor: Prepared latent variables.
        """
        if latents is None:
            if user_fixed_latents:
                generator = torch.Generator(device="cpu").manual_seed(0)
            latents = super().prepare_latents(
                batch_size,
                num_channels_latents,
                height,
                width,
                dtype,
                device,
                generator,
                latents,
            )
        return latents

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[callable] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.7,
        user_prompt_idx: Optional[int] = None,
        user_error: bool = False,
        user_error_type: List[str] = None,
        user_save_type: List[str] = None,
        user_fixed_latents: bool = True,
        user_save_iter: List[int] = None,
        user_test_idx: Optional[int] = 0,
        **kwargs,
    ):
        """
        Generates images with support for fault injection and intermediate output saving.

        Args:
            prompt (str or List[str]): Input prompt(s) for generation.
            height (int, optional): Output image height.
            width (int, optional): Output image width.
            num_inference_steps (int): Number of denoising steps.
            guidance_scale (float): Guidance scale for classifier-free guidance.
            negative_prompt (str or List[str], optional): Negative prompt(s).
            user_prompt_idx (int, optional): Prompt index for tracking.
            user_error (bool): If True, simulates an error.
            user_error_type (List[str]): Types of errors to simulate.
            user_save_type (List[str]): Data types to save (e.g., x_t, ut_null).
            user_fixed_latents (bool): If True, uses fixed latents.
            user_save_iter (List[int]): Iteration steps to save intermediate results.
            user_test_idx (int): Test index for error-free cases.
            **kwargs: Additional arguments passed to parent class.

        Returns:
            Tuple[StableDiffusionPipelineOutput, torch.FloatTensor, torch.FloatTensor]:
                - Generated images and NSFW flags.
                - Negative prompt layer outputs.
                - Negative prompt embeddings.
        """
        user_error_type = user_error_type or []
        user_save_type = user_save_type or []
        user_save_iter = user_save_iter or []
        user_test_idx = user_test_idx or 0

        logger.info(f"Calling CustomStableDiffusionPipeline with user_prompt_idx={user_prompt_idx}, "
                    f"user_error={user_error}, user_fixed_latents={user_fixed_latents}, "
                    f"user_save_type={user_save_type}, user_save_iter={user_save_iter}")

        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        my_encoder_layer_outputs_nega, prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        backup_negative_prompt_embeds = negative_prompt_embeds.cpu() if negative_prompt_embeds is not None else None
        backup_my_encoder_layer_outputs_nega = my_encoder_layer_outputs_nega.cpu() if my_encoder_layer_outputs_nega is not None else None

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            user_fixed_latents=user_fixed_latents,
        )

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if 'x_t' in user_save_type and i in user_save_iter:
                    user_folder_name = 'x_t_random' if user_fixed_latents else 'x_t_unfixed'
                    if user_error:
                        user_xt_name = f"P{user_prompt_idx}_x_t_{user_error_type}_iter{i}"
                        user_latents = latents.cpu()
                        torch.save(user_latents, f'./dataset/Unet/{user_folder_name}/{user_xt_name}.pth')
                        logger.info(f"Saved {user_xt_name}.pth")
                    else:
                        user_xt_name = f"P{user_prompt_idx}_x_t_iter{i}_test{user_test_idx}"
                        user_latents = latents.cpu()
                        torch.save(user_latents, f'./dataset/Unet/{user_folder_name}/{user_xt_name}.pth')
                        logger.info(f"Saved {user_xt_name}.pth")

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    if i in user_save_iter:
                        if user_error:
                            if 'ut_null' in user_save_type:
                                user_folder_name = 'noise_pred_uncond' if user_fixed_latents else 'noise_pred_uncond_unfixed'
                                user_noise_pred_uncond_name = f"P{user_prompt_idx}_noise_pred_uncond_{user_error_type}_iter{i}"
                                user_noise_pred_uncond = noise_pred_uncond.cpu()
                                torch.save(user_noise_pred_uncond,
                                           f'./dataset/Unet/{user_folder_name}/{user_noise_pred_uncond_name}.pth')
                                logger.info(f"Saved {user_noise_pred_uncond_name}.pth")
                            if 'ut_text' in user_save_type:
                                user_folder_name = 'noise_pred_text' if user_fixed_latents else 'noise_pred_text_unfixed'
                                user_noise_pred_text_name = f"P{user_prompt_idx}_noise_pred_text_{user_error_type}_iter{i}"
                                user_noise_pred_text = noise_pred_text.cpu()
                                torch.save(user_noise_pred_text,
                                           f'./dataset/Unet/{user_folder_name}/{user_noise_pred_text_name}.pth')
                                logger.info(f"Saved {user_noise_pred_text_name}.pth")
                        else:
                            if 'ut_null' in user_save_type:
                                user_folder_name = 'noise_pred_uncond' if user_fixed_latents else 'noise_pred_uncond_unfixed'
                                user_noise_pred_uncond_name = f"P{user_prompt_idx}_noise_pred_uncond_iter{i}_test{user_test_idx}"
                                user_noise_pred_uncond = noise_pred_uncond.cpu()
                                torch.save(user_noise_pred_uncond,
                                           f'./dataset/Unet/{user_folder_name}/{user_noise_pred_uncond_name}.pth')
                                logger.info(f"Saved {user_noise_pred_uncond_name}.pth")
                            if 'ut_text' in user_save_type:
                                user_folder_name = 'noise_pred_text' if user_fixed_latents else 'noise_pred_text_unfixed'
                                user_noise_pred_text_name = f"P{user_prompt_idx}_noise_pred_text_iter{i}_test{user_test_idx}"
                                user_noise_pred_text = noise_pred_text.cpu()
                                torch.save(user_noise_pred_text,
                                           f'./dataset/Unet/{user_folder_name}/{user_noise_pred_text_name}.pth')
                                logger.info(f"Saved {user_noise_pred_text_name}.pth")

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    from diffusers.utils import rescale_noise_cfg
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                if 'ut' in user_save_type and i in user_save_iter:
                    user_folder_name = 'noise_pred' if user_fixed_latents else 'noise_pred_unfixed'
                    if user_error:
                        user_noise_pred_name = f"P{user_prompt_idx}_noise_pred_{user_error_type}_iter{i}"
                        user_noise_pred = noise_pred.cpu()
                        torch.save(user_noise_pred,
                                   f'./dataset/Unet/{user_folder_name}/{user_noise_pred_name}.pth')
                        logger.info(f"Saved {user_noise_pred_name}.pth")
                    else:
                        user_noise_pred_name = f"P{user_prompt_idx}_noise_pred_iter{i}_test{user_test_idx}"
                        user_noise_pred = noise_pred.cpu()
                        torch.save(user_noise_pred,
                                   f'./dataset/Unet/{user_folder_name}/{user_noise_pred_name}.pth')
                        logger.info(f"Saved {user_noise_pred_name}.pth")

                latents = self.scheduler.step(
                    noise_pred,
                    t,
                    latents,
                    **extra_step_kwargs,
                    return_dict=False,
                    user_prompt_idx=user_prompt_idx,
                    user_error=user_error,
                    user_error_type=user_error_type,
                    user_save_type=user_save_type,
                    user_iter=i,
                )[0]

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if 'xf' in user_save_type:
            if user_error:
                user_xf_name = f"P{user_prompt_idx}_x_f_{user_error_type}"
                user_latents = latents.cpu()
                torch.save(user_latents, f'./dataset/Unet/x_t/{user_xf_name}.pth')
                logger.info(f"Saved {user_xf_name}.pth")
            else:
                user_xf_name = f"P{user_prompt_idx}_x_f"
                user_latents = latents.cpu()
                torch.save(user_latents, f'./dataset/Unet/x_t/{user_xf_name}.pth')
                logger.info(f"Saved {user_xf_name}.pth")

        if output_type != "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        self.maybe_free_model_hooks()

        if not return_dict:
            return image, has_nsfw_concept, backup_my_encoder_layer_outputs_nega, backup_negative_prompt_embeds

        return (
            StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept),
            backup_my_encoder_layer_outputs_nega,
            backup_negative_prompt_embeds,
        )

def register_custom_pipeline():
    """Registers the custom pipeline with diffusers."""
    from diffusers import DiffusionPipeline
    DiffusionPipeline.register_pipeline(
        "custom/stable-diffusion",
        pipeline_class=CustomStableDiffusionPipeline,
        default_components=StableDiffusionPipeline._get_default_pipeline_components(),
    )

if __name__ == "__main__":
    register_custom_pipeline()
    logger.info("CustomStableDiffusionPipeline registered")