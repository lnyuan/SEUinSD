from typing import Iterable, List, Optional
import numpy as np
import torch
from .utils import append_error_type_to_dict, create_image_folder

# Note: `injectorbit` and `injector0` are assumed to be external modules.
# Users must provide their implementations or replace with equivalent functionality.
import injectorbit as injector
import injector0

def error_free(
    pipe,
    test_file: np.ndarray,
    save_type: List[str],
    save_iter: List[int],
    fixed_latents: bool = True,
    test_times: int = 1,
    prompt_num: int = 5,
):
    """
    Generates images without fault injection to establish a baseline.

    Args:
        pipe: CustomStableDiffusionPipeline instance.
        test_file (np.ndarray): Array of prompts.
        save_type (List[str]): Types of intermediate outputs to save.
        save_iter (List[int]): Iteration steps to save outputs.
        fixed_latents (bool): If True, uses fixed latents.
        test_times (int): Number of test repetitions.
        prompt_num (int): Number of prompts to process.
    """
    for prompt_idx in range(prompt_num):
        for i in range(test_times):
            prompt = test_file[prompt_idx]
            image, negative_prompt_embeds_layer, negative_prompt_embeds = pipe(
                prompt,
                user_prompt_idx=prompt_idx,
                user_error=False,
                user_error_type=None,
                user_save_type=save_type,
                user_fixed_latents=fixed_latents,
                user_save_iter=save_iter,
                user_test_idx=i,
            )
            image = image.images[0]
            # Note: `Clip` module is not provided. Users must implement `calculate_clip_score`.
            clip_score = 0.0  # Placeholder
            print(f"Clip_score = {clip_score}")
            append_error_type_to_dict(
                f'P{prompt_idx}_test{i}',
                clip_score,
                filename=f'./error_detect/error_free_clipsore_dict.json' if fixed_latents else f'./error_detect/unfixed_error_free_clipsore_dict.json'
            )

def inj_Unet_d_attn(
    pipe,
    test_file: np.ndarray,
    save_type: List[str],
    save_iter: List[int],
    fixed_latents: bool = True,
    blockname: str = "down_blocks",
    partname: str = "attentions",
    blockseq: List[int] = [0, 1, 2, 3],
    layerseq: List[int] = [0, 1],
    bitseq: List[int] = [1],
    weight_num: int = 50,
    weight_num_save: int = 10,
    param_type: List[str] = ["attn1_v", "attn1_o", "attn2_v", "attn2_o", "fc1", "fc2"],
    prompt_num: Iterable[int] = range(5),
    proportion: Optional[float] = None,
):
    """
    Injects faults into UNet down_blocks attentions.

    Args:
        pipe: CustomStableDiffusionPipeline instance.
        test_file (np.ndarray): Array of prompts.
        save_type (List[str]): Types of intermediate outputs to save.
        save_iter (List[int]): Iteration steps to save outputs.
        fixed_latents (bool): If True, uses fixed latents.
        blockname (str): Name of the block (e.g., 'down_blocks').
        partname (str): Name of the part (e.g., 'attentions').
        blockseq (List[int]): List of block indices.
        layerseq (List[int]): List of layer indices.
        bitseq (List[int]): List of bit positions to inject faults.
        weight_num (int): Number of weights to select.
        weight_num_save (int): Number of weights to save.
        param_type (List[str]): Types of parameters to inject faults into.
        prompt_num (Iterable[int]): Indices of prompts to process.
        proportion (float, optional): Proportion of weights to inject faults into.
    """
    for block_idx in blockseq:
        for layer_idx in layerseq:
            for param_name in param_type:
                for bit in bitseq:
                    imagefolder_path = f"./Images_error/Unet/down_blocks/{block_idx}/attentions/{layer_idx}/transformer_blocks/{param_name}/bit{bit}"
                    create_image_folder(imagefolder_path)

    for block_idx in blockseq:
        for layer_idx in layerseq:
            for param_name in param_type:
                matrix_type = param_name
                if matrix_type == "attn1_v":
                    matrix = pipe.unet.down_blocks[block_idx].attentions[layer_idx].transformer_blocks[0].attn1.to_v.weight.storage()
                elif matrix_type == "attn2_v":
                    matrix = pipe.unet.down_blocks[block_idx].attentions[layer_idx].transformer_blocks[0].attn2.to_v.weight.storage()
                elif matrix_type == "fc1":
                    matrix = pipe.unet.down_blocks[block_idx].attentions[layer_idx].transformer_blocks[0].ff.net[0].proj.weight.storage()
                else:
                    continue  # Skip unsupported param types
                size = len(matrix)
                weight_position = np.linspace(0, size - 1, weight_num, dtype=int).tolist()
                weight_position_save = weight_position[::len(weight_position) // weight_num_save]

                for bit in bitseq:
                    bit_seed = bit
                    for weight in weight_position_save:
                        print(f"Current UD{block_idx}A{layer_idx}{param_name}Bit{bit}W{weight}")
                        injection_number_total = int(size * proportion) if proportion else 1
                        count, inj_position, origin_value = injector0.injector_bit(matrix, size, injection_number_total, bit_seed, weight, "float16")

                        for prompt_idx in prompt_num:
                            prompt = test_file[prompt_idx]
                            error_type = f"D{block_idx}A{layer_idx}{param_name}Bit{bit}W{weight}"
                            image, negative_prompt_embeds_layer, negative_prompt_embeds = pipe(
                                prompt,
                                user_prompt_idx=prompt_idx,
                                user_error=True,
                                user_error_type=error_type,
                                user_save_type=save_type,
                                user_fixed_latents=fixed_latents,
                                user_save_iter=save_iter,
                            )
                            image = image.images[0]
                            clip_score = 0.0  # Placeholder for Clip.calculate_clip_score
                            append_error_type_to_dict(
                                f'P{prompt_idx}_{error_type}',
                                clip_score,
                                filename=f'./error_detect/error_clipsore_dict_{blockname}_{partname}.json' if fixed_latents else f'./error_detect/unfixed_error_clipsore_dict_{blockname}_{partname}.json'
                            )
                        injector0.restore(matrix, inj_position, origin_value)
    print(f"Injected: {blockname}'s {partname}, block:{blockseq}, layer:{layerseq}, bit:{bitseq}, param:{param_type}, weightnum:{weight_num}, prompt:{prompt_num}.\n{save_type} saved")

def inj_Unet_u_attn(
    pipe,
    test_file: np.ndarray,
    save_type: List[str],
    save_iter: List[int],
    fixed_latents: bool = True,
    blockname: str = "up_blocks",
    partname: str = "attentions",
    blockseq: List[int] = [1, 2, 3],
    layerseq: List[int] = [0, 1],
    bitseq: List[int] = [1],
    weight_num: int = 50,
    weight_num_save: int = 10,
    param_type: List[str] = ["attn1_v", "attn1_o", "attn2_v", "attn2_o", "fc1", "fc2"],
    prompt_num: Iterable[int] = range(5),
    proportion: Optional[float] = None,
):
    """
    Injects faults into UNet up_blocks attentions.

    Args:
        pipe: CustomStableDiffusionPipeline instance.
        test_file (np.ndarray): Array of prompts.
        save_type (List[str]): Types of intermediate outputs to save.
        save_iter (List[int]): Iteration steps to save outputs.
        fixed_latents (bool): If True, uses fixed latents.
        blockname (str): Name of the block (e.g., 'up_blocks').
        partname (str): Name of the part (e.g., 'attentions').
        blockseq (List[int]): List of block indices.
        layerseq (List[int]): List of layer indices.
        bitseq (List[int]): List of bit positions to inject faults.
        weight_num (int): Number of weights to select.
        weight_num_save (int): Number of weights to save.
        param_type (List[str]): Types of parameters to inject faults into.
        prompt_num (Iterable[int]): Indices of prompts to process.
        proportion (float, optional): Proportion of weights to inject faults into.
    """
    for block_idx in blockseq:
        for layer_idx in layerseq:
            for param_name in param_type:
                for bit in bitseq:
                    imagefolder_path = f"./Images_error/Unet/up_blocks/{block_idx}/attentions/{layer_idx}/transformer_blocks/{param_name}/bit{bit}"
                    create_image_folder(imagefolder_path)

    for block_idx in blockseq:
        for layer_idx in layerseq:
            for param_name in param_type:
                matrix_type = param_name
                if matrix_type == "attn1_v":
                    matrix = pipe.unet.up_blocks[block_idx].attentions[layer_idx].transformer_blocks[0].attn1.to_v.weight.storage()
                elif matrix_type == "attn2_v":
                    matrix = pipe.unet.up_blocks[block_idx].attentions[layer_idx].transformer_blocks[0].attn2.to_v.weight.storage()
                elif matrix_type == "fc1":
                    matrix = pipe.unet.up_blocks[block_idx].attentions[layer_idx].transformer_blocks[0].ff.net[0].proj.weight.storage()
                else:
                    continue
                size = len(matrix)
                weight_position = np.linspace(0, size - 1, weight_num, dtype=int).tolist()
                weight_position_save = weight_position[::len(weight_position) // weight_num_save]

                for bit in bitseq:
                    bit_seed = bit
                    for weight in weight_position_save:
                        print(f"Current UU{block_idx}A{layer_idx}{param_name}Bit{bit}W{weight}")
                        injection_number_total = int(size * proportion) if proportion else 1
                        count, inj_position, origin_value = injector0.injector_bit(matrix, size, injection_number_total, bit_seed, weight, "float16")

                        for prompt_idx in prompt_num:
                            prompt = test_file[prompt_idx]
                            error_type = f"U{block_idx}A{layer_idx}{param_name}Bit{bit}W{weight}"
                            image, negative_prompt_embeds_layer, negative_prompt_embeds = pipe(
                                prompt,
                                user_prompt_idx=prompt_idx,
                                user_error=True,
                                user_error_type=error_type,
                                user_save_type=save_type,
                                user_fixed_latents=fixed_latents,
                                user_save_iter=save_iter,
                            )
                            image = image.images[0]
                            clip_score = 0.0  # Placeholder
                            append_error_type_to_dict(
                                f'P{prompt_idx}_{error_type}',
                                clip_score,
                                filename=f'./error_detect/error_clipsore_dict_{blockname}_{partname}.json' if fixed_latents else f'./error_detect/unfixed_error_clipsore_dict_{blockname}_{partname}.json'
                            )
                        injector0.restore(matrix, inj_position, origin_value)
    print(f"Injected: {blockname}'s {partname}, block:{blockseq}, layer:{layerseq}, bit:{bitseq}, param:{param_type}, weightnum:{weight_num}, prompt:{prompt_num}.\n{save_type} saved")

def inj_Unet_d_res(
    pipe,
    test_file: np.ndarray,
    save_type: List[str],
    save_iter: List[int],
    fixed_latents: bool = True,
    blockname: str = "down_blocks",
    partname: str = "resnets",
    blockseq: List[int] = [1, 2, 3],
    layerseq: List[int] = [0, 1],
    bitseq: List[int] = [1],
    weight_num: int = 50,
    weight_num_save: int = 10,
    param_type: List[str] = ["conv1", "conv2", "norm1", "norm2", "time_emb_proj"],
    prompt_num: Iterable[int] = range(5),
    proportion: Optional[float] = None,
):
    """
    Injects faults into UNet down_blocks resnets.

    Args:
        pipe: CustomStableDiffusionPipeline instance.
        test_file (np.ndarray): Array of prompts.
        save_type (List[str]): Types of intermediate outputs to save.
        save_iter (List[int]): Iteration steps to save outputs.
        fixed_latents (bool): If True, uses fixed latents.
        blockname (str): Name of the block (e.g., 'down_blocks').
        partname (str): Name of the part (e.g., 'resnets').
        blockseq (List[int]): List of block indices.
        layerseq (List[int]): List of layer indices.
        bitseq (List[int]): List of bit positions to inject faults.
        weight_num (int): Number of weights to select.
        weight_num_save (int): Number of weights to save.
        param_type (List[str]): Types of parameters to inject faults into.
        prompt_num (Iterable[int]): Indices of prompts to process.
        proportion (float, optional): Proportion of weights to inject faults into.
    """
    for block_idx in blockseq:
        for layer_idx in layerseq:
            for param_name in param_type:
                for bit in bitseq:
                    imagefolder_path = f"./Images_error/Unet/down_blocks/{block_idx}/resnets/ResnetBlock2D{layer_idx}/{param_name}/bit{bit}"
                    create_image_folder(imagefolder_path)

    for block_idx in blockseq:
        for layer_idx in layerseq:
            for param_name in param_type:
                matrix_type = param_name
                if matrix_type == "conv1":
                    matrix = pipe.unet.down_blocks[block_idx].resnets[layer_idx].conv1.weight.storage()
                elif matrix_type == "conv2":
                    matrix = pipe.unet.down_blocks[block_idx].resnets[layer_idx].conv2.weight.storage()
                elif matrix_type == "norm1":
                    matrix = pipe.unet.down_blocks[block_idx].resnets[layer_idx].norm1.weight.storage()
                elif matrix_type == "norm2":
                    matrix = pipe.unet.down_blocks[block_idx].resnets[layer_idx].norm2.weight.storage()
                elif matrix_type == "time_emb_proj":
                    matrix = pipe.unet.down_blocks[block_idx].resnets[layer_idx].time_emb_proj.weight.storage()
                else:
                    continue
                size = len(matrix)
                weight_position = np.linspace(0, size - 1, weight_num, dtype=int).tolist()
                weight_position_save = weight_position[::len(weight_position) // weight_num_save]

                for bit in bitseq:
                    bit_seed = bit
                    for weight in weight_position_save:
                        print(f"Current UD{block_idx}R{layer_idx}{param_name}Bit{bit}W{weight}")
                        injection_number_total = int(size * proportion) if proportion else 1
                        error_type = f"D{block_idx}R{layer_idx}{param_name}Bit{bit}W{weight}"
                        count, inj_position, origin_value = injector0.injector_bit(matrix, size, injection_number_total, bit_seed, weight, "float16")

                        for prompt_idx in prompt_num:
                            prompt = test_file[prompt_idx]
                            image, negative_prompt_embeds_layer, negative_prompt_embeds = pipe(
                                prompt,
                                user_prompt_idx=prompt_idx,
                                user_error=True,
                                user_error_type=error_type,
                                user_save_type=save_type,
                                user_fixed_latents=fixed_latents,
                                user_save_iter=save_iter,
                            )
                            image = image.images[0]
                            clip_score = 0.0  # Placeholder
                            append_error_type_to_dict(
                                f'P{prompt_idx}_{error_type}',
                                clip_score,
                                filename=f'./error_detect/error_clipsore_dict_{blockname}_{partname}.json'
                            )
                            print(f"UD{block_idx}R{layer_idx}{param_name}Bit{bit}W{weight}P{prompt_idx}: {clip_score}")
                        injector0.restore(matrix, inj_position, origin_value)
    print(f"Injected: {blockname}'s {partname}, block:{blockseq}, layer:{layerseq}, bit:{bitseq}, param:{param_type}, weightnum:{weight_num}, prompt:{prompt_num}.")

def inj_Unet_u_res(
    pipe,
    test_file: np.ndarray,
    save_type: List[str],
    save_iter: List[int],
    fixed_latents: bool = True,
    blockname: str = "up_blocks",
    partname: str = "resnets",
    blockseq: List[int] = [0, 1, 2, 3],
    layerseq: List[int] = [0, 1, 2],
    bitseq: List[int] = [1],
    weight_num: int = 50,
    weight_num_save: int = 10,
    param_type: List[str] = ["conv1", "conv2", "norm1", "norm2", "time_emb_proj", "conv_shortcut"],
    prompt_num: Iterable[int] = range(5),
    proportion: Optional[float] = None,
):
    """
    Injects faults into UNet up_blocks resnets.

    Args:
        pipe: CustomStableDiffusionPipeline instance.
        test_file (np.ndarray): Array of prompts.
        save_type (List[str]): Types of intermediate outputs to save.
        save_iter (List[int]): Iteration steps to save outputs.
        fixed_latents (bool): If True, uses fixed latents.
        blockname (str): Name of the block (e.g., 'up_blocks').
        partname (str): Name of the part (e.g., 'resnets').
        blockseq (List[int]): List of block indices.
        layerseq (List[int]): List of layer indices.
        bitseq (List[int]): List of bit positions to inject faults.
        weight_num (int): Number of weights to select.
        weight_num_save (int): Number of weights to save.
        param_type (List[str]): Types of parameters to inject faults into.
        prompt_num (Iterable[int]): Indices of prompts to process.
        proportion (float, optional): Proportion of weights to inject faults into.
    """
    for block_idx in blockseq:
        for layer_idx in layerseq:
            for param_name in param_type:
                for bit in bitseq:
                    imagefolder_path = f"./Images_error/Unet/{blockname}/{block_idx}/{partname}/ResnetBlock2D{layer_idx}/{param_name}/bit{bit}"
                    create_image_folder(imagefolder_path)

    for block_idx in blockseq:
        for layer_idx in layerseq:
            for param_name in param_type:
                matrix_type = param_name
                if matrix_type == "conv1":
                    matrix = pipe.unet.up_blocks[block_idx].resnets[layer_idx].conv1.weight.storage()
                elif matrix_type == "conv2":
                    matrix = pipe.unet.up_blocks[block_idx].resnets[layer_idx].conv2.weight.storage()
                elif matrix_type == "norm1":
                    matrix = pipe.unet.up_blocks[block_idx].resnets[layer_idx].norm1.weight.storage()
                elif matrix_type == "norm2":
                    matrix = pipe.unet.up_blocks[block_idx].resnets[layer_idx].norm2.weight.storage()
                elif matrix_type == "time_emb_proj":
                    matrix = pipe.unet.up_blocks[block_idx].resnets[layer_idx].time_emb_proj.weight.storage()
                elif matrix_type == "conv_shortcut":
                    matrix = pipe.unet.up_blocks[block_idx].resnets[layer_idx].conv_shortcut.weight.storage()
                else:
                    continue
                size = len(matrix)
                weight_position = np.linspace(0, size - 1, weight_num, dtype=int).tolist()
                weight_position_save = weight_position[::len(weight_position) // weight_num_save]

                for bit in bitseq:
                    bit_seed = bit
                    for weight in weight_position_save:
                        print(f"Current UU{block_idx}R{layer_idx}{param_name}Bit{bit}W{weight}")
                        injection_number_total = int(size * proportion) if proportion else 1
                        error_type = f"U{block_idx}R{layer_idx}{param_name}Bit{bit}W{weight}"
                        count, inj_position, origin_value = injector0.injector_bit(matrix, size, injection_number_total, bit_seed, weight, "float16")

                        for prompt_idx in prompt_num:
                            prompt = test_file[prompt_idx]
                            image, negative_prompt_embeds_layer, negative_prompt_embeds = pipe(
                                prompt,
                                user_prompt_idx=prompt_idx,
                                user_error=True,
                                user_error_type=error_type,
                                user_save_type=save_type,
                                user_fixed_latents=fixed_latents,
                                user_save_iter=save_iter,
                            )
                            image = image.images[0]
                            clip_score = 0.0  # Placeholder
                            append_error_type_to_dict(
                                f'P{prompt_idx}_{error_type}',
                                clip_score,
                                filename=f'./error_detect/error_clipsore_dict_{blockname}_{partname}.json'
                            )
                            print(f"UU{block_idx}R{layer_idx}{param_name}Bit{bit}W{weight}P{prompt_idx}: {clip_score}")
                        injector0.restore(matrix, inj_position, origin_value)
    print(f"Injected: {blockname}'s {partname}, block:{blockseq}, layer:{layerseq}, bit:{bitseq}, param:{param_type}, weightnum:{weight_num}, prompt:{prompt_num}.")