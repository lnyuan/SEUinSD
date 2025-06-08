#YLN
import torch
from torchmetrics.functional.multimodal import clip_score
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer
from PIL import Image
from functools import partial
from diffusers import utils
from transformers import image_utils, modeling_utils

#######################################
## calculate text-image similarity  ##
######################################
clip_score_fn = partial(clip_score, model_name_or_path="./clip-vit-base-patch16")#pertial():冻结
def calculate_clip_score(images_path, prompts):
    image = utils.load_image(images_path)
    image =image_utils.to_numpy_array(image)
    # print(image.shape)
    images_int = (image * 255).astype("uint8")
    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(2,0,1), prompts).detach()
    return round(float(clip_score), 4)

# def calculate_clip_score(image, prompts):
#     # image = utils.load_image(images_path)
#     image =image_utils.to_numpy_array(image)
#     # print(image.shape)
#     images_int = (image * 255).astype("uint8")
#     clip_score = clip_score_fn(torch.from_numpy(images_int).permute(2,0,1), prompts).detach()
#     return round(float(clip_score), 4)
##########################################
## calculate image-image similaruty     ##
##########################################
# Load the CLIP model
model_ID = "./clip-vit-base-patch16"
model = CLIPModel.from_pretrained(model_ID)
preprocess = CLIPImageProcessor.from_pretrained(model_ID)

# Define a function to load an image and preprocess it for CLIP
def load_and_preprocess_image(image_path):
    # Load the image from the specified path
    image = Image.open(image_path)
    # Apply the CLIP preprocessing to the image
    image = preprocess(image, return_tensors="pt")
    # Return the preprocessed image
    return image

def clip_img_score (img1_path,img2_path):
    # Load the two images and preprocess them for CLIP
    image_a = load_and_preprocess_image(img1_path)["pixel_values"]
    image_b = load_and_preprocess_image(img2_path)["pixel_values"]

    # Calculate the embeddings for the images using the CLIP
    with torch.no_grad():
        embedding_a = model.get_image_features(image_a)
        embedding_b = model.get_image_features(image_b)

    # Calculate the cosine similarity between the embeddings
    similarity_score = torch.nn.functional.cosine_similarity(embedding_a, embedding_b)
    return similarity_score.item()



# # test codes are as follows
# images_path="./a photo of an astronaut riding a horse on mars .png"
# prompts="a photo of an astronaut riding a horse on mars"
# # image = utils.load_image(images_path)
# # image=image_utils.to_numpy_array(image)
# images_path1 = "./Images-2-1/laion2b_en_part_00000_000000_bit0_34.8389.png"
# images_path2 = "./Images-2-1/laion2b_en_part_00000_000000_bit1_35.7811.png"
# prompt = "Blue Beach Umbrellas, Point Of Rocks, Crescent Beach, Siesta Key - Spiral Notebook"
#
# sd_clip_score = calculate_clip_score(images_path, prompts)
# print(f"CLIP score: {sd_clip_score}")
# image_similarity = clip_img_score(images_path1, images_path2)
# print(f"Image similarity:{image_similarity}")




