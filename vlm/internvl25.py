import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

path: str = "OpenGVLab/InternVL2_5-8B"
model: AutoModel = (
    AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    )
    .eval()
    .to("mps")
)

tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(path)

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def build_transform(input_size: int) -> T.Compose:
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False,
) -> list[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios: list[tuple[int, int]] = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        },
        key=lambda x: x[0] * x[1],
    )

    # find the closest aspect ratio to the target
    target_aspect_ratio: tuple[int, int] = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width: int = image_size * target_aspect_ratio[0]
    target_height: int = image_size * target_aspect_ratio[1]
    blocks: int = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img: Image.Image = image.resize((target_width, target_height))
    processed_images: list[Image.Image] = []
    for i in range(blocks):
        box: tuple[int, int, int, int] = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img: Image.Image = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img: Image.Image = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file: str, input_size: int = 448, max_num: int = 12) -> torch.Tensor:
    image: Image.Image = Image.open(image_file).convert("RGB")
    transform: T.Compose = build_transform(input_size=input_size)
    images: list[Image.Image] = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values: list[torch.Tensor] = [transform(image) for image in images]
    return torch.stack(pixel_values)


# If you want to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
path: str = "OpenGVLab/InternVL2_5-8B"
model: AutoModel = (
    AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
    )
    .eval()
    .to("mps")
)
tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
    path, trust_remote_code=True, use_fast=False
)

# set the max number of tiles in `max_num`
pixel_values: torch.Tensor = (
    load_image("/Users/jrp/Pictures/Processed/20241227-113523_DSC01174.jpg", max_num=12)
    .to(torch.bfloat16)
    .to("mps")
)
generation_config: dict = dict(max_new_tokens=1024, do_sample=True)

# pure-text conversation (纯文本对话)
question: str = "Hello, who are you?"
response: str
history: list[str]
response, history = model.chat(
    tokenizer, None, question, generation_config, history=None, return_history=True
)
print(f"User: {question}\nAssistant: {response}")

question = "Can you tell me a story?"
response, history = model.chat(
    tokenizer, None, question, generation_config, history=history, return_history=True
)
print(f"User: {question}\nAssistant: {response}")

# single-image single-round conversation (单图单轮对话)
question = "<image>\nPlease describe the image shortly."
response = model.chat(tokenizer, pixel_values, question, generation_config)
print(f"User: {question}\nAssistant: {response}")

# single-image multi-round conversation (单图多轮对话)
question = "<image>\nPlease describe the image in detail."
response, history = model.chat(
    tokenizer, pixel_values, question, generation_config, history=None, return_history=True
)
print(f"User: {question}\nAssistant: {response}")

question = "Please write a poem according to the image."
response, history = model.chat(
    tokenizer, pixel_values, question, generation_config, history=history, return_history=True
)
print(f"User: {question}\nAssistant: {response}")

# multi-image multi-round conversation, combined images (多图多轮对话，拼接图像)
pixel_values1: torch.Tensor = (
    load_image("/Users/jrp/Pictures/Processed/20241227-113523_DSC01174.jpg", max_num=12)
    .to(torch.bfloat16)
    .mps()
)
pixel_values2: torch.Tensor = (
    load_image("/Users/jrp/Pictures/Processed/20241227-115016_DSC01188.jpg", max_num=12)
    .to(torch.bfloat16)
    .mps()
)
pixel_values: torch.Tensor = torch.cat((pixel_values1, pixel_values2), dim=0)

question = "<image>\nDescribe the two images in detail."
response, history = model.chat(
    tokenizer, pixel_values, question, generation_config, history=None, return_history=True
)
print(f"User: {question}\nAssistant: {response}")

question = "What are the similarities and differences between these two images."
response, history = model.chat(
    tokenizer, pixel_values, question, generation_config, history=history, return_history=True
)
print(f"User: {question}\nAssistant: {response}")

# multi-image multi-round conversation, separate images (多图多轮对话，独立图像)
pixel_values1 = (
    load_image("/Users/jrp/Pictures/Processed/20241227-113523_DSC01174.jpg", max_num=12)
    .to(torch.bfloat16)
    .mps()
)
pixel_values2 = (
    load_image("/Users/jrp/Pictures/Processed/20241227-115016_DSC01188.jpg", max_num=12)
    .to(torch.bfloat16)
    .mps()
)
pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
num_patches_list: list[int] = [pixel_values1.size(0), pixel_values2.size(0)]

question = "Image-1: <image>\nImage-2: <image>\nDescribe the two images in detail."
response, history = model.chat(
    tokenizer,
    pixel_values,
    question,
    generation_config,
    num_patches_list=num_patches_list,
    history=None,
    return_history=True,
)
print(f"User: {question}\nAssistant: {response}")

question = "What are the similarities and differences between these two images."
response, history = model.chat(
    tokenizer,
    pixel_values,
    question,
    generation_config,
    num_patches_list=num_patches_list,
    history=history,
    return_history=True,
)
print(f"User: {question}\nAssistant: {response}")

# batch inference, single image per sample (单图批处理)
pixel_values1 = (
    load_image("/Users/jrp/Pictures/Processed/20241227-113523_DSC01174.jpg", max_num=12)
    .to(torch.bfloat16)
    .mps()
)
pixel_values2 = (
    load_image("/Users/jrp/Pictures/Processed/20241227-115016_DSC01188.jpg", max_num=12)
    .to(torch.bfloat16)
    .mps()
)
num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]
pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

questions: list[str] = ["<image>\nDescribe the image in detail."] * len(num_patches_list)
responses: list[str] = model.batch_chat(
    tokenizer,
    pixel_values,
    num_patches_list=num_patches_list,
    questions=questions,
    generation_config=generation_config,
)
for question, response in zip(questions, responses, strict=False):
    print(f"User: {question}\nAssistant: {response}")
