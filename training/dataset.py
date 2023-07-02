import random
from pathlib import Path
from typing import Dict, Any

import PIL
import numpy as np
import mediapipe as mp
from typing import List, Union
from tqdm import tqdm
import torch
import glob
import os
import torch.utils.checkpoint
from PIL import Image
from packaging import version
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTokenizer
from constants import IMAGENET_STYLE_TEMPLATES_SMALL, IMAGENET_TEMPLATES_SMALL

def face_mask_google_mediapipe(images: List[Image.Image], blur_amount: float = 80.0, bias: float = 0.05) -> List[Image.Image]:
    """
    Returns a list of images with mask on the face parts.
    """

    mp_face_detection = mp.solutions.face_detection

    face_detection = mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    )

    masks = []
    for image in tqdm(images):

        image = np.array(image)

        results = face_detection.process(image)
        black_image = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)

        if results.detections:

            for detection in results.detections:

                x_min = int(
                    detection.location_data.relative_bounding_box.xmin * image.shape[1]
                )
                y_min = int(
                    detection.location_data.relative_bounding_box.ymin * image.shape[0]
                )
                width = int(
                    detection.location_data.relative_bounding_box.width * image.shape[1]
                )
                height = int(
                    detection.location_data.relative_bounding_box.height
                    * image.shape[0]
                )

                # draw the colored rectangle
                black_image[y_min : y_min + height, x_min : x_min + width] = 255

        black_image = Image.fromarray(black_image)
        masks.append(black_image)

    return masks

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }


class TextualInversionDataset(Dataset):

    def __init__(self, data_root: Path,
                 tokenizer: CLIPTokenizer,
                 learnable_property: str = "object",  # [object, style]
                 size: int = 512,
                 repeats: int = 100,
                 interpolation: str = "bicubic",
                 flip_p: float = 0.5,
                 set: str = "train",
                 placeholder_token: str = "*",
                 use_face_segmentation_condition: bool = False,
                 placeholder_token_at_data: str = "{}",
                 center_crop: bool = False):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.use_face_segmentation_condition = use_face_segmentation_condition
        self.placeholder_token = placeholder_token
        self.placeholder_token_at_data = placeholder_token_at_data
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.image_paths = []

        possibily_src_images = (
            glob.glob(str(data_root) + "/*.jpg")
            + glob.glob(str(data_root) + "/*.png")
            + glob.glob(str(data_root) + "/*.jpeg")
        )
        possibily_src_images = (
            set(possibily_src_images)
            - set(glob.glob(str(data_root) + "/*mask.png"))
            - set([str(data_root) + "/caption.txt"])
        )

        self.instance_images_path = list(set(possibily_src_images))

        if use_face_segmentation_condition:

            for idx in range(len(self.instance_images_path)):
                targ = f"{data_root}/{idx}.mask.png"
                # see if the mask exists
                if not Path(targ).exists():
                    print(f"Mask not found for {targ}")

                    print(
                        "Warning : this will pre-process all the images in the instance data root."
                    )

                    if len(self.mask_path) > 0:
                        print(
                            "Warning : masks already exists, but will be overwritten."
                        )

                    masks = face_mask_google_mediapipe(
                        [
                            Image.open(f).convert("RGB")
                            for f in self.instance_images_path
                        ]
                    )
                    for idx, mask in enumerate(masks):
                        mask.save(f"{data_root}/{idx}.mask.png")

                    break

            for idx in range(len(self.instance_images_path)):
                self.mask_path.append(f"{data_root}/{idx}.mask.png")


        self.num_images = len(self.image_paths)
        self._length = self.num_images

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        print(f"Running on {self.num_images} images")

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.templates = IMAGENET_STYLE_TEMPLATES_SMALL if learnable_property == "style" else IMAGENET_TEMPLATES_SMALL
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, i: int) -> Dict[str, Any]:
        image_path=self.image_paths[i % self.num_images]
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        image = Image.open(image_path)

        if not image.mode == "RGB":
            image = image.convert("RGB")

        example = dict()
        
        if self.templates is not None:
            example['text'] = random.choice(self.templates).format(self.placeholder_token)
        else:
            caption=''.join([i for i in image_name if not i.isdigit()])
            caption=caption.replace("_"," ")
            caption=caption.replace("(","")
            caption=caption.replace(")","")
            caption=caption.replace("-","")
            example['text'] = caption.replace(self.placeholder_token_at_data, self.placeholder_token) 
        
        example["input_ids"] = self.tokenizer(
            example['text'],
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w = img.shape[0], img.shape[1]
            img = img[(h - crop) // 2: (h + crop) // 2, (w - crop) // 2: (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)

        if self.use_face_segmentation_condition:

            example["mask"] = (
                self.image_transforms(
                    Image.open(self.mask_path[i % self.num_instance_images])
                )
                * 0.5
                + 1.0
            )

        return example
