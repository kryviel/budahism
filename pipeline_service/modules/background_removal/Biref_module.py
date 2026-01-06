from typing import *
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from PIL import Image

from modules.background_removal.rmbg_manager import BackgroundRemovalService
import numpy as np


class BiRefNet:
    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):
        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model.eval()
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    
    def to(self, device: str):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()
        
    def __call__(self, image: Image.Image) -> Image.Image:
        image_size = image.size
        input_images = self.transform_image(image).unsqueeze(0).to("cuda")
        # Prediction
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        image.putalpha(mask)
        return image

class BiRefNetBackgroundRemovalService(BackgroundRemovalService):
    def _initialize_model_and_transforms(self) -> tuple[BiRefNet, transforms.Compose]:
        """
        Initialize BiRefNet model and transforms.
        """
        model: BiRefNet | None = None
        transform = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        return model, transform

    def _load_model(self) -> BiRefNet:
        """
        Load the BiRefNet background removal model.
        """
        model = BiRefNet(self.settings.model_id)
        model.to(self.device)
        return model


    def remove_background_new(self, image: Image.Image) -> Image.Image:
        """
        Remove the background from the image.
        """
        #Resize input 
        image = image.resize(self.settings.output_image_size, Image.Resampling.LANCZOS)
        
        output = self.model(image)
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        if bbox.size == 0:
            output = output.convert("RGB")
            output = output.resize(self.settings.output_image_size, Image.Resampling.LANCZOS)
            return output

        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))

        #Resize output 
        output = output.resize(self.settings.output_image_size, Image.Resampling.LANCZOS)
        return output

    def _remove_background(self, image: Image.Image) -> Image.Image:
        """
        Remove the background from the image.
        """
        return self.remove_background(image)