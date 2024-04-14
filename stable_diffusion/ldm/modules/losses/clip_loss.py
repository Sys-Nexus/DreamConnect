#  copied from styleclip codebase: https://github.com/orpatashnik/StyleCLIP/blob/main/criteria/clip_loss.py#L6



import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# class CLIPLoss(torch.nn.Module):

    # def __init__(self, stylegan_size=256):
    #     super(CLIPLoss, self).__init__()
    #     self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
    #     self.upsample = torch.nn.Upsample(scale_factor=7)
    #     self.avg_pool = torch.nn.AvgPool2d(kernel_size=stylegan_size // 32)

    # def forward(self, image, text):
    #     image = self.avg_pool(self.upsample(image))
    #     similarity = 1 - self.model(image, text)[0] / 100
    #     return similarity


class CLIPLoss(torch.nn.Module):

    def __init__(self, name: str = "ViT-L/14"):
        super(CLIPLoss, self).__init__()
        assert name in ("RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px")  # fmt: skip
        self.size = {"RN50x4": 288, "RN50x16": 384, "RN50x64": 448, "ViT-L/14@336px": 336}.get(name, 224)

        self.model, _ = clip.load(name, device="cpu", download_root="./")
        self.model.eval()

        self.register_buffer("mean", torch.tensor((0.48145466, 0.4578275, 0.40821073)))
        self.register_buffer("std", torch.tensor((0.26862954, 0.26130258, 0.27577711)))

    def encode_text(self, text: list[str]) -> torch.Tensor:
        text = clip.tokenize(text, truncate=True).to(next(self.parameters()).device)
        text_features = self.model.encode_text(text)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:  # Input images in range [0, 1].
        image = F.interpolate(image.float(), size=self.size, mode="bicubic", align_corners=False)
        image = image - rearrange(self.mean, "c -> 1 c 1 1")
        image = image / rearrange(self.std, "c -> 1 c 1 1")
        image_features = self.model.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    def forward(self, input_image, output_image, input_text, output_text):
        with torch.no_grad():
            input_v = self.encode_image(input_image)
            input_t = self.encode_text(input_text)
            output_t = self.encode_text(output_text)
        
        input_v = input_v.requrie_grad_(True)
        input_t = input_t.requrie_grad_(True)
        output_t = output_t.requrie_grad_(True)

        output_v = self.encode_image(output_image)
        
        import pdb; pdb.set_trace();
        print(input_text, output_text)
        import torchvision
        cat = torch.cat([input_image, output_image], dim=2)
        torchvision.utils.save_image(cat, "cat.png")
        similarity = 1 - F.cosine_similarity(output_v-input_v, output_t-input_t)/100.
        return similarity
