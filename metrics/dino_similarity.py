import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class DINO_v2_Similarity(nn.Module):
    def __init__(self):
        super().__init__()
        # BACKBONE_SIZE = "small" # in ("small", "base", "large" or "giant")
        BACKBONE_SIZE = "large" # in ("small", "base", "large" or "giant")

        backbone_archs = {
            "small": "vits14",
            "base": "vitb14",
            "large": "vitl14",
            "giant": "vitg14",
        }
        backbone_arch = backbone_archs[BACKBONE_SIZE]
        backbone_name = f"dinov2_{backbone_arch}"

        backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
        backbone_model.eval()
        backbone_model.cuda()
        self.backbone_model = backbone_model
    
        self.register_buffer("mean", torch.tensor((0.48145466, 0.4578275, 0.40821073)))
        self.register_buffer("std", torch.tensor((0.26862954, 0.26130258, 0.27577711)))

    def encode_image(self, image):
        image = F.interpolate(image, (224,224))
        image = image - rearrange(self.mean, "c -> 1 c 1 1")
        image = image / rearrange(self.std, "c -> 1 c 1 1")
        image_features = self.backbone_model(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features


if __name__ == "__main__":
    dino_v2 = DINO_v2_Similarity().cuda()
    res_image_feat = dino_v2.encode_image(torch.randn((1, 3, 224, 224)).cuda())

