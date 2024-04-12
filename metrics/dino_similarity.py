import torch
import torch.nn as nn


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
    
    def encode_image(self, image):
        image_feat = self.backbone_model(image)
        import pdb; pdb.set_trace()


if __name__ == "__main__":
    dino_v2 = DINO_v2_Similarity().cuda()
    dino_v2.encode_image(torch.randn((1, 3, 224, 224)).cuda())

