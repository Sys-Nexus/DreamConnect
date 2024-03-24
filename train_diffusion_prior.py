from types import SimpleNamespace
import sys
sys.path.append('third_party/versatile_diffusion')
from lib.model_zoo.vd import VDCLIP

import json
import yaml
from easydict import EasyDict

def main():
    clip_cfg = {'symbol': 'clip',
                'args': {},
                'name': 'clip_frozen',
                'type': 'clip_frozen'}
    # clip_cfg = SimpleNamespace(**clip_cfg)    
    # clip_cfg = yaml.load(yaml.dump(clip_cfg),Loader=yaml.Loader)
    # clip_cfg = yaml.safe_load(yaml.dump(clip_cfg))
    clip_cfg = EasyDict(clip_cfg)
    vd_clip = VDCLIP(clip_cfg)


if __name__ == '__main__':
    main()
