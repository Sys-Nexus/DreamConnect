from types import SimpleNamespace
import sys
sys.path.append('third_party/versatile_diffusion')
from lib.model_zoo.vd import VDCLIP

import json
import yaml

def main():
    clip_cfg = {'symbol': 'clip',
                'args': {},
                'name': 'clip_frozen',
                'type': 'clip_frozen'}
    # clip_cfg = SimpleNamespace(**clip_cfg)    
    clip_cfg = yaml.load(yaml.dumps(clip_cfg))
    vd_clip = VDCLIP(clip_cfg)


if __name__ == '__main__':
    main()
