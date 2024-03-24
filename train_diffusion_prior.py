import sys
sys.path.append('third_party/versatile_diffusion')
from lib.model_zoo.vd import VDCLIP


def main():
    clip_cfg = {'symbol': 'clip',
                'args': {},
                'name': 'clip_frozen',
                'type': 'clip_frozen'}
    vd_clip = VDCLIP(clip_cfg)


if __name__ == '__main__':
    main()
