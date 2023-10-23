pkl_path = 'third_party/StableDiffusionReconstruction/codes/utils/misc/nsd_coco_caption.pkl'


def read_pkl(pkl_path):
    import pickle
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    return data


data = read_pkl(pkl_path)
import pdb; pdb.set_trace()