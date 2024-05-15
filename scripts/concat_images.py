import glob
import os
import cv2
import numpy as np


def cmp_concat():
    a_path = '/Users/sunyasheng/Desktop/Files/submissions/ongoing/mind_manipulation/materials/support_material/supp/test_res_value_inject_idback_css1_test_res_value_inject_idback_css1_2024-05-13/visualize/images/val'
    b_path = '/Users/sunyasheng/Desktop/Files/submissions/ongoing/mind_manipulation/materials/support_material/supp/test_res_value_inject_idback_css15_test_res_value_inject_idback_css15_2024-05-13/visualize/images/val'

    a_paths = sorted(glob.glob(os.path.join(a_path, '*.png')))
    b_paths = sorted(glob.glob(os.path.join(b_path, '*.png')))

    num = min(len(a_paths), len(b_paths))
    a_paths, b_paths = a_paths[:num], b_paths[:num]

    res_path = '/Users/sunyasheng/Desktop/Files/submissions/ongoing/mind_manipulation/materials/support_material/supp/test_res_value_inject_idback_css_concat_test_res_value_inject_idback_css15_2024-05-13/visualize/images/val'
    os.makedirs(res_path, exist_ok=True)

    for i, (a_path, b_path) in enumerate(zip(a_paths, b_paths)):
        a = cv2.imread(a_path)
        b = cv2.imread(b_path)
        res = np.concatenate([a, b], axis=1)

        basename = os.path.basename(a_path)
        cv2.imwrite(os.path.join(res_path, basename), res)
    


def select_concat():
    root_css15 = '/Users/sunyasheng/Desktop/asych_option/css15_intermediate/'
    root_css1 = '/Users/sunyasheng/Desktop/asych_option/css1_intermediate/'

    css15_paths = sorted(glob.glob(os.path.join(root_css15, '*.png')))
    css1_paths = sorted(glob.glob(os.path.join(root_css1, '*.png')))

    for css15_path in css15_paths:
        css15_img = cv2.imread(css15_path)
        import pdb; pdb.set_trace()

    # for css1_path in css1_paths:
    #     pass


if __name__ == '__main__':
    # cmp_concat()
