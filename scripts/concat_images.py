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
    

def get_bbox(w, h, w_num, idx):
    row_idx = idx // w_num
    col_idx = idx % w_num
    row_st = (row_idx * h)
    col_st = (col_idx * w)

    row_ed = ((row_idx + 1) * h)
    col_ed = ((col_idx + 1) * w)
    return row_st, col_st, row_ed, col_ed


def select_concat():
    root_css15 = '/Users/sunyasheng/Desktop/asych_option/css15_intermediate/'
    root_css1 = '/Users/sunyasheng/Desktop/asych_option/css1_intermediate/'

    css15_paths = sorted(glob.glob(os.path.join(root_css15, '*.png')))
    css1_paths = sorted(glob.glob(os.path.join(root_css1, '*.png')))

    # output_img_dir = 'css15_seq/'
    # for css15_path in css15_paths:
    #     css15_img = cv2.imread(css15_path)
    #     width = css15_img.shape[1] // 8
    #     height = css15_img.shape[0] // 8
    #     # print('shape: ', css15_img.shape)
    #     idx_lists = list(range(0,65,5))
    #     all_crops = []
    #     for idx in idx_lists:
    #         (row_st, col_st, row_ed, col_ed) = get_bbox(width, height, 8, idx)
    #         crop_i = css15_img[row_st:row_ed, col_st:col_ed]
    #         all_crops.append(crop_i)
    #     all_crops = np.hstack(all_crops)
    #     output_img_path = os.path.join(output_img_dir, os.path.basename(css15_path))
    #     os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
    #     cv2.imwrite(output_img_path, all_crops)

    output_img_dir = 'css1_seq/'
    for css1_path in css1_paths:
        css1_img = cv2.imread(css1_path)
        width = css1_img.shape[1] // 8
        height = css1_img.shape[0] // 7
        idx_lists = list(range(0,50,5))
        all_crops = []
        for idx in idx_lists:
            (row_st, col_st, row_ed, col_ed) = get_bbox(width, height, 8, idx)
            print('row_st, col_st, row_ed, col_ed: ', row_st, col_st, row_ed, col_ed)
            crop_i = css1_img[row_st:row_ed, col_st:col_ed]
            all_crops.append(crop_i)
        all_crops = np.hstack(all_crops)
        output_img_path = os.path.join(output_img_dir, os.path.basename(css1_path))
        os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
        cv2.imwrite(output_img_path, all_crops)


if __name__ == '__main__':
    # cmp_concat()
    select_concat()
