import os
import time

import cv2
import torch
from tqdm import tqdm

from metrics.metric_base import Emeasure as Emeasure_base
from metrics.metric_best import Emeasure as Emeasure_best
from metrics.metric_better import Emeasure as Emeasure_better

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cal_em_base = Emeasure_base(only_adaptive_em=False)
cal_em_better = Emeasure_better()
cal_em_best = Emeasure_best()

cal_ems = dict(base=cal_em_base, better=cal_em_better, best=cal_em_best)


def test(pred_root, mask_root, cal_em):
    mask_name_list = sorted(os.listdir(mask_root))
    tqdm_iter = tqdm(enumerate(mask_name_list), total=len(mask_name_list), leave=False)
    for i, mask_name in tqdm_iter:
        tqdm_iter.set_description(f"te=>{i + 1} ")
        mask_array = cv2.imread(os.path.join(mask_root, mask_name), cv2.IMREAD_GRAYSCALE)
        pred_array = cv2.imread(os.path.join(pred_root, mask_name), cv2.IMREAD_GRAYSCALE)
        cal_em.step(pred_array, mask_array)
    fixed_seg_results = cal_em.get_results()['em']
    return fixed_seg_results


def main():
    times = dict()
    for name, cal_em in cal_ems.items():
        start = time.time()
        seg_results = test(
            pred_root=('my_pred_path'),
            mask_root='my_gt_path',
            cal_em=cal_em
        )
        end = time.time()
        print('\n', seg_results)
        times[name] = end - start
    print(times)


if __name__ == '__main__':
    main()
