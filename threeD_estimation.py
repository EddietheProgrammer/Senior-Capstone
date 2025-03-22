import os
import numpy as np
from tqdm import tqdm
import imageio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lib.utils.tools import *
from lib.utils.learning import *
from lib.utils.utils_data import flip_data
from lib.data.dataset_wild import WildDetDataset
from lib.utils.vismo import render_and_save
from estimation import twoD_estimation

args = get_config('configs/pose3d/MB_ft_h36m_global_lite.yaml')


backbone = load_backbone(args)
checkpoint = torch.load('checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin', map_location=lambda storage, loc: storage)
checkpoint = {k.replace('module.', ''): v for k, v in checkpoint['model_pos'].items()}
backbone.load_state_dict(checkpoint, strict=True)
model_pos = backbone
model_pos.eval()

testloader_params = {
          'batch_size': 1,
          'shuffle': False,
          'num_workers': 0, # was 8
          'pin_memory': True,
          'prefetch_factor': None,
          'persistent_workers': False,
          'drop_last': False
}

vid = imageio.get_reader('assets/test.mp4',  'ffmpeg')
fps_in = vid.get_meta_data()['fps']
vid_size = vid.get_meta_data()['size']
# os.makedirs('assets/', exist_ok=True)

wild_dataset = WildDetDataset('test.json', clip_len=243, vid_size=vid_size, scale_range=None, focus=None)
test_loader = DataLoader(wild_dataset, **testloader_params)

results = []
with torch.no_grad():
    for batch_input in tqdm(test_loader):
        N, T = batch_input.shape[:2]

        predicted_3d_pos = model_pos(batch_input)
        predicted_3d_pos[:, 0, 0, 2] = 0

        results.append(predicted_3d_pos.cpu().numpy())

results = np.hstack(results)
results = np.concatenate(results)

print(results)