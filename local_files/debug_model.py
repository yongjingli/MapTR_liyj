import argparse
import mmcv
import os
import shutil
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet3d.utils import collect_env, get_root_logger
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.bevformer.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.patches import Rectangle
import cv2

CAMS = ['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT',]
# we choose these samples not because it is easy but because it is hard
CANDIDATE=['n008-2018-08-01-15-16-36-0400_1533151184047036',
           'n008-2018-08-01-15-16-36-0400_1533151200646853',
           'n008-2018-08-01-15-16-36-0400_1533151274047332',
           'n008-2018-08-01-15-16-36-0400_1533151369947807',
           'n008-2018-08-01-15-16-36-0400_1533151581047647',
           'n008-2018-08-01-15-16-36-0400_1533151585447531',
           'n008-2018-08-01-15-16-36-0400_1533151741547700',
           'n008-2018-08-01-15-16-36-0400_1533151854947676',
           'n008-2018-08-22-15-53-49-0400_1534968048946931',
           'n008-2018-08-22-15-53-49-0400_1534968255947662',
           'n008-2018-08-01-15-16-36-0400_1533151616447606',
           'n015-2018-07-18-11-41-49+0800_1531885617949602',
           'n008-2018-08-28-16-43-51-0400_1535489136547616',
           'n008-2018-08-28-16-43-51-0400_1535489145446939',
           'n008-2018-08-28-16-43-51-0400_1535489152948944',
           'n008-2018-08-28-16-43-51-0400_1535489299547057',
           'n008-2018-08-28-16-43-51-0400_1535489317946828',
           'n008-2018-09-18-15-12-01-0400_1537298038950431',
           'n008-2018-09-18-15-12-01-0400_1537298047650680',
           'n008-2018-09-18-15-12-01-0400_1537298056450495',
           'n008-2018-09-18-15-12-01-0400_1537298074700410',
           'n008-2018-09-18-15-12-01-0400_1537298088148941',
           'n008-2018-09-18-15-12-01-0400_1537298101700395',
           'n015-2018-11-21-19-21-35+0800_1542799330198603',
           'n015-2018-11-21-19-21-35+0800_1542799345696426',
           'n015-2018-11-21-19-21-35+0800_1542799353697765',
           'n015-2018-11-21-19-21-35+0800_1542799525447813',
           'n015-2018-11-21-19-21-35+0800_1542799676697935',
           'n015-2018-11-21-19-21-35+0800_1542799758948001',
           ]

def perspective(cam_coords, proj_mat):
    pix_coords = proj_mat @ cam_coords
    valid_idx = pix_coords[2, :] > 0
    pix_coords = pix_coords[:, valid_idx]
    pix_coords = pix_coords[:2, :] / (pix_coords[2, :] + 1e-7)
    pix_coords = pix_coords.transpose(1, 0)
    return pix_coords

def parse_args():
    parser = argparse.ArgumentParser(description='vis hdmaptr map gt label')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--score-thresh', default=0.4, type=float, help='samples to visualize')
    parser.add_argument(
        '--show-dir', help='directory where visualizations will be saved')
    parser.add_argument('--show-cam', action='store_true', help='show camera pic')
    parser.add_argument(
        '--gt-format',
        type=str,
        nargs='+',
        default=['fixed_num_pts',],
        help='vis format, default should be "points",'
        'support ["se_pts","bbox","fixed_num_pts","polyline_pts"]')
    args = parser.parse_args()
    return args


def debug_dataset(config_path):
    args = parse_args()
    args.config = config_path

    cfg = Config.fromfile(args.config)

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    # samples_per_gpu = 1
    # if isinstance(cfg.data.test, dict):
    #     cfg.data.test.test_mode = True
    #     samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    #     if samples_per_gpu > 1:
    #         # Replace 'ImageToTensor' to 'DefaultFormatBundle'
    #         cfg.data.test.pipeline = replace_ImageToTensor(
    #             cfg.data.test.pipeline)
    # elif isinstance(cfg.data.test, list):
    #     for ds_cfg in cfg.data.test:
    #         ds_cfg.test_mode = True
    #     samples_per_gpu = max(
    #         [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
    #     if samples_per_gpu > 1:
    #         for ds_cfg in cfg.data.test:
    #             ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)
    #
    # if args.show_dir is None:
    #     args.show_dir = osp.join('./work_dirs',
    #                             osp.splitext(osp.basename(args.config))[0],
    #                             'vis_pred')
    # # create vis_label dir
    # mmcv.mkdir_or_exist(osp.abspath(args.show_dir))
    # cfg.dump(osp.join(args.show_dir, osp.basename(args.config)))
    logger = get_root_logger()
    # logger.info(f'DONE create vis_pred dir: {args.show_dir}')

    samples_per_gpu = 2
    dataset = build_dataset(cfg.data.train)
    # dataset = build_dataset(cfg.data.test)
    # dataset.is_vis_on_test = True #TODO, this is a hack
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        # workers_per_gpu=cfg.data.workers_per_gpu,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    # logger.info('Done build test data set')

    # 对于test-pipeline
    # LoadMultiViewImageFromFiles RandomScaleImageMultiViewImage NormalizeMultiviewImage
    # LoadMultiViewImageFromFiles: 读取results['img_filename']路径里的图像
    # RandomScaleImageMultiViewImage对图片进行0.5的缩放，
    # 对图像进行缩放后，'lidar2img'这个参数需要进行调整，对u,v分成乘以rand_scale
    # lidar2img = [scale_factor @ l2i for l2i in results['lidar2img']]
    # l2i 左乘雷达坐标[X,Y,Z,1]后得到[u,v,z,1], 然后乘以scale_factor，相当于u,v分成乘以rand_scale
    # results['img_aug_matrix'] = scale_factor(记录到img_aug_matrix中)
    # NormalizeMultiviewImage对图像进行归一化处理

    # MultiScaleFlipAug3D:包含汇总transform所需参数，实际进行的操作为transform
    # PadMultiViewImage: 对image进行padding到32的倍数，图像shape过程为:900x1600 -->scale 为450x800 --> padding 为480x800
    # DefaultFormatBundle3D: 在3D领域的一些同一的格式化处理
    # It simplifies the pipeline of formatting common fields for voxels,
    #     including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    #     "gt_semantic_seg".
    # CustomCollect3D: 提取所需的信息, 这里只取img作为模型的输入 data['img_metas']包含为其他的信息

    # 对于train-pipeline
    # LoadMultiViewImageFromFiles 同上
    # RandomScaleImageMultiViewImage 同上
    # PhotoMetricDistortionMultiViewImage
    # 对图像进行不同的数据增强,有些对立的数据增强根据不同mode进行选择
    # random brightness random contrast (mode 0) convert color from BGR to HSV
    # random saturation random hue  convert color from HSV to BGR random contrast (mode 1)  randomly swap channels
    # NormalizeMultiviewImage  同上
    # LoadPointsFromFile
    # 读取results['pts_filename']位置的文件，指定输入的维度(5)和数据的格式(LIDAR),记录到results['points']
    # CustomPointToMultiViewDepth   将results['points']投影图像坐标中，得到图像的深度图results['gt_depth']
    # PadMultiViewImageDepth 对gt-map进行padding
    # DefaultFormatBundle3D  同上
    # CustomCollect3D        取 'img', 'gt_depth'信息 data['img_metas']包含了其他信息

    # 产生线段gt的位置在dataset, vectormap_pipeline
    # 产生anns_results['gt_vecs_label'] anns_results['gt_vecs_pts_loc']并记录在gt_labels_3d，gt_bboxes_3d

    # example['gt_labels_3d'] = DC(gt_vecs_label, cpu_only=False)
    # example['gt_bboxes_3d'] = DC(gt_vecs_pts_loc, cpu_only=True)
    # gt_vecs_pts_loc 为所有线段的集合，为LiDARInstanceLines的对象，包含了很多实现方法在里面

    # def vectormap_pipeline(self, example, input_dict):
    #     anns_results = self.vector_map.gen_vectorized_samples(input_dict['annotation'] if 'annotation' in input_dict.keys() else input_dict['ann_info'],
    #                  example=example, feat_down_sample=self.aux_seg['feat_down_sample'])


    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    # - It supports a custom type :class:`DataContainer` which allows more
    #   flexible control of input data during both GPU and CPU inference.
    # dataloader 输出的data为DataContainer,需要将模型包装到MMDataParallel
    model = MMDataParallel(
        model.cuda(0), device_ids=[0])

    for i, data in enumerate(data_loader):
        losses = model(return_loss=True, **data)
        # result = model(return_loss=False, rescale=True, **data)
        print("fff")


        # img = data['img'][0].data[0]
        # img_metas = data['img_metas'][0].data[0]
        # gt_bboxes_3d = data['gt_bboxes_3d'].data[0]
        # gt_labels_3d = data['gt_labels_3d'].data[0]
        #
        # pts_filename = img_metas[0]['pts_filename']
        # pts_filename = osp.basename(pts_filename)
        # pts_filename = pts_filename.replace('__LIDAR_TOP__', '_').split('.')[0]
        #


        exit(1)


if __name__ == "__main__":
    print("Start")
    import os
    os.chdir("../")
    config_path = "./projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py"
    debug_dataset(config_path)
    print("End")
