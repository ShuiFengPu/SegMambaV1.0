# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import tempfile
from pathlib import Path

import torch
from mmengine import Config, DictAction
from mmengine.logging import MMLogger
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope

from mmseg.models import BaseSegmentor
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
import backbones

try:
    from mmengine.analysis import get_model_complexity_info
    from mmengine.analysis.print_helper import _format_size
except ImportError:
    raise ImportError('Please upgrade mmengine >= 0.6.0 to use this script.')
import time

def parse_args():
    parser = argparse.ArgumentParser(
        description='Get the FLOPs of a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[2048, 1024],
        help='input image size')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args

def get_timepc():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()

def replace_batchnorm(net):
    for child_name, child in net.named_children():
        if hasattr(child, 'fuse'):
            fused = child.fuse()
            setattr(net, child_name, fused)
            replace_batchnorm(fused)
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(net, child_name, torch.nn.Identity())
        else:
            replace_batchnorm(child)

def inference(args: argparse.Namespace, logger: MMLogger) -> dict:
    config_name = Path(args.config)

    if not config_name.exists():
        logger.error(f'Config file {config_name} does not exist')

    cfg: Config = Config.fromfile(config_name)
    cfg.work_dir = tempfile.TemporaryDirectory().name
    cfg.log_level = 'WARN'
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('scope', 'mmseg'))

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')
    result = {}

    model: BaseSegmentor = MODELS.build(cfg.model)
    if hasattr(model, 'auxiliary_head'):
        model.auxiliary_head = None
    if torch.cuda.is_available():
        model.cuda()
    model = revert_sync_batchnorm(model)
    result['ori_shape'] = input_shape[-2:]
    result['pad_shape'] = input_shape[-2:]
    data_batch = {
        'inputs': [torch.rand(input_shape)],
        'data_samples': [SegDataSample(metainfo=result)]
    }
    data = model.data_preprocessor(data_batch)
    model.eval()
    if cfg.model.decode_head.type in ['MaskFormerHead', 'Mask2FormerHead']:
        # TODO: Support MaskFormer and Mask2Former
        raise NotImplementedError('MaskFormer and Mask2Former are not '
                                  'supported yet.')
    outputs = get_model_complexity_info(
        model,
        input_shape=None,
        inputs=data['inputs'],
        show_table=False,
        show_arch=False)
    result['flops'] = _format_size(outputs['flops'])
    result['params'] = _format_size(outputs['params'])
    result['compute_type'] = 'direct: randomly generate a picture'

    print(result['pad_shape'])
    print(result['pad_shape'][0])
    print(result['pad_shape'][1])

    # gpu_id = 0
    # batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    # # batch_sizes = [32,64,128,256]
    # for bs in batch_sizes:
    #     print('batch size: ', bs)
    #     img_size_h = result['pad_shape'][0]
    #     img_size_w = result['pad_shape'][1]
    #     with torch.no_grad():
    #         x = torch.randn(bs, 3, img_size_h, img_size_w)
    #         # net = MobileMamba_T4s192()
    #         net = model
    #         # print("model_name is:", model_dict[model_name])
    #         replace_batchnorm(net)
    #         net.eval()
    #         pre_cnt, cnt = 2, 5
    #         if gpu_id > -1:
    #             torch.cuda.set_device(gpu_id)
    #             x = x.cuda()
    #             net.cuda()
    #             pre_cnt, cnt = 50, 20
    #         # FLOPs.fvcore_flop_count(net, torch.randn(1, 3, img_size, img_size).cuda(), show_arch=False)
    #
    #         # GPU
    #         for _ in range(pre_cnt):
    #             net(x)
    #         t_s = get_timepc()
    #         for _ in range(cnt):
    #             net(x)
    #         t_e = get_timepc()
    #         speed = f'{bs * cnt / (t_e - t_s):>7.3f}'
    #         laten = f'{1000 * (t_e - t_s) / cnt :>7.3f}'
    #         bs1laten = f'{1000 * (t_e - t_s) / cnt / bs :>7.3f}'
    #         print(
    #             f'[Batchsize: {bs}]\t [GPU-Speed: {speed}]\t [GPU-Latency-wbs:{laten}]\t [GPU-Latency-wobs:{bs1laten}]')
    #
    #         # GPU latency
    #         num_runs = 200
    #         z = torch.randn(1, 3, img_size_h, img_size_w).cuda()
    #         latencies = []
    #         with torch.no_grad():
    #             for _ in range(num_runs):
    #                 torch.cuda.synchronize()
    #                 start_time = time.time()
    #                 net(z)
    #                 torch.cuda.synchronize()
    #                 end_time = time.time()
    #                 latencies.append(end_time - start_time)
    #         avg_latency = sum(latencies) / num_runs
    #         avg_latency = f'{avg_latency * 1000:>7.3f}'
    #         print(f'[BS=1-GPU-Latency: {avg_latency}]')
    #
    #         # CPU
    #         net.cpu()
    #         net.eval()
    #         x = x.cpu()
    #         for _ in range(20):
    #             net(x)
    #         t_s = get_timepc()
    #         for _ in range(cnt):
    #             net(x)
    #         t_e = get_timepc()
    #         speed = f'{bs * cnt / (t_e - t_s):>7.3f}'
    #         laten = f'{1000 * (t_e - t_s) / cnt :>7.3f}'
    #         bs1laten = f'{1000 * (t_e - t_s) / cnt / bs :>7.3f}'
    #         print(
    #             f'[Batchsize: {bs}]\t [CPU-Speed: {speed}]\t [CPU-Latency-wbs: {laten}]\t [CPU-Latency-wobs: {bs1laten}]')
    #
    #         # if bs == 64:
    #         # CPU latency
    #         net.cpu()
    #         num_runs = 20
    #         z = torch.randn(1, 3, img_size_h, img_size_w).cpu()
    #         net.cpu()
    #         net.eval()
    #         latencies = []
    #         with torch.no_grad():
    #             for _ in range(num_runs):
    #                 start_time = time.time()
    #                 net(z)
    #                 end_time = time.time()
    #                 latencies.append(end_time - start_time)
    #         avg_latency = sum(latencies) / num_runs
    #         avg_latency = f'{avg_latency * 1000:>7.3f}'
    #         print(f'[BS=1-CPU-Latency: {avg_latency}]')

    return result


def main():

    args = parse_args()
    logger = MMLogger.get_instance(name='MMLogger')

    result = inference(args, logger)
    split_line = '=' * 30
    ori_shape = result['ori_shape']
    pad_shape = result['pad_shape']
    flops = result['flops']
    params = result['params']
    compute_type = result['compute_type']

    if pad_shape != ori_shape:
        print(f'{split_line}\nUse size divisor set input shape '
              f'from {ori_shape} to {pad_shape}')
    print(f'{split_line}\nCompute type: {compute_type}\n'
          f'Input shape: {pad_shape}\nFlops: {flops}\n'
          f'Params: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify '
          'that the flops computation is correct.')


if __name__ == '__main__':
    main()
