# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import mmengine
from mmengine import Config, DictAction
from mmengine.evaluator import Evaluator
from mmengine.registry import init_default_scope

from mmdet.registry import DATASETS

import sys
sys.path.insert(0, '/root/Sofia/Genioos/detection_models/codetr/mmdetection/')

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate metric of the '
                                     'results saved in pkl format')
    parser.add_argument('config', help='Config of the model')
    parser.add_argument('pkl_results', help='Results in pickle format')
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


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    #print(cfg)
    init_default_scope(cfg.get('default_scope', 'mmdet'))

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    predictions = mmengine.load(args.pkl_results)

    evaluator = Evaluator(cfg.test_evaluator)
    #evaluator.dataset_meta = dataset.metainfo  # standard MS COCO dataset classes are provided
    evaluator.dataset_meta = cfg.metainfo  # provide custom classes for food dataset
    #print(evaluator.dataset_meta)  # visualize custom classes
    eval_results = evaluator.offline_evaluate(predictions)
    print(eval_results)


if __name__ == '__main__':
    main()
