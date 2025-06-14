import sys
import argparse
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
import random
from PIL import Image
import json
from pathlib import Path
from data.whoops.whoops_utils import get_timestamp
from eval_utils.dpc_cluster import cluster_dpc_knn
import h5py

def parse_args():
    parser = argparse.ArgumentParser(description='t-SNE visualization')
    # t-SNE settings
    parser.add_argument(
        '--n_components',
        type=int,
        default=2,
        help='(Deprecated, please use --n-components) the dimension of results'
    )
    parser.add_argument(
        '--perplexity',
        type=float,
        default=30.0,
        help='The perplexity is related to the number of nearest neighbors'
        'that is used in other manifold learning algorithms.')
    parser.add_argument(
        '--early_exaggeration',
        type=float,
        default=12.0,
        help='(Deprecated, please use --early-exaggeration) Controls how '
        'tight natural clusters in the original space are in the embedded '
        'space and how much space will be between them.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=200.0,
        help='(Deprecated, please use --learning-rate) The learning rate '
        'for t-SNE is usually in the range [10.0, 1000.0]. '
        'If the learning rate is too high, the data may look'
        'like a ball with any point approximately equidistant from its nearest'
        'neighbours. If the learning rate is too low, most points may look'
        'compressed in a dense cloud with few outliers.')
    parser.add_argument(
        '--n_iter',
        type=int,
        default=1000,
        help='(Deprecated, please use --n-iter) Maximum number of iterations '
        'for the optimization. Should be at least 250.')
    parser.add_argument(
        '--n_iter_without_progress',
        type=int,
        default=300,
        help='(Deprecated, please use --n-iter-without-progress) Maximum '
        'number of iterations without progress before we abort the '
        'optimization.')
    parser.add_argument(
        '--init', type=str, default='random', help='The init method')
    
    # data settings
    parser.add_argument("--result_path", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--norm_feat', type=bool, default=True, help='whether to L2 norm feature')
    args = parser.parse_args()
    return args

def fix_random_seeds(seed):
    """
    Fix random seeds.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def main(args):
    
    fs_dir = args.res_root + args.date_dir + '/'   # get_timestamp()
    # Path(fs_dir).mkdir(parents=True, exist_ok=True)
    
    tsne_work_dir = os.path.join(fs_dir, 'saved_pictures')
    Path(tsne_work_dir).mkdir(parents=True, exist_ok=True)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fix_random_seeds(args.random_seed)
    
    # data
    feats_list = []
    labels_list = []
    fr_file_names = [q for q in os.listdir(os.path.join(args.res_root, args.date_dir)) if q.endswith(".pt")]
    result_file_name = os.path.join(args.res_root, args.date_dir, fr_file_names[0])
    visual_tokens = torch.load(result_file_name)
    assert (num_toxic_tokens := visual_tokens.shape[0]) >= 1
    assert len(visual_tokens.shape) == 2
    
    if args.norm_feat:
        visual_tokens = nn.functional.normalize(visual_tokens, p=2, dim=-1)
        # feats_list.append(visual_tokens)
    
    with open(result_file_name.replace(".pt", ".json"), 'r') as f:
        labels_list = list(json.load(f))
        assert len(labels_list) == num_toxic_tokens
    
    color_n = 1
    color_list = [color_n]
    for i in range(1, len(labels_list)):  # TODO
        if labels_list[i] == labels_list[i-1]:
            color_list.append(color_n)
        else:
            color_n += 1
            color_list.append(color_n)
    
    print(f'{(ll := len(labels_list))} visual tokens prototypes found, Apply t-SNE to visualize them.')
    labels = np.array(labels_list)
    # features_pt = torch.cat(feats_list, dim=0)
    features_pt = visual_tokens
    assert features_pt.shape[0] == ll
        
    features = features_pt.detach().cpu().numpy()
    len_cmap = ll
    camp_name = 'Spectral'
    # print(features.shape, labels.shape)
    # build t-SNE model
    tsne_model = TSNE(
        n_components=args.n_components,
        perplexity=args.perplexity,
        early_exaggeration=args.early_exaggeration,
        learning_rate=args.learning_rate,
        n_iter=args.n_iter,
        n_iter_without_progress=args.n_iter_without_progress,
        init=args.init
        )

    # run and get results
    print('Running t-SNE......')
    result = tsne_model.fit_transform(features)
    res_min, res_max = result.min(0), result.max(0)
    res_norm = (result - res_min) / (res_max - res_min)
    # res_norm = result
    plt.figure(figsize=(10, 10), dpi=200)
    scatter = plt.scatter(
        res_norm[:, 0],
        res_norm[:, 1],
        alpha=1.0,
        s=10,
        c=color_list,
        label=labels,
        cmap=plt.cm.get_cmap(camp_name, len_cmap)
    )
    font = {"color": "darkred",
            "size": 12, 
            "family" : "serif"}
    
    # for i, idx_ in enumerate(labels_list):
        # plt.text(res_norm[:, 0][i], res_norm[:, 1][i], idx_, fontsize=16,
                    # ha='right', va='bottom')
                    
    # plt.xticks(range(len_cmap), labels_tokens)
    # cbar.set_label(label='groups', fontdict=font)
    ## cbar = plt.colorbar(ticks=range(len_cmap))
    # cbar.set_ticks()
    # print(label_tick)
    ## cbar.set_ticklabels(labels_list)
    # cbar.ax.set_yticklabels(labels_tokens, rotation=45)
    # cbar.set_ticklabels([st[-5:] for st in list(sample_label_dict.keys())])
    # plt.legend(*scatter.legend_elements(prop='colors', num=len(set(labels))), loc=2, title='classes')
    tsne_image_name = f'{args.dataset_name}_'
    plt.savefig(os.path.join(tsne_work_dir, tsne_image_name + f'prototype_{ll}_samples.png'))
    print(f'Success! Saved results to {fs_dir}')


if __name__ == '__main__':
    args = parse_args()
    args.res_root = '/DATA3/yangdingchen/coco/results/'
    args.date_dir = '241125-212715'  # TODO
    args.norm_feat = False  # TODO
    # args.interested_sample_id = '520208'  # TODO
    args.dataset_name = "coco_train"  # TODO
    args.random_seed = 0
    
    main(args)