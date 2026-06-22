import argparse
import os
import pickle
import sys

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

from data_loader.dataset import DataSet
from modules.model import DevignModel, GGNNSum
from trainer import train, test, load_best_checkpoint
from utils import tally_param, debug


if __name__ == '__main__':
    torch.manual_seed(1000)
    np.random.seed(1000)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_type',
        type=str,
        help='Type of the model (devign/ggnn)',
        choices=['devign', 'ggnn'],
        default='devign'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Name of the dataset for experiment.'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Input Directory of the parser'
    )
    parser.add_argument(
        '--node_tag',
        type=str,
        help='Name of the node feature.',
        default='node_features'
    )
    parser.add_argument(
        '--graph_tag',
        type=str,
        help='Name of the graph feature.',
        default='graph'
    )
    parser.add_argument(
        '--label_tag',
        type=str,
        help='Name of the label feature.',
        default='targets'
    )

    parser.add_argument(
        '--feature_size',
        type=int,
        help='Size of feature vector for each node',
        default=100
    )
    parser.add_argument(
        '--graph_embed_size',
        type=int,
        help='Size of the Graph Embedding',
        default=200
    )
    parser.add_argument(
        '--num_steps',
        type=int,
        help='Number of steps in GGNN',
        default=6
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        help='Batch Size for training',
        default=128
    )

    parser.add_argument(
        '--num_epochs',
        type=int,
        default=1000,
        help='Maximum number of training steps/epochs'
    )
    parser.add_argument(
        '--dev_every',
        type=int,
        default=10,
        help='Run validation every N training steps (each step is one batch). '
             'For ~21k graphs and batch 128, one pass ≈ 167 steps; tiny dev_every is noisy.'
    )
    parser.add_argument(
        '--log_every',
        type=int,
        default=10,
        help='Log every N steps'
    )
    parser.add_argument(
        '--max_patience',
        type=int,
        default=20,
        help='Early stopping patience'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='',
        help='Path to saved checkpoint for eval_only'
    )
    parser.add_argument(
        '--eval_only',
        action='store_true',
        help='Only evaluate using a saved checkpoint'
    )
    parser.add_argument(
        '--plain_bce',
        action='store_true',
        help='Kept for compatibility. Model returns sigmoid probabilities, so training uses plain BCELoss.'
    )

    args = parser.parse_args()

    model_dir = os.path.join('models', args.dataset)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    checkpoint_path = args.checkpoint if args.checkpoint else os.path.join(model_dir, 'best_model.pt')

    input_dir = args.input_dir
    processed_data_path = os.path.join(input_dir, 'processed.bin')

    if False and os.path.exists(processed_data_path):
        debug('Reading already processed data from %s!' % processed_data_path)
        dataset = pickle.load(open(processed_data_path, 'rb'))
    else:
        dataset = DataSet(
            train_src=os.path.join(input_dir, 'train_GGNNinput.json'),
            valid_src=os.path.join(input_dir, 'valid_GGNNinput.json'),
            test_src=os.path.join(input_dir, 'test_GGNNinput.json'),
            batch_size=args.batch_size,
            n_ident=args.node_tag,
            g_ident=args.graph_tag,
            l_ident=args.label_tag
        )
        pickle.dump(dataset, open(processed_data_path, 'wb'))

    # Node feature width comes from the JSON; defaults (e.g. 100) often mismatch
    # create_ggnn_data embeddings (e.g. 169). DevignModel.concat_dim must match
    # graph_embed_size + actual feature dim or Conv1d fails in forward.
    inferred_fs = dataset.feature_size
    if inferred_fs > 0 and args.feature_size != inferred_fs:
        debug(
            'Overriding --feature_size %d with dataset value %d (from node_features).'
            % (args.feature_size, inferred_fs)
        )
        args.feature_size = inferred_fs

    if args.feature_size > args.graph_embed_size:
        print(
            'Warning!!! Graph Embed dimension should be at least equal to the feature dimension.\n'
            'Setting graph embedding size to feature size',
            file=sys.stderr
        )
        args.graph_embed_size = args.feature_size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    debug('Using device: %s' % device)

    if args.model_type == 'devign':
        model = DevignModel(
            input_dim=args.feature_size,
            output_dim=args.graph_embed_size,
            max_edge_types=dataset.max_edge_type,
            num_steps=args.num_steps
        )
    else:
        model = GGNNSum(
            input_dim=args.feature_size,
            output_dim=args.graph_embed_size,
            max_edge_types=dataset.max_edge_type,
            num_steps=args.num_steps
        )

    debug('Total Parameters : %d' % tally_param(model))

    # Compute class-balanced pos_weight = neg_count / pos_count.
    # BCEWithLogitsLoss applies sigmoid internally; model now returns raw logits.
    pos_count = sum(e.target for e in dataset.train_examples)
    neg_count = len(dataset.train_examples) - pos_count
    pw = torch.tensor([neg_count / pos_count], device=device)
    debug('Class counts  pos=%d  neg=%d  pos_weight=%.4f' % (pos_count, neg_count, pw.item()))
    loss_function = BCEWithLogitsLoss(pos_weight=pw)

    optimizer = Adam(model.parameters(), lr=args.lr)

    if args.eval_only:
        debug('Running in eval_only mode')
        model, threshold = load_best_checkpoint(model, checkpoint_path, device)
        test(
            model=model,
            dataset=dataset,
            loss_function=loss_function,
            device=device,
            threshold=threshold
        )
    else:
        train(
            model=model,
            dataset=dataset,
            max_steps=args.num_epochs,
            dev_every=args.dev_every,
            loss_function=loss_function,
            optimizer=optimizer,
            save_path=checkpoint_path,
            log_every=args.log_every,
            max_patience=args.max_patience,
            device=device
        )

        model, threshold = load_best_checkpoint(model, checkpoint_path, device)
        test(
            model=model,
            dataset=dataset,
            loss_function=loss_function,
            device=device,
            threshold=threshold
        )