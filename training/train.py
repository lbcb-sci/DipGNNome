import torch
import torch.nn as nn
import wandb
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
import dgl
import os
from datetime import datetime
import utils
import argparse
import yaml

from SymGatedGCN import SymGatedGCNModel

class WeightedBCEWSymmetryLoss(nn.Module):
    def __init__(self, pos_weight_gt_bin, pos_weight_malicious):
        super(WeightedBCEWSymmetryLoss, self).__init__()
        # Use BCEWithLogitsLoss with proper pos_weight for each head
        self.criterion_gt_bin = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_gt_bin)
        self.criterion_malicious = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_malicious)
        self.alpha = 0.1
    
    def forward(self, predictions_gt_bin, rev_predictions_gt_bin, labels_gt_bin,
                predictions_malicious, rev_predictions_malicious, labels_malicious):
        labels_gt_bin = labels_gt_bin.float()
        labels_malicious = labels_malicious.float()
        
        # GT_BIN losses
        BCE_org_gt_bin = self.criterion_gt_bin(predictions_gt_bin, labels_gt_bin)
        BCE_rev_gt_bin = self.criterion_gt_bin(rev_predictions_gt_bin, labels_gt_bin)
        abs_diff_gt_bin = torch.abs(predictions_gt_bin - rev_predictions_gt_bin)
        loss_gt_bin = (BCE_org_gt_bin + BCE_rev_gt_bin + self.alpha * abs_diff_gt_bin).mean()
        
        # MALICIOUS losses
        BCE_org_malicious = self.criterion_malicious(predictions_malicious, labels_malicious)
        BCE_rev_malicious = self.criterion_malicious(rev_predictions_malicious, labels_malicious)
        abs_diff_malicious = torch.abs(predictions_malicious - rev_predictions_malicious)
        loss_malicious = (BCE_org_malicious + BCE_rev_malicious + self.alpha * abs_diff_malicious).mean()
        
        # Combine losses
        total_loss = loss_gt_bin + loss_malicious
        return total_loss, loss_gt_bin, loss_malicious


def prepare_subgraph(g, sub_g, device):
    # Move graph to device before accessing its data
    sub_g = sub_g.to(device)
    
    edge_ids = sub_g.edata['_ID'].to(device)
    ol_len = g.edata['overlap_length'][edge_ids].float()
    ol_len /= 10000
    #ol_sim = g.edata['overlap_similarity'][edge_ids].float()
    e = ol_len.unsqueeze(-1)
    # Node features - get node IDs and access attributes from original graph
    node_ids = sub_g.ndata['_ID'].to(device)
    
    # Node degree features from subgraph
    pe_in = sub_g.in_degrees().float().unsqueeze(1)
    pe_out = sub_g.out_degrees().float().unsqueeze(1)
    
    # Access node attributes from original graph using node IDs
    read_length = g.ndata['read_length'][node_ids].to(device)/100000
    support = g.ndata['support'][node_ids].to(device)

    # Create new gt_score as sum of skip, unknown, and cross_strand attributes
    skip_attr = g.edata.get('skip', torch.zeros(g.num_edges(), device=device))[edge_ids].to(device)
    unknown_attr = g.edata.get('unknown', torch.zeros(g.num_edges(), device=device))[edge_ids].to(device)
    cross_strand_attr = g.edata.get('cross_strand', torch.zeros(g.num_edges(), device=device))[edge_ids].to(device)
    
    # Sum the three attributes to create the new gt_score
    y_bin = (skip_attr + unknown_attr + cross_strand_attr).to(device)
    
    # Use cross_chr for malicious label
    y_malicious = g.edata.get('cross_chr', torch.zeros(g.num_edges(), device=device))[edge_ids].to(device)
    
    """
    y_strand_change = g.edata['strand_change'][edge_ids].to(device)
    y_cross_chr = g.edata['cross_chr'][edge_ids].to(device)
    y_skip_forward = g.edata['skip_forward'][edge_ids].to(device)
    y_skip_backward = g.edata['skip_backward'][edge_ids].to(device)
    y_all = torch.max(torch.stack([y_strand_change, y_cross_chr, y_skip_forward, y_skip_backward]), dim=0)[0]
    """

    # Combine node features - ensure all tensors have same dimensions
    pe = torch.cat((pe_in, pe_out, support.unsqueeze(1), read_length.unsqueeze(1)), dim=1).to(device)

    return sub_g, e, pe, y_bin, y_malicious


def train(model, data_path, train_selection, valid_selection, device, config, diploid=False, mode='default', loss_weight_gt_bin=1.0, loss_weight_malicious=1.0):
    compute_metrics_every_n_train = 1
    compute_metrics_every_n_valid = 1
    overfit = not bool(valid_selection)

    criterion = WeightedBCEWSymmetryLoss(pos_weight_gt_bin=torch.tensor([loss_weight_gt_bin]).to(device), pos_weight_malicious=torch.tensor([loss_weight_malicious]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['decay'], patience=config['patience'], verbose=True)

    time_start = datetime.now()

    with wandb.init(project=wandb_project, config=config, mode=config['wandb_mode'], name=run_name):
        wandb.watch(model, criterion, log='all', log_freq=1000)

        best_f1_score = 0
        for epoch in range(config['num_epochs']):
            print(f'===> TRAINING EPOCH {epoch}')
            model.train()
            random.shuffle(train_selection)
            train_loss = []
            train_metrics = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
            train_metrics_malicious = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}

            for idx, graph_name in enumerate(train_selection):
                g = dgl.load_graphs(os.path.join(data_path, graph_name + '.dgl'))[0][0]
                num_clusters = utils.get_num_nodes_per_cluster(g, config)
                g = g.long()
                full_graph = bool(num_clusters <= 1)
                if not full_graph:
                    d = dgl.metis_partition(g, num_clusters, extra_cached_hops=config['k_extra_hops'])
                    sub_gs = list(d.values())
                    random.shuffle(sub_gs)
                else:
                    # Create a subgraph that includes all nodes/edges
                    all_nodes = torch.arange(g.number_of_nodes())
                    sub_gs = [dgl.node_subgraph(g, all_nodes)]
                g = g.to(device)
                for sub_g in sub_gs:
                    sub_g, e, x, y_bin, y_malicious = prepare_subgraph(g, sub_g, device)
                    
                    # Forward predictions for both heads
                    edge_predictions_gt_bin, edge_predictions_malicious = model(sub_g, x, e)
                    edge_predictions_gt_bin = edge_predictions_gt_bin.squeeze(-1)
                    edge_predictions_malicious = edge_predictions_malicious.squeeze(-1)
                    
                    # Reverse predictions for both heads
                    sub_g_reverse = dgl.reverse(sub_g, copy_ndata=True, copy_edata=True)
                    rev_edge_predictions_gt_bin, rev_edge_predictions_malicious = model(sub_g_reverse, x, e)
                    rev_edge_predictions_gt_bin = rev_edge_predictions_gt_bin.squeeze(-1)
                    rev_edge_predictions_malicious = rev_edge_predictions_malicious.squeeze(-1)
                    
                    sigmoid_predictions = torch.sigmoid(edge_predictions_gt_bin)  # Apply sigmoid to get [0, 1] range
                    
                    optimizer.zero_grad()
                    loss, loss_gt_bin, loss_malicious = criterion(
                        edge_predictions_gt_bin, rev_edge_predictions_gt_bin, y_bin,
                        edge_predictions_malicious, rev_edge_predictions_malicious, y_malicious
                    )
                    loss.backward()
                    optimizer.step()
                    train_loss.append(loss.item())

                    if epoch % compute_metrics_every_n_train == 0:
                        # Compute metrics for gt_bin head
                        predictions = (sigmoid_predictions > 0.5).float()
                        # Treat class 1 as positive class for metrics
                        tp = ((predictions == 1) & (y_bin == 1)).sum().item()  # Correctly predicted 1
                        fp = ((predictions == 1) & (y_bin == 0)).sum().item()  # Incorrectly predicted 1
                        fn = ((predictions == 0) & (y_bin == 1)).sum().item()  # Missed class 1
                        tn = ((predictions == 0) & (y_bin == 0)).sum().item()  # Correctly predicted 0
                        train_metrics['tp'] += tp
                        train_metrics['fp'] += fp
                        train_metrics['fn'] += fn
                        train_metrics['tn'] += tn
                        
                        # Compute metrics for malicious head
                        sigmoid_predictions_malicious = torch.sigmoid(edge_predictions_malicious)
                        predictions_malicious = (sigmoid_predictions_malicious > 0.5).float()
                        tp_mal = ((predictions_malicious == 1) & (y_malicious == 1)).sum().item()
                        fp_mal = ((predictions_malicious == 1) & (y_malicious == 0)).sum().item()
                        fn_mal = ((predictions_malicious == 0) & (y_malicious == 1)).sum().item()
                        tn_mal = ((predictions_malicious == 0) & (y_malicious == 0)).sum().item()
                        train_metrics_malicious['tp'] += tp_mal
                        train_metrics_malicious['fp'] += fp_mal
                        train_metrics_malicious['fn'] += fn_mal
                        train_metrics_malicious['tn'] += tn_mal

            # Calculate training metrics
            if train_metrics['tp'] + train_metrics['fp'] > 0:
                precision = train_metrics['tp'] / (train_metrics['tp'] + train_metrics['fp'])
            else:
                precision = 0
            
            if train_metrics['tp'] + train_metrics['fn'] > 0:
                recall = train_metrics['tp'] / (train_metrics['tp'] + train_metrics['fn'])
            else:
                recall = 0
            
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Calculate false positive rate and false negative rate
            if train_metrics['fp'] + train_metrics['tn'] > 0:
                fpr = train_metrics['fp'] / (train_metrics['fp'] + train_metrics['tn'])
            else:
                fpr = 0
                
            if train_metrics['fn'] + train_metrics['tp'] > 0:
                fnr = train_metrics['fn'] / (train_metrics['fn'] + train_metrics['tp'])
            else:
                fnr = 0
                
            # Calculate malicious head training metrics
            if train_metrics_malicious['tp'] + train_metrics_malicious['fp'] > 0:
                precision_malicious = train_metrics_malicious['tp'] / (train_metrics_malicious['tp'] + train_metrics_malicious['fp'])
            else:
                precision_malicious = 0
            
            if train_metrics_malicious['tp'] + train_metrics_malicious['fn'] > 0:
                recall_malicious = train_metrics_malicious['tp'] / (train_metrics_malicious['tp'] + train_metrics_malicious['fn'])
            else:
                recall_malicious = 0
            
            f1_score_malicious = 2 * (precision_malicious * recall_malicious) / (precision_malicious + recall_malicious) if (precision_malicious + recall_malicious) > 0 else 0

            # Validation loop
            if not overfit:
                print('===> VALIDATION')
                model.eval()
                valid_loss = []
                valid_metrics = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
                valid_metrics_malicious = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}

                with torch.no_grad():
                    for idx, graph_name in enumerate(valid_selection):
                        g = dgl.load_graphs(os.path.join(data_path, graph_name + '.dgl'))[0][0]
                        num_clusters = utils.get_num_nodes_per_cluster(g, config)
                        g = g.long()
                        full_graph = bool(num_clusters <= 1)
                        if not full_graph:
                            d = dgl.metis_partition(g, num_clusters, extra_cached_hops=config['k_extra_hops'])
                            sub_gs = list(d.values())
                        else:
                            # Create a subgraph that includes all nodes/edges
                            all_nodes = torch.arange(g.number_of_nodes())
                            sub_gs = [dgl.node_subgraph(g, all_nodes)]
                        
                        g = g.to(device)
                        for sub_g in sub_gs:
                            sub_g, e, x, y_bin, y_malicious = prepare_subgraph(g, sub_g, device)
                            
                            # Forward predictions for both heads
                            edge_predictions_gt_bin, edge_predictions_malicious = model(sub_g, x, e)
                            edge_predictions_gt_bin = edge_predictions_gt_bin.squeeze(-1)
                            edge_predictions_malicious = edge_predictions_malicious.squeeze(-1)
                            
                            # Reverse predictions for both heads
                            sub_g_reverse = dgl.reverse(sub_g, copy_ndata=True, copy_edata=True)
                            rev_edge_predictions_gt_bin, rev_edge_predictions_malicious = model(sub_g_reverse, x, e)
                            rev_edge_predictions_gt_bin = rev_edge_predictions_gt_bin.squeeze(-1)
                            rev_edge_predictions_malicious = rev_edge_predictions_malicious.squeeze(-1)
                            
                            sigmoid_predictions = torch.sigmoid(edge_predictions_gt_bin)  # Apply sigmoid to get [0, 1] range
                            
                            loss, loss_gt_bin, loss_malicious = criterion(
                                edge_predictions_gt_bin, rev_edge_predictions_gt_bin, y_bin,
                                edge_predictions_malicious, rev_edge_predictions_malicious, y_malicious
                            )
                            valid_loss.append(loss.item())

                            if epoch % compute_metrics_every_n_valid == 0:
                                # Compute metrics for gt_bin head
                                predictions = (sigmoid_predictions > 0.5).float()
                                # Treat class 1 as positive class for metrics
                                tp = ((predictions == 1) & (y_bin == 1)).sum().item()  # Correctly predicted 1
                                fp = ((predictions == 1) & (y_bin == 0)).sum().item()  # Incorrectly predicted 1
                                fn = ((predictions == 0) & (y_bin == 1)).sum().item()  # Missed class 1
                                tn = ((predictions == 0) & (y_bin == 0)).sum().item()  # Correctly predicted 0
                                valid_metrics['tp'] += tp
                                valid_metrics['fp'] += fp
                                valid_metrics['fn'] += fn
                                valid_metrics['tn'] += tn
                                
                                # Compute metrics for malicious head
                                sigmoid_predictions_malicious = torch.sigmoid(edge_predictions_malicious)
                                predictions_malicious = (sigmoid_predictions_malicious > 0.5).float()
                                tp_mal = ((predictions_malicious == 1) & (y_malicious == 1)).sum().item()
                                fp_mal = ((predictions_malicious == 1) & (y_malicious == 0)).sum().item()
                                fn_mal = ((predictions_malicious == 0) & (y_malicious == 1)).sum().item()
                                tn_mal = ((predictions_malicious == 0) & (y_malicious == 0)).sum().item()
                                valid_metrics_malicious['tp'] += tp_mal
                                valid_metrics_malicious['fp'] += fp_mal
                                valid_metrics_malicious['fn'] += fn_mal
                                valid_metrics_malicious['tn'] += tn_mal

                print(f'Validation metrics:')
                print(f'  GT_BIN - True positives: {valid_metrics["tp"]}')
                print(f'  GT_BIN - False positives: {valid_metrics["fp"]}')
                print(f'  GT_BIN - False negatives: {valid_metrics["fn"]}')
                print(f'  GT_BIN - True negatives: {valid_metrics["tn"]}')
                print(f'  MALICIOUS - True positives: {valid_metrics_malicious["tp"]}')
                print(f'  MALICIOUS - False positives: {valid_metrics_malicious["fp"]}')
                print(f'  MALICIOUS - False negatives: {valid_metrics_malicious["fn"]}')
                print(f'  MALICIOUS - True negatives: {valid_metrics_malicious["tn"]}')

                # Calculate validation metrics for gt_bin
                if valid_metrics['tp'] + valid_metrics['fp'] > 0:
                    val_precision = valid_metrics['tp'] / (valid_metrics['tp'] + valid_metrics['fp'])
                else:
                    val_precision = 0
                
                if valid_metrics['tp'] + valid_metrics['fn'] > 0:
                    val_recall = valid_metrics['tp'] / (valid_metrics['tp'] + valid_metrics['fn'])
                else:
                    val_recall = 0
                
                val_f1_score = 2 * (val_precision * val_recall) / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0

                # Calculate validation false positive rate and false negative rate for gt_bin
                if valid_metrics['fp'] + valid_metrics['tn'] > 0:
                    val_fpr = valid_metrics['fp'] / (valid_metrics['fp'] + valid_metrics['tn'])
                else:
                    val_fpr = 0
                    
                if valid_metrics['fn'] + valid_metrics['tp'] > 0:
                    val_fnr = valid_metrics['fn'] / (valid_metrics['fn'] + valid_metrics['tp'])
                else:
                    val_fnr = 0
                    
                # Calculate validation metrics for malicious
                if valid_metrics_malicious['tp'] + valid_metrics_malicious['fp'] > 0:
                    val_precision_malicious = valid_metrics_malicious['tp'] / (valid_metrics_malicious['tp'] + valid_metrics_malicious['fp'])
                else:
                    val_precision_malicious = 0
                
                if valid_metrics_malicious['tp'] + valid_metrics_malicious['fn'] > 0:
                    val_recall_malicious = valid_metrics_malicious['tp'] / (valid_metrics_malicious['tp'] + valid_metrics_malicious['fn'])
                else:
                    val_recall_malicious = 0
                
                val_f1_score_malicious = 2 * (val_precision_malicious * val_recall_malicious) / (val_precision_malicious + val_recall_malicious) if (val_precision_malicious + val_recall_malicious) > 0 else 0

                # Update learning rate scheduler
                scheduler.step(sum(valid_loss) / len(valid_loss) if valid_loss else float('inf'))

            # Log metrics
            metrics_dict = {
                'loss': sum(train_loss) / len(train_loss) if train_loss else 0,
                'gt_bin_precision': precision,
                'gt_bin_recall': recall,
                'gt_bin_fpr': fpr,
                'gt_bin_fnr': fnr,
                'gt_bin_f1': f1_score,
                'malicious_precision': precision_malicious,
                'malicious_recall': recall_malicious,
                'malicious_f1': f1_score_malicious,
                'epoch': epoch
            }

            if not overfit:
                metrics_dict.update({
                    'val_loss': sum(valid_loss) / len(valid_loss) if valid_loss else 0,
                    'val_gt_bin_precision': val_precision,
                    'val_gt_bin_recall': val_recall,
                    'val_gt_bin_fpr': val_fpr,
                    'val_gt_bin_fnr': val_fnr,
                    'val_gt_bin_f1': val_f1_score,
                    'val_malicious_precision': val_precision_malicious,
                    'val_malicious_recall': val_recall_malicious,
                    'val_malicious_f1': val_f1_score_malicious
                })

            wandb.log(metrics_dict)

            # Save best model based on F1 score
            current_f1 = val_f1_score if not overfit else f1_score
            if current_f1 > best_f1_score:
                best_f1_score = current_f1
                model_path = os.path.join(save_model_path, f'{args.run_name}_best.pt')
                torch.save(model.state_dict(), model_path)
                print(f"Saved model with F1 score: {best_f1_score:.4f}")
            
            # Save model every 10 epochs
            if epoch % 10 == 0:
                checkpoint_path = os.path.join(save_model_path, f'{args.run_name}_epoch_{epoch}.pt')
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Saved checkpoint at epoch {epoch}")

            print(f"Epoch {epoch} completed in {datetime.now() - time_start}")
            time_start = datetime.now()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="experiment eval script")
    parser.add_argument("--load_checkpoint", type=str, default='', help="dataset path")
    parser.add_argument("--save_model_path", type=str, default='/mnt/sod2-project/csb4/wgs/martin/trained_models', help="dataset path")
    parser.add_argument("--data_path", type=str, default='data', help="dataset path")
    parser.add_argument("--run_name", type=str, default='test', help="dataset path")
    parser.add_argument("--device", type=str, default='cpu', help="dataset path")
    parser.add_argument("--data_config", type=str, default='data_debug.yml', help="dataset path")
    parser.add_argument("--hyper_config", type=str, default='configs/config.yml', help="dataset path")
    parser.add_argument('--diploid', action='store_true', default=False, help="Enable evaluation (default: True)")
    parser.add_argument("--wandb", type=str, default='debug', help="dataset path")
    parser.add_argument("--seed", type=int, default=0, help="dataset path")
    parser.add_argument('--mode', type=str, default='default')

    args = parser.parse_args()

    wandb_project = args.wandb
    run_name = args.run_name
    load_checkpoint = args.load_checkpoint
    save_model_path = args.save_model_path
    data_path = args.data_path
    device = args.device
    data_config = args.data_config
    diploid = args.diploid
    
    hyper_config = args.hyper_config
    mode = args.mode

    full_dataset, valid_selection, train_selection = utils.create_dataset_dicts(data_config=data_config)
    valid_data = utils.get_numbered_graphs(valid_selection)
    train_data = utils.get_numbered_graphs(train_selection, starting_counts=valid_selection)

    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)

    with open(hyper_config) as file:
        config = yaml.safe_load(file)['training']

    utils.set_seed(args.seed)

    add_features = False
    if mode.startswith('ghost'):
        mode_split = mode.split('-')
        add_features = bool(int(mode_split[2]))

    if add_features:
        node_features = config['node_features'] + 1
        edge_features = config['edge_features'] + 1
    else:
        node_features = config['node_features']
        edge_features = config['edge_features']

    model = SymGatedGCNModel(
        node_features,
        edge_features,
        config['hidden_features'],
        config['hidden_edge_features'],
        config['num_gnn_layers'],
        config['hidden_edge_scores'],
        config['nb_pos_enc'],
        config['nr_classes'],
        dropout=config['dropout'],
        pred_dropout=config.get('pred_dropout', 0),
        norm=config.get('norm', 'layer')  # Default to 'layer' if not specified
    )

    # Add statistics gathering
    total_edges = 0
    skip_edges = 0
    unknown_edges = 0
    cross_strand_edges = 0
    cross_chr_edges = 0
    gt_bin_edges = 0
    malicious_edges = 0
    
    # Node statistics
    total_nodes = 0
    total_degree_sum = 0
    train_nodes = 0
    train_degree_sum = 0
    valid_nodes = 0
    valid_degree_sum = 0

    # Analyze training graphs
    for graph_name in train_data:
        g = dgl.load_graphs(os.path.join(data_path, graph_name + '.dgl'))[0][0]
        total_edges += g.num_edges()
        train_nodes += g.num_nodes()
        
        # Calculate node degrees for training set
        in_degrees = g.in_degrees()
        out_degrees = g.out_degrees()
        total_degrees = in_degrees + out_degrees
        train_degree_sum += total_degrees.sum().item()
        
        # Count individual attributes
        if 'skip' in g.edata:
            skip_edges += g.edata['skip'].sum().item()
        if 'unknown' in g.edata:
            unknown_edges += g.edata['unknown'].sum().item()
        if 'cross_strand' in g.edata:
            cross_strand_edges += g.edata['cross_strand'].sum().item()
        if 'cross_chr' in g.edata:
            cross_chr_edges += g.edata['cross_chr'].sum().item()
        
        # Count the new gt_score (sum of skip, unknown, cross_strand)
        skip_attr = g.edata.get('skip', torch.zeros(g.num_edges()))
        unknown_attr = g.edata.get('unknown', torch.zeros(g.num_edges()))
        cross_strand_attr = g.edata.get('cross_strand', torch.zeros(g.num_edges()))
        gt_score_sum = (skip_attr + unknown_attr + cross_strand_attr)
        gt_bin_edges += (gt_score_sum > 0).sum().item()
        
        # Count cross_chr as malicious edges
        if 'cross_chr' in g.edata:
            malicious_edges += g.edata['cross_chr'].sum().item()
    
    # Analyze validation graphs
    for graph_name in valid_data:
        g = dgl.load_graphs(os.path.join(data_path, graph_name + '.dgl'))[0][0]
        total_edges += g.num_edges()
        valid_nodes += g.num_nodes()
        
        # Calculate node degrees for validation set
        in_degrees = g.in_degrees()
        out_degrees = g.out_degrees()
        total_degrees = in_degrees + out_degrees
        valid_degree_sum += total_degrees.sum().item()
        
        # Count individual attributes
        if 'skip' in g.edata:
            skip_edges += g.edata['skip'].sum().item()
        if 'unknown' in g.edata:
            unknown_edges += g.edata['unknown'].sum().item()
        if 'cross_strand' in g.edata:
            cross_strand_edges += g.edata['cross_strand'].sum().item()
        if 'cross_chr' in g.edata:
            cross_chr_edges += g.edata['cross_chr'].sum().item()
        
        # Count the new gt_score (sum of skip, unknown, cross_strand)
        skip_attr = g.edata.get('skip', torch.zeros(g.num_edges()))
        unknown_attr = g.edata.get('unknown', torch.zeros(g.num_edges()))
        cross_strand_attr = g.edata.get('cross_strand', torch.zeros(g.num_edges()))
        gt_score_sum = (skip_attr + unknown_attr + cross_strand_attr)
        gt_bin_edges += (gt_score_sum > 0).sum().item()
        
        # Count cross_chr as malicious edges
        if 'cross_chr' in g.edata:
            malicious_edges += g.edata['cross_chr'].sum().item()
    
    # Calculate total node statistics
    total_nodes = train_nodes + valid_nodes
    total_degree_sum = train_degree_sum + valid_degree_sum
    
    # Calculate average degrees
    avg_degree_total = total_degree_sum / total_nodes if total_nodes > 0 else 0
    avg_degree_train = train_degree_sum / train_nodes if train_nodes > 0 else 0
    avg_degree_valid = valid_degree_sum / valid_nodes if valid_nodes > 0 else 0
    
    print(f"\nDataset Statistics:")
    print(f"Total nodes: {total_nodes}")
    print(f"  Training nodes: {train_nodes} ({train_nodes/total_nodes*100:.1f}%)")
    print(f"  Validation nodes: {valid_nodes} ({valid_nodes/total_nodes*100:.1f}%)")
    print(f"Average node degree:")
    print(f"  Total dataset: {avg_degree_total:.2f}")
    print(f"  Training set: {avg_degree_train:.2f}")
    print(f"  Validation set: {avg_degree_valid:.2f}")
    print(f"Total edges: {total_edges}")
    print(f"Edges with skip=1: {skip_edges} ({(skip_edges/total_edges)*100:.2f}%)")
    print(f"Edges with unknown=1: {unknown_edges} ({(unknown_edges/total_edges)*100:.2f}%)")
    print(f"Edges with cross_strand=1: {cross_strand_edges} ({(cross_strand_edges/total_edges)*100:.2f}%)")
    print(f"Edges with cross_chr=1: {cross_chr_edges} ({(cross_chr_edges/total_edges)*100:.2f}%)")
    print(f"Edges with gt_score>0 (skip+unknown+cross_strand): {gt_bin_edges} ({(gt_bin_edges/total_edges)*100:.2f}%)")
    print(f"Edges with malicious=1 (cross_chr): {malicious_edges} ({(malicious_edges/total_edges)*100:.2f}%)\n")

    # Calculate loss weights for the new labels
    pos_weight_gt_bin = (total_edges - gt_bin_edges) / gt_bin_edges if gt_bin_edges > 0 else 1.0  # num_negatives / num_positives
    pos_weight_malicious = (total_edges - malicious_edges) / malicious_edges if malicious_edges > 0 else 1.0  # num_negatives / num_positives
    
    print(f"GT_BIN Loss weight (pos_weight): {pos_weight_gt_bin}")
    print(f"GT_BIN Positive class ratio: {gt_bin_edges/total_edges:.4f}")
    print(f"MALICIOUS Loss weight (pos_weight): {pos_weight_malicious}")
    print(f"MALICIOUS Positive class ratio: {malicious_edges/total_edges:.4f}")
    
    if gt_bin_edges > 0:
        print(f"GT_BIN Expected initial bias for balanced predictions: {-torch.log(torch.tensor((total_edges - gt_bin_edges) / gt_bin_edges)):.4f}")
    else:
        print(f"GT_BIN Expected initial bias: Cannot calculate (no positive examples)")
        exit()
    if malicious_edges > 0:
        print(f"MALICIOUS Expected initial bias for balanced predictions: {-torch.log(torch.tensor((total_edges - malicious_edges) / malicious_edges)):.4f}")
    else:
        print(f"MALICIOUS Expected initial bias: Cannot calculate (no positive examples)")
        print(f"WARNING: No positive examples found for 'malicious' class. This may cause training issues.")
        exit()

    model.to(device)
    if load_checkpoint:
        model.load_state_dict(torch.load(load_checkpoint, map_location=device))
        print(f"Loaded model from {load_checkpoint}")

    train(model, data_path, train_data, valid_data, device, config, diploid, mode=mode, loss_weight_gt_bin=pos_weight_gt_bin, loss_weight_malicious=pos_weight_malicious)

