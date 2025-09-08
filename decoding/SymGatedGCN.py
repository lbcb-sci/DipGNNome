import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

class PairNorm(nn.Module):
    """
    PairNorm normalization layer.
    Based on the paper: "PairNorm: Tackling Oversmoothing in GNNs"
    """
    def __init__(self, mode='PN', scale=1.0):
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

    def forward(self, x):
        if self.mode == 'PN':
            # PairNorm: x_i = x_i - mean(x) / std(x) * scale
            mean = x.mean(dim=0, keepdim=True)
            var = x.var(dim=0, keepdim=True, unbiased=False)
            x = (x - mean) / (var + 1e-8).sqrt() * self.scale
        elif self.mode == 'PN-SI':
            # PairNorm with scale-invariant: x_i = x_i - mean(x) / std(x) * scale
            mean = x.mean(dim=0, keepdim=True)
            var = x.var(dim=0, keepdim=True, unbiased=False)
            x = (x - mean) / (var + 1e-8).sqrt() * self.scale
        elif self.mode == 'PN-SCS':
            # PairNorm with scale and shift: x_i = (x_i - mean(x)) / std(x) * scale + shift
            mean = x.mean(dim=0, keepdim=True)
            var = x.var(dim=0, keepdim=True, unbiased=False)
            x = (x - mean) / (var + 1e-8).sqrt() * self.scale
        return x

class SymGatedGCNModel(nn.Module):
    def __init__(self, node_features, edge_features, hidden_features, hidden_edge_features, num_layers,
                 hidden_edge_scores, nb_pos_enc, nr_classes=1, dropout=None, strand_change_head=False, pred_dropout=0, norm='layer'):
        super().__init__()
        hidden_node_features = hidden_edge_features
        self.linear1_node = nn.Linear(node_features, hidden_node_features)
        self.linear2_node = nn.Linear(hidden_node_features, hidden_features)
        self.linear1_edge = nn.Linear(edge_features, hidden_edge_features)
        self.linear2_edge = nn.Linear(hidden_edge_features, hidden_features)
        self.gnn = SymGatedGCN_processor(num_layers, hidden_features, dropout=dropout, norm=norm)
        self.predictor = ScorePredictor(hidden_features, hidden_edge_scores, nr_classes, dropout=pred_dropout)
        #self.predictor = ScorePredictor_direct(hidden_features, hidden_edge_scores, nr_classes, dropout=pred_dropout)

        self.strand_change_head = strand_change_head
        if self.strand_change_head:
            self.strand_change_predictor = ScorePredictor(hidden_features, hidden_edge_scores, 1, dropout=pred_dropout)
    def forward(self, graph, x, e_):
        x = self.linear1_node(x)
        x = torch.relu(x)
        x = self.linear2_node(x)

        e = self.linear1_edge(e_)
        e = torch.relu(e)
        e = self.linear2_edge(e)

        x, e = self.gnn(graph, x, e)
        scores = self.predictor(graph, x, e) #, e_
        if self.strand_change_head:
            strand_change_scores = self.strand_change_predictor(graph, x, e)
            return scores, strand_change_scores
        else:
            return scores, None


class SymGatedGCNModelDoubleHead(nn.Module):
    """
    SymGatedGCN model with two separate heads for gt_bin and malicious prediction.
    """
    def __init__(self, node_features, edge_features, hidden_features, hidden_edge_features, num_layers,
                 hidden_edge_scores, nb_pos_enc, nr_classes=1, dropout=None, pred_dropout=0, norm='layer'):
        super().__init__()
        hidden_node_features = hidden_edge_features
        self.linear1_node = nn.Linear(node_features, hidden_node_features)
        self.linear2_node = nn.Linear(hidden_node_features, hidden_features)
        self.linear1_edge = nn.Linear(edge_features, hidden_edge_features)
        self.linear2_edge = nn.Linear(hidden_edge_features, hidden_features)
        self.gnn = SymGatedGCN_processor(num_layers, hidden_features, dropout=dropout, norm=norm)
        
        # Two separate heads for gt_bin and malicious prediction
        self.gt_bin_predictor = ScorePredictor(hidden_features, hidden_edge_scores, 1, dropout=pred_dropout)
        self.malicious_predictor = ScorePredictor(hidden_features, hidden_edge_scores, 1, dropout=pred_dropout)
    
    def forward(self, graph, x, e_):
        x = self.linear1_node(x)
        x = torch.relu(x)
        x = self.linear2_node(x)

        e = self.linear1_edge(e_)
        e = torch.relu(e)
        e = self.linear2_edge(e)

        x, e = self.gnn(graph, x, e)
        
        # Get predictions from both heads
        gt_bin_scores = self.gt_bin_predictor(graph, x, e)
        malicious_scores = self.malicious_predictor(graph, x, e)
        
        return gt_bin_scores, malicious_scores

class SymGatedGCNModel_direct(nn.Module):
    def __init__(self, node_features, edge_features, hidden_features, hidden_edge_features, num_layers,
                 hidden_edge_scores, nb_pos_enc, nr_classes=1, dropout=None, strand_change_head=False, pred_dropout=0, norm='layer'):
        super().__init__()
        hidden_node_features = hidden_edge_features
        self.linear1_node = nn.Linear(node_features, hidden_node_features)
        self.linear2_node = nn.Linear(hidden_node_features, hidden_features)
        self.linear1_edge = nn.Linear(edge_features, hidden_edge_features)
        self.linear2_edge = nn.Linear(hidden_edge_features, hidden_features)
        self.gnn = SymGatedGCN_processor(num_layers, hidden_features, dropout=dropout, norm=norm)
        self.predictor = ScorePredictor_direct(hidden_features, hidden_edge_scores, nr_classes, dropout=pred_dropout)

        self.strand_change_head = strand_change_head
        if self.strand_change_head:
            self.strand_change_predictor = ScorePredictor(hidden_features, hidden_edge_scores, 1, dropout=pred_dropout)
    def forward(self, graph, x, e_, x_full, e_full):
        x = self.linear1_node(x)
        x = torch.relu(x)
        x = self.linear2_node(x)

        e = self.linear1_edge(e_)
        e = torch.relu(e)
        e = self.linear2_edge(e)

        x, e = self.gnn(graph, x, e)
        scores = self.predictor(graph, x, e, x_full, e_full) #, e_
        if self.strand_change_head:
            strand_change_scores = self.strand_change_predictor(graph, x, e)
            return scores, strand_change_scores
        else:
            return scores, None
        
class CutGatedGCNModel_(nn.Module):
    def __init__(self, node_features, edge_features, hidden_features, hidden_edge_features, num_layers,
                 hidden_edge_scores, nb_pos_enc, nr_classes=1, dropout=None, strand_change_head=False, pred_dropout=0, norm='layer'):
        super().__init__()
        hidden_node_features = hidden_edge_features
        self.linear1_node = nn.Linear(node_features, hidden_node_features)
        self.linear2_node = nn.Linear(hidden_node_features, hidden_features)
        self.linear1_edge = nn.Linear(edge_features, hidden_edge_features)
        self.linear2_edge = nn.Linear(hidden_edge_features, hidden_features)
        self.gnn = SymGatedGCN_processor(num_layers, hidden_features, dropout=dropout, norm=norm)
        self.cut_predictor = ScorePredictor(hidden_features, hidden_edge_scores, 1, dropout=pred_dropout)
        self.bs_predictor = ScorePredictor(hidden_features, hidden_edge_scores, 1, dropout=pred_dropout)

    def forward(self, graph, x, e_):
        x = self.linear1_node(x)
        x = torch.relu(x)
        x = self.linear2_node(x)

        e = self.linear1_edge(e_)
        e = torch.relu(e)
        e = self.linear2_edge(e)

        x, e = self.gnn(graph, x, e)
        cut_pred = self.cut_predictor(graph, x, e) #, e_
        bs_pred = self.bs_predictor(graph, x, e) #, e_
        return cut_pred, bs_pred


class CutGatedGCNModel_2(nn.Module):
    def __init__(self, node_features, edge_features, hidden_features, hidden_edge_features, num_layers,
                 hidden_edge_scores, nb_pos_enc, nr_classes=1, dropout=None, strand_change_head=False, pred_dropout=0, norm='layer'):
        super().__init__()
        hidden_node_features = hidden_edge_features
        self.linear1_node = nn.Linear(node_features, hidden_node_features)
        self.linear2_node = nn.Linear(hidden_node_features, hidden_features)
        self.linear1_edge = nn.Linear(edge_features, hidden_edge_features)
        self.linear2_edge = nn.Linear(hidden_edge_features, hidden_features)
        self.gnn = SymGatedGCN_processor(num_layers, hidden_features, dropout=dropout, norm=norm)
        self.predictor = ScorePredictor(hidden_features, hidden_edge_scores, 2, dropout=pred_dropout)

    def forward(self, graph, x, e_):
        x = self.linear1_node(x)
        x = torch.relu(x)
        x = self.linear2_node(x)

        e = self.linear1_edge(e_)
        e = torch.relu(e)
        e = self.linear2_edge(e)

        x, e = self.gnn(graph, x, e)
        pred = self.predictor(graph, x, e) #, e_
        return pred
    

class CutGatedGCNModel(nn.Module):
    def __init__(self, node_features, edge_features, hidden_features, hidden_edge_features, num_layers,
                 hidden_edge_scores, nb_pos_enc, nr_classes=1, dropout=None, strand_change_head=False, pred_dropout=0, norm='layer'):
        super().__init__()
        hidden_node_features = hidden_edge_features
        self.linear1_node = nn.Linear(node_features, hidden_node_features)
        self.linear2_node = nn.Linear(hidden_node_features, hidden_features)
        self.linear1_edge = nn.Linear(edge_features, hidden_edge_features)
        self.linear2_edge = nn.Linear(hidden_edge_features, hidden_features)
        self.gnn = SymGatedGCN_processor(num_layers, hidden_features, dropout=dropout, norm=norm)
        self.predictor = ScorePredictor(hidden_features, hidden_edge_scores, 3, dropout=pred_dropout)

    def forward(self, graph, x, e_):
        x = self.linear1_node(x)
        x = torch.relu(x)
        x = self.linear2_node(x)

        e = self.linear1_edge(e_)
        e = torch.relu(e)
        e = self.linear2_edge(e)

        x, e = self.gnn(graph, x, e)
        pred = self.predictor(graph, x, e) #, e_

        return pred


class SymGatedGCN_processor(nn.Module):
    def __init__(self, num_layers, hidden_features, dropout=None, norm='layer'):
        super().__init__()
        self.convs = nn.ModuleList([
            SymGatedGCN(hidden_features, hidden_features, dropout, norm=norm) for _ in range(num_layers)
        ])

    def forward(self, graph, h, e):
        for i in range(len(self.convs)):
            h, e = self.convs[i](graph, h, e)
        return h, e


class SymGatedGCN(nn.Module):
    """
    Symmetric GatedGCN, based on the idea of  GatedGCN from 'Residual Gated Graph ConvNets'
    paper by Xavier Bresson and Thomas Laurent, ICLR 2018.
    https://arxiv.org/pdf/1711.07553v2.pdf
    """

    def __init__(self, in_channels, out_channels, dropout=None, residual=True, norm='layer'):
        super().__init__()
        if dropout:
            # print(f'Using dropout: {dropout}')
            self.dropout = dropout
        else:
            # print(f'Using dropout: 0.00')
            self.dropout = 0.0
        self.residual = residual
        self.norm = norm

        if in_channels != out_channels:
            self.residual = False

        dtype = torch.float32

        self.A_1 = nn.Linear(in_channels, out_channels, dtype=dtype)
        self.A_2 = nn.Linear(in_channels, out_channels, dtype=dtype)
        self.A_3 = nn.Linear(in_channels, out_channels, dtype=dtype)

        self.B_1 = nn.Linear(in_channels, out_channels, dtype=dtype)
        self.B_2 = nn.Linear(in_channels, out_channels, dtype=dtype)
        self.B_3 = nn.Linear(in_channels, out_channels, dtype=dtype)

        # Initialize normalization layers based on norm parameter
        if norm == 'batch':
            self.bn_h = nn.BatchNorm1d(out_channels, track_running_stats=True)
            self.bn_e = nn.BatchNorm1d(out_channels, track_running_stats=True)
        elif norm == 'layer':
            self.bn_h = nn.LayerNorm(out_channels)
            self.bn_e = nn.LayerNorm(out_channels)
        elif norm == 'pair':
            self.bn_h = PairNorm(mode='PN', scale=1.0)
            self.bn_e = PairNorm(mode='PN', scale=1.0)
        elif norm == 'pair_si':
            self.bn_h = PairNorm(mode='PN-SI', scale=1.0)
            self.bn_e = PairNorm(mode='PN-SI', scale=1.0)
        elif norm == 'pair_scs':
            self.bn_h = PairNorm(mode='PN-SCS', scale=1.0)
            self.bn_e = PairNorm(mode='PN-SCS', scale=1.0)
        else:
            raise ValueError(f"Unsupported normalization type: {norm}. Use 'batch', 'layer', 'pair', 'pair_si', or 'pair_scs'")
    def forward(self, g, h, e):
        """Return updated node representations."""
        with g.local_scope():
            h_in = h.clone()
            e_in = e.clone()

            g.ndata['h'] = h
            g.edata['e'] = e

            g.ndata['A1h'] = self.A_1(h)
            g.ndata['A2h'] = self.A_2(h)
            g.ndata['A3h'] = self.A_3(h)

            g.ndata['B1h'] = self.B_1(h)
            g.ndata['B2h'] = self.B_2(h)
            g.edata['B3e'] = self.B_3(e)

            g_reverse = dgl.reverse(g, copy_ndata=True, copy_edata=True)

            # Reference: https://github.com/graphdeeplearning/benchmarking-gnns/blob/master-dgl-0.6/layers/gated_gcn_layer.py

            # Forward-message passing
            g.apply_edges(fn.u_add_v('B1h', 'B2h', 'B12h'))
            e_ji = g.edata['B12h'] + g.edata['B3e']
            e_ji = self.bn_e(e_ji)
            e_ji = F.relu(e_ji)
            if self.residual:
                e_ji = e_ji + e_in
            g.edata['e_ji'] = e_ji
            g.edata['sigma_f'] = torch.sigmoid(g.edata['e_ji'])
            g.update_all(fn.u_mul_e('A2h', 'sigma_f', 'm_f'), fn.sum('m_f', 'sum_sigma_h_f'))
            g.update_all(fn.copy_e('sigma_f', 'm_f'), fn.sum('m_f', 'sum_sigma_f'))
            g.ndata['h_forward'] = g.ndata['sum_sigma_h_f'] / (g.ndata['sum_sigma_f'] + 1e-6)

            # Backward-message passing
            g_reverse.apply_edges(fn.u_add_v('B2h', 'B1h', 'B21h'))
            e_ik = g_reverse.edata['B21h'] + g_reverse.edata['B3e']
            e_ik = self.bn_e(e_ik)
            e_ik = F.relu(e_ik)
            if self.residual:
                e_ik = e_ik + e_in
            g_reverse.edata['e_ik'] = e_ik
            g_reverse.edata['sigma_b'] = torch.sigmoid(g_reverse.edata['e_ik'])
            g_reverse.update_all(fn.u_mul_e('A3h', 'sigma_b', 'm_b'), fn.sum('m_b', 'sum_sigma_h_b'))
            g_reverse.update_all(fn.copy_e('sigma_b', 'm_b'), fn.sum('m_b', 'sum_sigma_b'))
            g_reverse.ndata['h_backward'] = g_reverse.ndata['sum_sigma_h_b'] / (g_reverse.ndata['sum_sigma_b'] + 1e-6)

            h = g.ndata['A1h'] + g.ndata['h_forward'] + g_reverse.ndata['h_backward']

            h = self.bn_h(h)

            h = F.relu(h)

            if self.residual:
                h = h + h_in

            h = F.dropout(h, self.dropout, training=self.training)
            e = g.edata['e_ji']

            return h, e

class ScorePredictor(nn.Module):
    def __init__(self, in_features, hidden_edge_scores, nr_classes, dropout=0.0):
        super().__init__()
        self.W1 = nn.Linear(3 * in_features, in_features)
        self.W2 = nn.Linear(in_features, hidden_edge_scores)
        self.W3 = nn.Linear(hidden_edge_scores, nr_classes)
        self.dropout = nn.Dropout(p=dropout)
        
        # More aggressive initialization to center outputs around 0
        nn.init.xavier_uniform_(self.W1.weight, gain=0.1)
        nn.init.zeros_(self.W1.bias)
        nn.init.xavier_uniform_(self.W2.weight, gain=0.1)
        nn.init.zeros_(self.W2.bias)
        nn.init.xavier_uniform_(self.W3.weight, gain=0.01)
        nn.init.zeros_(self.W3.bias)

        #Initialize with small negative bias to start predictions closer to 0
        class_ratio = 0.1  # Adjust based on your positive class ratio
        initial_bias = -torch.log(torch.tensor((1 - class_ratio) / class_ratio))
        nn.init.constant_(self.W3.bias, initial_bias.item())

    def apply_edges(self, edges):
        data = torch.cat((edges.src['x'], edges.dst['x'], edges.data['e']), dim=1)
        h = self.W1(data)
        h = torch.relu(h)
        h = self.dropout(h)
        #####
        """h = self.Wa(h)
        h = torch.relu(h)
        h = self.dropout(h)
        h = self.Wb(h)
        h = torch.relu(h)
        h = self.dropout(h)"""
        #####
        h = self.W2(h)
        h = torch.relu(h)
        h = self.dropout(h)
        score = self.W3(h)
        #score = self.act(score)
        return {'score': score}

    def forward(self, graph, x, e):
        with graph.local_scope():
            graph.ndata['x'] = x
            graph.edata['e'] = e
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']

class ScorePredictor_direct(nn.Module):
    def __init__(self, in_features, hidden_edge_scores, nr_classes, edge_ftrs_dim=4, dropout=0.0):
        super().__init__()
        # Store edge features dimension and make input size flexible
        self.edge_ftrs_dim = edge_ftrs_dim
        self.W1 = nn.Linear((3 * in_features) + 2 * 10 + 2, hidden_edge_scores)
        #self.Wa = nn.Linear(hidden_edge_scores, hidden_edge_scores)
        #self.Wb = nn.Linear(hidden_edge_scores, hidden_edge_scores)
        self.W2 = nn.Linear(hidden_edge_scores, 32)
        self.W3 = nn.Linear(32, nr_classes)
        self.dropout = nn.Dropout(p=dropout)

    def apply_edges(self, edges):
        # Concatenate the original data with the additional edge features
        data = torch.cat((edges.src['x'], edges.dst['x'], edges.data['e'], edges.src['nodeftrs'], edges.dst['nodeftrs'], edges.data['edgeftrs']), dim=1)
        h = self.W1(data)
        h = torch.relu(h)
        h = self.dropout(h)
        #####
        """h = self.Wa(h)
        h = torch.relu(h)
        h = self.dropout(h)
        h = self.Wb(h)
        h = torch.relu(h)
        h = self.dropout(h)"""
        #####
        h = self.W2(h)
        h = torch.relu(h)
        h = self.dropout(h)
        score = self.W3(h)
        #score = self.act(score)
        return {'score': score}

    def forward(self, graph, x, e, nodeftrs, edgeftrs):
        with graph.local_scope():
            graph.ndata['x'] = x
            graph.edata['e'] = e
            graph.ndata['nodeftrs'] = nodeftrs
            graph.edata['edgeftrs'] = edgeftrs
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']

class StopNodePredictor(nn.Module):
    def __init__(self, dim_latent, hidden_edge_scores, output_dim, dropout=0.0):
        super().__init__()
        self.W1 = nn.Linear(dim_latent, dim_latent)
        self.W2 = nn.Linear(dim_latent, hidden_edge_scores)
        self.W3 = nn.Linear(hidden_edge_scores, output_dim)
        self.to_squeeze = output_dim == 1
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        h = self.W1(x)
        h = torch.relu(h)
        h = self.dropout(h)
        h = self.W2(h)
        h = torch.relu(h)
        h = self.dropout(h)
        score = self.W3(h)
        if self.to_squeeze:
            score = score.squeeze()
        return score

class YakSymGatedGCNModel(nn.Module):
    def __init__(self, node_features, edge_features, hidden_features, hidden_edge_features, num_layers,
                 hidden_edge_scores, nb_pos_enc, nr_classes=1, dropout=None, stop_head=False, norm='layer'):
        super().__init__()
        hidden_node_features = hidden_edge_features
        self.linear1_node = nn.Linear(node_features, hidden_node_features)
        self.linear2_node = nn.Linear(hidden_node_features, hidden_features)
        self.linear1_edge = nn.Linear(edge_features, hidden_edge_features)
        self.linear2_edge = nn.Linear(hidden_edge_features, hidden_features)
        self.gnn = SymGatedGCN_processor(num_layers, hidden_features, dropout=dropout, norm=norm)
        self.predictor = YakNodePredictor(hidden_features, hidden_edge_scores)

    def forward(self, graph, x, e):
        x = self.linear1_node(x)
        x = torch.relu(x)
        x = self.linear2_node(x)

        e = self.linear1_edge(e)
        e = torch.relu(e)
        e = self.linear2_edge(e)

        x, e = self.gnn(graph, x, e)
        scores = self.predictor(x)

        return scores


class SymGatedGCNYakModel(nn.Module):
    def __init__(self, node_features, edge_features, hidden_features, hidden_edge_features, num_layers,
                 hidden_edge_scores, nb_pos_enc, nr_classes=1, dropout=None, stop_head=False,
                 pred_dropout=0, norm='layer'):
        super().__init__()
        hidden_node_features = hidden_edge_features
        self.linear1_node = nn.Linear(node_features, hidden_node_features)
        self.linear2_node = nn.Linear(hidden_node_features, hidden_features)
        self.linear1_edge = nn.Linear(edge_features, hidden_edge_features)
        self.linear2_edge = nn.Linear(hidden_edge_features, hidden_features)
        self.gnn = SymGatedGCN_processor(num_layers, hidden_features, dropout=dropout, norm=norm)

        self.predictor = YakNodePredictor(hidden_features, hidden_edge_scores, dropout=pred_dropout)

    def forward(self, graph, x, e):
        x = self.linear1_node(x)
        x = torch.relu(x)
        x = self.linear2_node(x)

        e = self.linear1_edge(e)
        e = torch.relu(e)
        e = self.linear2_edge(e)

        x, e = self.gnn(graph, x, e)
        scores = self.predictor(x)
        return scores

class YakNodePredictor(nn.Module):
    def __init__(self, dim_latent, hidden_edge_scores, dropout=0.0):
        super().__init__()
        self.W1 = nn.Linear(dim_latent, dim_latent)
        self.W2 = nn.Linear(dim_latent, hidden_edge_scores)
        self.W3 = nn.Linear(hidden_edge_scores, 1)

        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        h = self.W1(x)
        h = torch.relu(h)
        h = self.dropout(h)
        h = self.W2(h)
        h = torch.relu(h)
        h = self.dropout(h)
        scores = self.W3(h)
        return scores