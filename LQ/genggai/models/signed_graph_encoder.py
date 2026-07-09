import torch
import torch.nn as nn
import torch.nn.functional as F


class SignedGraphEncoder(nn.Module):
    """
    7-node signed heterogeneous graph encoder
    nodes: [GT, GA, GV, Q, LT, LA, LV]
    """

    def __init__(self, dim=128, depth=2):
        super().__init__()

        self.dim = dim

        # message projection
        self.msg = nn.Linear(dim, dim)

        # positive / negative edge gating
        self.pos_gate = nn.Linear(dim * 2, 1)
        self.neg_gate = nn.Linear(dim * 2, 1)

        # update
        self.update = nn.GRUCell(dim, dim)

        self.depth = depth

    def forward(self, x):
        """
        x: [B, 7, D]
        """

        B, N, D = x.shape
        h = x

        # adjacency (fixed structure)
        edges_pos, edges_neg = self.build_edges()

        for _ in range(self.depth):

            new_h = h.clone()

            for i in range(N):
                msg_pos = 0
                msg_neg = 0

                for j in range(N):
                    if i == j:
                        continue

                    hi = h[:, i, :]
                    hj = h[:, j, :]

                    edge_feat = torch.cat([hi, hj], dim=-1)

                    # positive / negative weights
                    w_pos = torch.sigmoid(self.pos_gate(edge_feat))
                    w_neg = torch.sigmoid(self.neg_gate(edge_feat))

                    m = self.msg(hj)

                    if (i, j) in edges_pos:
                        msg_pos += w_pos * m

                    if (i, j) in edges_neg:
                        msg_neg += w_neg * m

                # ===== signed update =====
                agg = msg_pos - msg_neg

                new_h[:, i, :] = self.update(
                    agg[:, i, :] if agg.dim() == 3 else agg,
                    h[:, i, :]
                )

            h = new_h

        return h

    def build_edges(self):
        """
        define signed heterogeneous graph
        """

        # nodes:
        # 0 GT
        # 1 GA
        # 2 GV
        # 3 Q
        # 4 LT
        # 5 LA
        # 6 LV

        pos = set()
        neg = set()

        # ===== consistency edges (positive) =====
        pos.update([(0,1),(0,2),(1,2)])  # global consistency
        pos.update([(4,5),(4,6),(5,6)])  # local consistency
        pos.update([(3,0),(3,4)])        # Q alignment

        # ===== conflict edges (negative) =====
        neg.update([(0,5),(0,6)])        # GT vs local audio/video
        neg.update([(1,6)])              # GA vs LV
        neg.update([(2,5)])              # GV vs LA

        return pos, neg