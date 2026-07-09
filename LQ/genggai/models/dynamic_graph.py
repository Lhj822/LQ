import torch
from torch import nn
import torch.nn.functional as F


class SignedHeterogeneousGraphLayer(nn.Module):
    """
    显式异构有符号图传播层。

    默认节点顺序：
        0: GT
        1: GA
        2: GV
        3: Q
        4 ... 4+S-1: 文本局部情感事件节点
        4+S ... 4+2S-1: 音频局部情感事件节点
        4+2S ... 4+3S-1: 视觉局部情感事件节点

    三类异构关系：
        relation 0：跨模态同粒度关系
        relation 1：同模态跨粒度关系
        relation 2：查询锚定关系

    输入：
        nodes: [B, N, D]

    输出：
        updated_nodes: [B, N, D]
        relation_details: 每类关系对应的正边、负边权重
    """

    def __init__(self, dim=128, node_num=7, dropout=0.1):
        super().__init__()

        if node_num < 7:
            raise ValueError(
                f"SignedHeterogeneousGraphLayer 至少需要7个节点，"
                f"但收到 node_num={node_num}"
            )

        if (node_num - 4) % 3 != 0:
            raise ValueError(
                f"node_num必须满足4+3S的结构，"
                f"但收到 node_num={node_num}"
            )

        self.dim = dim
        self.node_num = node_num
        self.event_slot_num = (node_num - 4) // 3
        self.relation_num = 3

        # 三类关系分别使用独立的正、负消息映射
        self.positive_message = nn.ModuleList([
            nn.Linear(dim, dim, bias=False)
            for _ in range(self.relation_num)
        ])

        self.negative_message = nn.ModuleList([
            nn.Linear(dim, dim, bias=False)
            for _ in range(self.relation_num)
        ])

        # 每一类关系独立计算边的正负概率。
        # 输入由 target、source、|target-source|、target*source 拼接得到。
        self.sign_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim * 4, dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim, 2)
            )
            for _ in range(self.relation_num)
        ])

        # 每一类关系独立计算动态边强度
        self.strength_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim * 4, dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim, 1)
            )
            for _ in range(self.relation_num)
        ])

        # 三类关系的输出映射
        self.relation_output = nn.ModuleList([
            nn.Linear(dim, dim)
            for _ in range(self.relation_num)
        ])

        # 每一类关系对应一个可学习贡献系数
        self.relation_scale = nn.Parameter(
            torch.ones(self.relation_num)
        )

        # 负消息抑制强度，初始 sigmoid(-1.386) 约等于0.2
        self.negative_scale_logit = nn.Parameter(
            torch.tensor(-1.3862944)
        )

        # 外部事件博弈算子可以给图提供正/负边先验。
        # 初始值约为0.5，避免先验一开始完全压过图自身学习到的边。
        self.prior_scale_logit = nn.Parameter(
            torch.tensor(0.0)
        )

        self.input_norm = nn.LayerNorm(dim)
        self.message_norm = nn.LayerNorm(dim)
        self.output_norm = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )

        relation_masks = self._build_relation_masks()
        self.register_buffer(
            "relation_masks",
            relation_masks,
            persistent=True
        )

    def _build_relation_masks(self):
        """
        构造三类显式异构关系掩码。

        mask[target, source] = 1 表示：
        source节点可以向target节点传递消息。

        relation 0：跨模态同粒度/同层关系
            - GT、GA、GV之间互连
            - 文本/音频/视觉局部情感事件节点跨模态互连

        relation 1：同模态跨粒度关系
            - GT连接所有文本事件节点
            - GA连接所有音频事件节点
            - GV连接所有视觉事件节点

        relation 2：查询锚定关系
            - Q与所有非Q节点双向连接
        """

        masks = torch.zeros(
            self.relation_num,
            self.node_num,
            self.node_num,
            dtype=torch.float32
        )

        global_nodes = [0, 1, 2]
        query_index = 3

        text_start = 4
        audio_start = text_start + self.event_slot_num
        vision_start = audio_start + self.event_slot_num

        text_events = list(range(text_start, audio_start))
        audio_events = list(range(audio_start, vision_start))
        vision_events = list(range(vision_start, self.node_num))

        # relation 0：跨模态同粒度关系
        for target in global_nodes:
            for source in global_nodes:
                if target != source:
                    masks[0, target, source] = 1.0

        local_event_groups = [
            text_events,
            audio_events,
            vision_events
        ]

        for target_group_index, target_group in enumerate(
            local_event_groups
        ):
            for source_group_index, source_group in enumerate(
                local_event_groups
            ):
                if target_group_index == source_group_index:
                    continue

                for target in target_group:
                    for source in source_group:
                        masks[0, target, source] = 1.0

        # relation 1：同模态跨粒度关系
        cross_granularity_groups = [
            (0, text_events),
            (1, audio_events),
            (2, vision_events)
        ]

        for global_index, event_indices in cross_granularity_groups:
            for event_index in event_indices:
                masks[1, global_index, event_index] = 1.0
                masks[1, event_index, global_index] = 1.0

        # relation 2：查询锚定关系
        for node_index in range(self.node_num):
            if node_index == query_index:
                continue

            masks[2, query_index, node_index] = 1.0
            masks[2, node_index, query_index] = 1.0

        return masks

    @staticmethod
    def _row_normalize(adjacency, eps=1e-8):
        """
        对邻接矩阵按source维度归一化。

        adjacency:
            [B, target_node, source_node]
        """

        degree = adjacency.sum(dim=-1, keepdim=True)
        return adjacency / degree.clamp_min(eps)

    def _build_pair_features(self, nodes):
        """
        构造所有节点对特征。

        nodes:
            [B, N, D]

        返回：
            [B, N, N, 4D]

        第二维为target节点，第三维为source节点。
        """

        batch_size, node_num, dim = nodes.shape

        target_nodes = nodes.unsqueeze(2).expand(
            batch_size,
            node_num,
            node_num,
            dim
        )

        source_nodes = nodes.unsqueeze(1).expand(
            batch_size,
            node_num,
            node_num,
            dim
        )

        absolute_difference = torch.abs(
            target_nodes - source_nodes
        )

        element_product = target_nodes * source_nodes

        pair_features = torch.cat(
            [
                target_nodes,
                source_nodes,
                absolute_difference,
                element_product
            ],
            dim=-1
        )

        return pair_features

    def forward(self, nodes, positive_prior=None, negative_prior=None):
        if nodes.dim() != 3:
            raise ValueError(
                f"图节点输入必须为三维张量[B,N,D]，"
                f"实际形状为{tuple(nodes.shape)}"
            )

        if nodes.size(1) != self.node_num:
            raise ValueError(
                f"图节点数量必须为{self.node_num}，"
                f"实际为{nodes.size(1)}"
            )

        if nodes.size(2) != self.dim:
            raise ValueError(
                f"图节点维度必须为{self.dim}，"
                f"实际为{nodes.size(2)}"
            )

        residual = nodes
        normalized_nodes = self.input_norm(nodes)

        pair_features = self._build_pair_features(
            normalized_nodes
        )

        total_message = torch.zeros_like(nodes)

        positive_adjacencies = []
        negative_adjacencies = []

        relation_scales = F.softmax(
            self.relation_scale,
            dim=0
        )

        negative_scale = torch.sigmoid(
            self.negative_scale_logit
        )

        prior_scale = torch.sigmoid(
            self.prior_scale_logit
        )

        for relation_index in range(self.relation_num):
            relation_mask = self.relation_masks[
                relation_index
            ].unsqueeze(0)

            relation_mask = relation_mask.to(
                device=nodes.device,
                dtype=nodes.dtype
            )

            # 正负边概率
            sign_logits = self.sign_predictors[
                relation_index
            ](pair_features)

            sign_probabilities = F.softmax(
                sign_logits,
                dim=-1
            )

            positive_probability = sign_probabilities[
                ..., 0
            ]

            negative_probability = sign_probabilities[
                ..., 1
            ]

            # 动态边强度
            edge_strength = torch.sigmoid(
                self.strength_predictors[
                    relation_index
                ](pair_features)
            ).squeeze(-1)

            # 仅保留当前关系中显式定义的边
            positive_adjacency = (
                positive_probability
                * edge_strength
                * relation_mask
            )

            negative_adjacency = (
                negative_probability
                * edge_strength
                * relation_mask
            )

            if positive_prior is not None:
                positive_adjacency = (
                    positive_adjacency
                    + prior_scale
                    * positive_prior.to(
                        device=nodes.device,
                        dtype=nodes.dtype
                    )
                    * relation_mask
                )

            if negative_prior is not None:
                negative_adjacency = (
                    negative_adjacency
                    + prior_scale
                    * negative_prior.to(
                        device=nodes.device,
                        dtype=nodes.dtype
                    )
                    * relation_mask
                )

            positive_adjacency = self._row_normalize(
                positive_adjacency
            )

            negative_adjacency = self._row_normalize(
                negative_adjacency
            )

            positive_source = self.positive_message[
                relation_index
            ](normalized_nodes)

            negative_source = self.negative_message[
                relation_index
            ](normalized_nodes)

            positive_message = torch.bmm(
                positive_adjacency,
                positive_source
            )

            negative_message = torch.bmm(
                negative_adjacency,
                negative_source
            )

            signed_message = (
                positive_message
                - negative_scale * negative_message
            )

            signed_message = self.relation_output[
                relation_index
            ](signed_message)

            total_message = (
                total_message
                + relation_scales[relation_index]
                * signed_message
            )

            positive_adjacencies.append(
                positive_adjacency
            )

            negative_adjacencies.append(
                negative_adjacency
            )

        # 第一层残差更新
        nodes = self.message_norm(
            residual + self.dropout(total_message)
        )

        # 前馈网络残差更新
        nodes = self.output_norm(
            nodes + self.feed_forward(nodes)
        )

        relation_details = {
            "positive_adjacency": torch.stack(
                positive_adjacencies,
                dim=1
            ),
            "negative_adjacency": torch.stack(
                negative_adjacencies,
                dim=1
            ),
            "relation_scales": relation_scales,
            "negative_scale": negative_scale,
            "prior_scale": prior_scale
        }

        return nodes, relation_details


class QueryAnchoredDynamicGraphEncoder(nn.Module):
    """
    查询锚定的显式异构有符号图编码器。

    默认输出：
        graph_feature: [B, 128]

    return_details=True时输出：
        graph_feature: [B, 128]
        updated_nodes: [B, N, 128]
        graph_details: 图关系信息
    """

    def __init__(
        self,
        dim=128,
        node_num=7,
        depth=2,
        heads=4,
        dropout=0.1
    ):
        super().__init__()

        if node_num < 7:
            raise ValueError(
                f"QueryAnchoredDynamicGraphEncoder 至少需要7个节点，"
                f"但收到 node_num={node_num}"
            )

        if (node_num - 4) % 3 != 0:
            raise ValueError(
                f"node_num必须满足4+3S的结构，"
                f"但收到 node_num={node_num}"
            )

        self.dim = dim
        self.node_num = node_num
        self.depth = depth
        self.query_index = 3

        # heads参数保留，保证与原调用接口兼容
        self.heads = heads

        # 节点类型嵌入
        self.node_type_embedding = nn.Parameter(
            torch.empty(1, node_num, dim)
        )

        nn.init.normal_(
            self.node_type_embedding,
            mean=0.0,
            std=0.02
        )

        self.input_norm = nn.LayerNorm(dim)

        self.layers = nn.ModuleList([
            SignedHeterogeneousGraphLayer(
                dim=dim,
                node_num=node_num,
                dropout=dropout
            )
            for _ in range(depth)
        ])

        self.query_output = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

        self.final_norm = nn.LayerNorm(dim)

    def forward(
        self,
        nodes,
        return_details=False,
        positive_prior=None,
        negative_prior=None
    ):
        if nodes.dim() != 3:
            raise ValueError(
                f"输入必须为[B,N,{self.dim}]，"
                f"实际为{tuple(nodes.shape)}"
            )

        if nodes.size(1) != self.node_num:
            raise ValueError(
                f"节点数量必须为{self.node_num}，"
                f"实际为{nodes.size(1)}"
            )

        if nodes.size(2) != self.dim:
            raise ValueError(
                f"节点维度必须为{self.dim}，"
                f"实际为{nodes.size(2)}"
            )

        nodes = nodes + self.node_type_embedding
        nodes = self.input_norm(nodes)

        layer_details = []

        for layer in self.layers:
            nodes, current_details = layer(
                nodes,
                positive_prior=positive_prior,
                negative_prior=negative_prior
            )
            layer_details.append(current_details)

        updated_query = nodes[:, self.query_index, :]

        graph_feature = self.final_norm(
            updated_query + self.query_output(updated_query)
        )

        if return_details:
            graph_details = {
                "layers": layer_details
            }

            return graph_feature, nodes, graph_details

        return graph_feature