import torch
from torch import nn
import torch.nn.functional as F

from .GlobalFusionClassifier import GlobalFusionClassifier
from .LocalFusionClassifier import LocalFusionClassifier
from .dynamic_graph import QueryAnchoredDynamicGraphEncoder


class TriModalEventGameOperator(nn.Module):
    """
    轻量级三模态事件博弈算子。

    输入同一槽位上的文本、音频、视觉事件节点，学习三类作用：
        support：模态之间的相互支持，作为正边先验；
        veto：模态之间的相互否决，作为负边先验；
        reserve：单个模态的特有信息保留强度。

    该模块只在事件槽级别工作，不在原始时间步上做全量两两交互，
    因此额外开销相对可控。
    """

    def __init__(self, dim=128, event_slot_num=5, dropout=0.1):
        super().__init__()

        self.dim = dim
        self.event_slot_num = event_slot_num
        self.modality_num = 3

        pair_dim = dim * 4

        self.support_score = nn.Sequential(
            nn.Linear(pair_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

        self.veto_score = nn.Sequential(
            nn.Linear(pair_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

        self.reserve_score = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )

        self.update_gate = nn.Sequential(
            nn.Linear(dim * 4 + 3, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

        self.update_projection = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

        self.output_norm = nn.LayerNorm(dim)

    def _pair_features(self, events):
        target = events.unsqueeze(3).expand(
            -1,
            -1,
            -1,
            self.modality_num,
            -1
        )
        source = events.unsqueeze(2).expand(
            -1,
            -1,
            self.modality_num,
            -1,
            -1
        )
        return torch.cat(
            [
                target,
                source,
                torch.abs(target - source),
                target * source
            ],
            dim=-1
        )

    def forward(self, text_events, audio_events, vision_events, active_masks):
        events = torch.stack(
            [text_events, audio_events, vision_events],
            dim=2
        )
        masks = torch.stack(active_masks, dim=2)
        pair_mask = masks.unsqueeze(3) * masks.unsqueeze(2)

        identity = torch.eye(
            self.modality_num,
            device=events.device,
            dtype=events.dtype
        ).view(1, 1, self.modality_num, self.modality_num)
        pair_mask = pair_mask * (1.0 - identity)

        pair_features = self._pair_features(events)

        support = self.support_score(pair_features).squeeze(-1)
        veto = self.veto_score(pair_features).squeeze(-1)

        support = support * pair_mask
        veto = veto * pair_mask

        reserve = self.reserve_score(events).squeeze(-1) * masks

        support_denominator = support.sum(
            dim=-1,
            keepdim=True
        ).clamp_min(1e-6)
        veto_denominator = veto.sum(
            dim=-1,
            keepdim=True
        ).clamp_min(1e-6)

        support_context = torch.sum(
            support.unsqueeze(-1) * events.unsqueeze(2),
            dim=3
        ) / support_denominator

        veto_context = torch.sum(
            veto.unsqueeze(-1) * events.unsqueeze(2),
            dim=3
        ) / veto_denominator

        consensus_strength = (
            support.sum(dim=(2, 3))
            / pair_mask.sum(dim=(2, 3)).clamp_min(1.0)
        )
        conflict_strength = (
            veto.sum(dim=(2, 3))
            / pair_mask.sum(dim=(2, 3)).clamp_min(1.0)
        )
        reserve_strength = (
            reserve.sum(dim=2)
            / masks.sum(dim=2).clamp_min(1.0)
        )

        game_complexity = torch.clamp(
            conflict_strength
            + torch.abs(consensus_strength - reserve_strength),
            min=0.0,
            max=1.0
        )

        gate_input = torch.cat(
            [
                events,
                support_context,
                veto_context,
                events * reserve.unsqueeze(-1),
                support.sum(dim=-1, keepdim=True),
                veto.sum(dim=-1, keepdim=True),
                reserve.unsqueeze(-1)
            ],
            dim=-1
        )

        update_gate = self.update_gate(gate_input)
        game_update = (
            support_context
            - veto_context
            + reserve.unsqueeze(-1) * events
        )
        updated_events = self.output_norm(
            events
            + update_gate
            * self.update_projection(game_update)
        )
        updated_events = updated_events * masks.unsqueeze(-1)

        consensus_event = (
            support_context * masks.unsqueeze(-1)
        ).sum(dim=2) / masks.sum(dim=2, keepdim=True).clamp_min(1.0)

        specific_event = (
            reserve.unsqueeze(-1) * events
        ).sum(dim=2) / reserve.sum(dim=2, keepdim=True).clamp_min(1e-6)

        conflict_event = (
            veto_context * masks.unsqueeze(-1)
        ).sum(dim=2) / masks.sum(dim=2, keepdim=True).clamp_min(1.0)

        details = {
            'support': support,
            'veto': veto,
            'reserve': reserve,
            'consensus_strength': consensus_strength,
            'conflict_strength': conflict_strength,
            'reserve_strength': reserve_strength,
            'game_complexity': game_complexity,
            'consensus_event': consensus_event,
            'specific_event': specific_event,
            'conflict_event': conflict_event,
            'game_update_gate': update_gate.squeeze(-1)
        }

        return (
            updated_events[:, :, 0, :],
            updated_events[:, :, 1, :],
            updated_events[:, :, 2, :],
            details
        )


class Gate_fusion(nn.Module):
    """
    查询锚定的情感事件有符号异构图融合模型。

    Global Path 只提取 GT、GA、GV、Q。
    Local Path 不再进行均值池化，而是为每个模态生成多个查询引导的
    无窗口局部情感事件节点：LT_events、LA_events、LV_events。

    图节点顺序：
        0: GT
        1: GA
        2: GV
        3: Q
        4 ... 4+S-1: 文本局部情感事件节点
        4+S ... 4+2S-1: 音频局部情感事件节点
        4+2S ... 4+3S-1: 视觉局部情感事件节点
    """

    def __init__(self, args):
        super().__init__()

        self.Global_path = GlobalFusionClassifier(args)
        self.Local_path = LocalFusionClassifier(args)

        self.event_slot_num = getattr(
            args.model,
            'event_slot_num',
            3
        )

        self.iteration_num = max(
            1,
            getattr(
                args.model,
                'iteration_num',
                2
            )
        )

        self.node_num = 4 + 3 * self.event_slot_num

        self.graph_encoder = QueryAnchoredDynamicGraphEncoder(
            dim=128,
            node_num=self.node_num,
            depth=2,
            heads=4,
            dropout=0.1
        )

        self.event_game_operator = TriModalEventGameOperator(
            dim=128,
            event_slot_num=self.event_slot_num,
            dropout=0.1
        )

        self.contrastive_projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128)
        )

        self.contrastive_temperature = 0.1

        self.last_contrastive_loss = torch.tensor(0.0)
        self.last_graph_regularization_loss = torch.tensor(0.0)
        self.last_event_regularization_loss = torch.tensor(0.0)
        self.last_counterfactual_loss = torch.tensor(0.0)
        self.last_counterfactual_effects = None
        self.last_bipolar_prediction = None
        self.last_bipolar_gate = None

        self.graph_gate = nn.Sequential(
            nn.Linear(128 * 4, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.global_refine = nn.Sequential(
            nn.Linear(128 * 3, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128)
        )

        self.local_event_score = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

        self.query_enhance = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128)
        )

        self.event_refresh_gate = nn.Sequential(
            nn.Linear(128 * 3, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.fusion_norm = nn.LayerNorm(128)

        self.regression = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

        self.text_prior_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

        self.positive_evidence_score = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

        self.negative_evidence_score = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

        self.positive_strength_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Softplus()
        )

        self.negative_strength_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Softplus()
        )

        self.bipolar_fusion_gate = nn.Sequential(
            nn.Linear(128 * 3 + 4, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def _event_ranges(self):
        text_start = 4
        audio_start = text_start + self.event_slot_num
        vision_start = audio_start + self.event_slot_num

        return {
            'text': range(text_start, audio_start),
            'audio': range(audio_start, vision_start),
            'vision': range(vision_start, self.node_num),
            'all': range(text_start, self.node_num)
        }

    def _symmetric_info_nce(self, feature_a, feature_b):
        """双向 InfoNCE 损失。"""

        projected_a = self.contrastive_projection(feature_a)
        projected_b = self.contrastive_projection(feature_b)

        projected_a = F.normalize(projected_a, p=2, dim=-1)
        projected_b = F.normalize(projected_b, p=2, dim=-1)

        batch_size = projected_a.size(0)

        if batch_size == 1:
            cosine_similarity = F.cosine_similarity(
                projected_a,
                projected_b,
                dim=-1
            )
            return (1.0 - cosine_similarity).mean()

        logits_ab = torch.matmul(
            projected_a,
            projected_b.transpose(0, 1)
        ) / self.contrastive_temperature

        logits_ba = logits_ab.transpose(0, 1)

        labels = torch.arange(
            batch_size,
            device=feature_a.device
        )

        loss_ab = F.cross_entropy(logits_ab, labels)
        loss_ba = F.cross_entropy(logits_ba, labels)

        return 0.5 * (loss_ab + loss_ba)

    def _calculate_contrastive_loss(self, updated_nodes):
        """
        同模态跨粒度对比学习：
            GT' <-> mean(LT_events')
            GA' <-> mean(LA_events')
            GV' <-> mean(LV_events')
        """

        ranges = self._event_ranges()

        GT = updated_nodes[:, 0, :]
        GA = updated_nodes[:, 1, :]
        GV = updated_nodes[:, 2, :]

        event_mask = getattr(self, '_current_event_mask', None)
        text_mask = None
        audio_mask = None
        vision_mask = None
        if event_mask is not None:
            text_mask = event_mask[:, :self.event_slot_num]
            audio_mask = event_mask[:, self.event_slot_num:2 * self.event_slot_num]
            vision_mask = event_mask[:, 2 * self.event_slot_num:]

        LT = self._masked_event_mean(updated_nodes, ranges['text'], text_mask)
        LA = self._masked_event_mean(updated_nodes, ranges['audio'], audio_mask)
        LV = self._masked_event_mean(updated_nodes, ranges['vision'], vision_mask)

        text_contrastive_loss = self._symmetric_info_nce(GT, LT)
        audio_contrastive_loss = self._symmetric_info_nce(GA, LA)
        vision_contrastive_loss = self._symmetric_info_nce(GV, LV)

        return (
            text_contrastive_loss
            + audio_contrastive_loss
            + vision_contrastive_loss
        ) / 3.0

    def _calculate_graph_regularization_loss(self, graph_details):
        """正负边互斥约束。"""

        layer_losses = []

        for layer_detail in graph_details['layers']:
            positive_adjacency = layer_detail['positive_adjacency']
            negative_adjacency = layer_detail['negative_adjacency']

            overlap_loss = (
                positive_adjacency * negative_adjacency
            ).mean()

            layer_losses.append(overlap_loss)

        if len(layer_losses) == 0:
            return torch.zeros(
                (),
                device=next(self.parameters()).device
            )

        return torch.stack(layer_losses).mean()

    def _graph_enhanced_gate(self, updated_nodes):
        """
        使用图传播后的全局节点、局部情感事件节点和查询节点生成最终表示。
        """

        ranges = self._event_ranges()

        global_nodes = updated_nodes[:, [0, 1, 2], :]
        local_event_nodes = updated_nodes[:, list(ranges['all']), :]

        query_graph = updated_nodes[:, 3, :]
        all_graph = updated_nodes.mean(dim=1)

        global_mean = global_nodes.mean(dim=1)

        global_graph = self.global_refine(
            global_nodes.reshape(global_nodes.size(0), -1)
        ) + global_mean

        query_for_events = query_graph.unsqueeze(1).expand_as(
            local_event_nodes
        )

        local_event_logits = self.local_event_score(
            torch.cat(
                [local_event_nodes, query_for_events],
                dim=-1
            )
        ).squeeze(-1)

        event_mask = getattr(self, '_current_event_mask', None)
        if event_mask is not None:
            local_event_logits = local_event_logits.masked_fill(
                event_mask <= 0,
                -1e4
            )

        local_event_weight = F.softmax(
            local_event_logits,
            dim=-1
        )

        local_graph = torch.sum(
            local_event_weight.unsqueeze(-1) * local_event_nodes,
            dim=1
        )

        gate_input = torch.cat(
            [query_graph, global_graph, local_graph, all_graph],
            dim=-1
        )

        graph_weight = self.graph_gate(gate_input)

        gated_feature = (
            graph_weight * global_graph
            + (1.0 - graph_weight) * local_graph
        )

        query_context = self.query_enhance(
            torch.cat([query_graph, all_graph], dim=-1)
        )

        fused_feature = self.fusion_norm(
            gated_feature + query_graph + query_context
        )

        gate_details = {
            'graph_weight': graph_weight,
            'global_graph': global_graph,
            'local_graph': local_graph,
            'query_graph': query_graph,
            'local_event_weight': local_event_weight
        }

        return fused_feature, gate_details

    def _text_anchored_bipolar_branch(
        self,
        updated_nodes,
        fused_feature
    ):
        """
        文本锚定双极性证据分支。

        该分支以图传播后的文本全局节点GT'生成文本先验，
        再以查询节点Q'为锚，从所有图节点中分别聚合正向证据
        与负向证据。最终用“文本先验 + 正向证据 - 负向证据”
        生成双极性预测，用于修正主图分支输出。
        """

        text_node = updated_nodes[:, 0, :]
        query_node = updated_nodes[:, 3, :]

        query_for_nodes = query_node.unsqueeze(1).expand_as(
            updated_nodes
        )

        evidence_input = torch.cat(
            [updated_nodes, query_for_nodes],
            dim=-1
        )

        positive_logits = self.positive_evidence_score(
            evidence_input
        ).squeeze(-1)

        negative_logits = self.negative_evidence_score(
            evidence_input
        ).squeeze(-1)

        positive_weight = F.softmax(
            positive_logits,
            dim=-1
        )

        negative_weight = F.softmax(
            negative_logits,
            dim=-1
        )

        positive_context = torch.sum(
            positive_weight.unsqueeze(-1) * updated_nodes,
            dim=1
        )

        negative_context = torch.sum(
            negative_weight.unsqueeze(-1) * updated_nodes,
            dim=1
        )

        text_prior = self.text_prior_head(text_node)
        positive_strength = self.positive_strength_head(
            positive_context
        )
        negative_strength = self.negative_strength_head(
            negative_context
        )

        bipolar_prediction = (
            text_prior
            + positive_strength
            - negative_strength
        )

        conflict_strength = torch.minimum(
            positive_strength,
            negative_strength
        )

        bipolar_gate_input = torch.cat(
            [
                fused_feature,
                positive_context,
                negative_context,
                text_prior,
                positive_strength,
                negative_strength,
                conflict_strength
            ],
            dim=-1
        )

        bipolar_gate = self.bipolar_fusion_gate(
            bipolar_gate_input
        )

        bipolar_details = {
            'text_prior': text_prior,
            'positive_strength': positive_strength,
            'negative_strength': negative_strength,
            'conflict_strength': conflict_strength,
            'positive_context': positive_context,
            'negative_context': negative_context,
            'positive_evidence_weight': positive_weight,
            'negative_evidence_weight': negative_weight,
            'bipolar_prediction': bipolar_prediction,
            'bipolar_gate': bipolar_gate
        }

        return bipolar_prediction, bipolar_gate, bipolar_details

    def _predict_from_updated_nodes(self, updated_nodes):
        fused_feature, gate_details = self._graph_enhanced_gate(
            updated_nodes
        )

        graph_output = self.regression(fused_feature)

        (
            bipolar_prediction,
            bipolar_gate,
            bipolar_details
        ) = self._text_anchored_bipolar_branch(
            updated_nodes,
            fused_feature
        )

        output = (
            (1.0 - bipolar_gate) * graph_output
            + bipolar_gate * bipolar_prediction
        )

        prediction_details = {
            'fused_feature': fused_feature,
            'graph_output': graph_output,
            'output': output,
            'gate_details': gate_details,
            'bipolar_details': bipolar_details
        }

        return output, prediction_details

    def _calculate_counterfactual_effects(
        self,
        updated_nodes,
        original_output
    ):
        """
        每次前向都计算软反事实事件贡献。

        对每个局部情感事件节点进行软抑制：
        将该节点替换为整体图上下文，重新执行图增强门控和回归头，
        以预测变化幅度衡量该事件节点的决策贡献。
        """

        ranges = self._event_ranges()
        event_indices = list(ranges['all'])

        neutral_node = updated_nodes.mean(dim=1)
        effects = []

        for event_index in event_indices:
            intervened_nodes = updated_nodes.clone()
            intervened_nodes[:, event_index, :] = neutral_node

            counterfactual_output, _ = self._predict_from_updated_nodes(
                intervened_nodes
            )

            effect = torch.abs(
                original_output - counterfactual_output
            )

            effects.append(effect)

        counterfactual_effects = torch.cat(
            effects,
            dim=1
        )

        event_mask = getattr(self, '_current_event_mask', None)
        if event_mask is not None:
            counterfactual_effects = counterfactual_effects * event_mask
            active_denominator = event_mask.sum().clamp_min(1.0)
            counterfactual_loss = (
                F.relu(0.01 - counterfactual_effects)
                * event_mask
            ).sum() / active_denominator
        else:
            # 仅作为贡献监测项：鼓励事件节点不要全部变成零贡献。
            counterfactual_loss = F.relu(
                0.01 - counterfactual_effects
            ).mean()

        return counterfactual_effects, counterfactual_loss
    @staticmethod
    def _global_nodes_from_updated(updated_nodes):
        return {
            'GT': updated_nodes[:, 0, :],
            'GA': updated_nodes[:, 1, :],
            'GV': updated_nodes[:, 2, :],
            'Q': updated_nodes[:, 3, :]
        }

    @staticmethod
    def _split_event_nodes(event_nodes):
        return {
            'LT_events': event_nodes[0],
            'LA_events': event_nodes[1],
            'LV_events': event_nodes[2]
        }

    def _compose_node_tensor(self, global_nodes, local_nodes):
        return torch.cat(
            [
                global_nodes['GT'].unsqueeze(1),
                global_nodes['GA'].unsqueeze(1),
                global_nodes['GV'].unsqueeze(1),
                global_nodes['Q'].unsqueeze(1),
                local_nodes['LT_events'],
                local_nodes['LA_events'],
                local_nodes['LV_events']
            ],
            dim=1
        )

    def _build_game_edge_priors(self, game_details, device, dtype):
        positive_prior = torch.zeros(
            game_details['support'].size(0),
            self.node_num,
            self.node_num,
            device=device,
            dtype=dtype
        )
        negative_prior = torch.zeros_like(positive_prior)

        ranges = self._event_ranges()
        event_groups = [
            list(ranges['text']),
            list(ranges['audio']),
            list(ranges['vision'])
        ]

        support = game_details['support'].to(device=device, dtype=dtype)
        veto = game_details['veto'].to(device=device, dtype=dtype)

        for slot_index in range(self.event_slot_num):
            for target_modality, target_group in enumerate(event_groups):
                for source_modality, source_group in enumerate(event_groups):
                    if target_modality == source_modality:
                        continue

                    target_index = target_group[slot_index]
                    source_index = source_group[slot_index]

                    positive_prior[:, target_index, source_index] = (
                        support[
                            :,
                            slot_index,
                            target_modality,
                            source_modality
                        ]
                    )
                    negative_prior[:, target_index, source_index] = (
                        veto[
                            :,
                            slot_index,
                            target_modality,
                            source_modality
                        ]
                    )

        return positive_prior, negative_prior

    def _apply_event_game(self, local_nodes):
        (
            text_events,
            audio_events,
            vision_events,
            game_details
        ) = self.event_game_operator(
            local_nodes['LT_events'],
            local_nodes['LA_events'],
            local_nodes['LV_events'],
            [
                local_nodes['text_active_mask'],
                local_nodes['audio_active_mask'],
                local_nodes['vision_active_mask']
            ]
        )

        local_nodes = dict(local_nodes)
        local_nodes['LT_events'] = text_events
        local_nodes['LA_events'] = audio_events
        local_nodes['LV_events'] = vision_events
        local_nodes['LT'] = self._masked_local_mean(
            text_events,
            local_nodes['text_active_mask']
        )
        local_nodes['LA'] = self._masked_local_mean(
            audio_events,
            local_nodes['audio_active_mask']
        )
        local_nodes['LV'] = self._masked_local_mean(
            vision_events,
            local_nodes['vision_active_mask']
        )

        positive_prior, negative_prior = self._build_game_edge_priors(
            game_details,
            device=text_events.device,
            dtype=text_events.dtype
        )

        return local_nodes, game_details, positive_prior, negative_prior

    def _event_mask_from_local_nodes(self, local_nodes):
        return torch.cat(
            [
                local_nodes['text_active_mask'],
                local_nodes['audio_active_mask'],
                local_nodes['vision_active_mask']
            ],
            dim=1
        )

    def _apply_event_mask(self, updated_nodes, event_mask):
        if event_mask is None:
            return updated_nodes

        masked_nodes = updated_nodes.clone()
        ranges = self._event_ranges()
        event_indices = list(ranges['all'])
        masked_nodes[:, event_indices, :] = (
            masked_nodes[:, event_indices, :]
            * event_mask.unsqueeze(-1)
        )
        return masked_nodes

    def _masked_event_mean(self, updated_nodes, indices, event_mask=None):
        event_nodes = updated_nodes[:, list(indices), :]
        if event_mask is None:
            return event_nodes.mean(dim=1)

        mask = event_mask[:, :event_nodes.size(1)]
        denominator = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (event_nodes * mask.unsqueeze(-1)).sum(dim=1) / denominator

    @staticmethod
    def _active_count_to_mask(active_count, event_slot_num):
        slot_index = torch.arange(
            event_slot_num,
            device=active_count.device
        ).unsqueeze(0)
        return (slot_index < active_count.unsqueeze(1)).to(active_count.dtype)

    @staticmethod
    def _stabilize_active_count(previous_count, current_count):
        previous_level = torch.where(
            previous_count <= 1.5,
            torch.zeros_like(previous_count),
            torch.where(
                previous_count <= 3.5,
                torch.ones_like(previous_count),
                torch.full_like(previous_count, 2.0)
            )
        )
        current_level = torch.where(
            current_count <= 1.5,
            torch.zeros_like(current_count),
            torch.where(
                current_count <= 3.5,
                torch.ones_like(current_count),
                torch.full_like(current_count, 2.0)
            )
        )
        stable_level = torch.clamp(
            current_level,
            min=previous_level - 1.0,
            max=previous_level + 1.0
        )
        return torch.where(
            stable_level <= 0.5,
            torch.ones_like(stable_level),
            torch.where(
                stable_level <= 1.5,
                torch.full_like(stable_level, 3.0),
                torch.full_like(stable_level, 5.0)
            )
        )

    def _stabilize_local_nodes(self, current_local_nodes, previous_local_nodes=None):
        if previous_local_nodes is None:
            return current_local_nodes

        for prefix, event_key, attention_key in [
            ('text', 'LT_events', 'text_event_attention'),
            ('audio', 'LA_events', 'audio_event_attention'),
            ('vision', 'LV_events', 'vision_event_attention')
        ]:
            count_key = f'{prefix}_active_count'
            mask_key = f'{prefix}_active_mask'
            stable_count = self._stabilize_active_count(
                previous_local_nodes[count_key],
                current_local_nodes[count_key]
            )
            stable_mask = self._active_count_to_mask(
                stable_count,
                self.event_slot_num
            )
            current_local_nodes[count_key] = stable_count
            current_local_nodes[mask_key] = stable_mask
            current_local_nodes[event_key] = (
                current_local_nodes[event_key]
                * stable_mask.unsqueeze(-1)
            )
            current_local_nodes[attention_key] = (
                current_local_nodes[attention_key]
                * stable_mask.unsqueeze(-1)
            )

        current_local_nodes['LT'] = self._masked_local_mean(
            current_local_nodes['LT_events'],
            current_local_nodes['text_active_mask']
        )
        current_local_nodes['LA'] = self._masked_local_mean(
            current_local_nodes['LA_events'],
            current_local_nodes['audio_active_mask']
        )
        current_local_nodes['LV'] = self._masked_local_mean(
            current_local_nodes['LV_events'],
            current_local_nodes['vision_active_mask']
        )
        return current_local_nodes

    @staticmethod
    def _masked_local_mean(event_nodes, active_mask):
        denominator = active_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (event_nodes * active_mask.unsqueeze(-1)).sum(dim=1) / denominator

    def _refresh_local_events(self, graph_event_nodes, new_local_nodes, query_node):
        new_event_nodes = torch.cat(
            [
                new_local_nodes['LT_events'],
                new_local_nodes['LA_events'],
                new_local_nodes['LV_events']
            ],
            dim=1
        )

        query_for_events = query_node.unsqueeze(1).expand_as(
            graph_event_nodes
        )

        refresh_gate = self.event_refresh_gate(
            torch.cat(
                [
                    graph_event_nodes,
                    new_event_nodes,
                    query_for_events
                ],
                dim=-1
            )
        )

        event_mask = self._event_mask_from_local_nodes(new_local_nodes)

        refreshed_events = (
            refresh_gate * graph_event_nodes
            + (1.0 - refresh_gate) * new_event_nodes
        )
        refreshed_events = refreshed_events * event_mask.unsqueeze(-1)
        refresh_gate = refresh_gate * event_mask.unsqueeze(-1)

        text_events, audio_events, vision_events = torch.split(
            refreshed_events,
            self.event_slot_num,
            dim=1
        )

        local_nodes = dict(new_local_nodes)
        local_nodes.update({
            'LT_events': text_events,
            'LA_events': audio_events,
            'LV_events': vision_events,
            'LT': self._masked_local_mean(
                text_events,
                new_local_nodes['text_active_mask']
            ),
            'LA': self._masked_local_mean(
                audio_events,
                new_local_nodes['audio_active_mask']
            ),
            'LV': self._masked_local_mean(
                vision_events,
                new_local_nodes['vision_active_mask']
            ),
            'event_refresh_gate': refresh_gate
        })

        return local_nodes

    def forward(self, video, audio, text, return_nodes=False):
        text_embedding = self.Global_path.encode_text(text)

        (
            global_nodes,
            _
        ) = self.Global_path.forward_from_text_embedding(
            video,
            audio,
            text_embedding,
            return_sequences=True
        )

        local_encoded_features = self.Local_path.encode_from_text_embedding(
            video,
            audio,
            text_embedding
        )

        local_nodes = self.Local_path.extract_events_from_encoded(
            *local_encoded_features,
            query_node=global_nodes['Q'],
            global_nodes=global_nodes,
            return_nodes=True
        )
        local_nodes = self._stabilize_local_nodes(local_nodes)
        (
            local_nodes,
            latest_game_details,
            latest_positive_prior,
            latest_negative_prior
        ) = self._apply_event_game(local_nodes)
        latest_event_mask = self._event_mask_from_local_nodes(local_nodes)
        self._current_event_mask = latest_event_mask

        event_regularization_losses = [
            self.Local_path.last_event_regularization_loss
        ]

        latest_graph_feature = None
        latest_graph_details = None
        latest_updated_nodes = None
        latest_local_nodes = local_nodes
        latest_global_nodes = global_nodes

        for iteration_index in range(self.iteration_num):
            node_tensor = self._compose_node_tensor(
                latest_global_nodes,
                latest_local_nodes
            )

            (
                latest_graph_feature,
                latest_updated_nodes,
                latest_graph_details
            ) = self.graph_encoder(
                node_tensor,
                return_details=True,
                positive_prior=latest_positive_prior,
                negative_prior=latest_negative_prior
            )
            latest_updated_nodes = self._apply_event_mask(
                latest_updated_nodes,
                latest_event_mask
            )
            self._current_event_mask = latest_event_mask

            if iteration_index == self.iteration_num - 1:
                break

            latest_global_nodes = self._global_nodes_from_updated(
                latest_updated_nodes
            )

            ranges = self._event_ranges()
            previous_event_nodes = {
                'text': latest_updated_nodes[:, list(ranges['text']), :],
                'audio': latest_updated_nodes[:, list(ranges['audio']), :],
                'vision': latest_updated_nodes[:, list(ranges['vision']), :]
            }

            re_extracted_local_nodes = (
                self.Local_path.extract_events_from_encoded(
                    *local_encoded_features,
                    query_node=latest_global_nodes['Q'],
                    global_nodes=latest_global_nodes,
                    return_nodes=True,
                    previous_event_nodes=previous_event_nodes
                )
            )
            re_extracted_local_nodes = self._stabilize_local_nodes(
                re_extracted_local_nodes,
                latest_local_nodes
            )

            event_regularization_losses.append(
                self.Local_path.last_event_regularization_loss
            )

            ranges = self._event_ranges()
            graph_event_nodes = latest_updated_nodes[
                :,
                list(ranges['all']),
                :
            ]

            latest_local_nodes = self._refresh_local_events(
                graph_event_nodes,
                re_extracted_local_nodes,
                latest_global_nodes['Q']
            )
            (
                latest_local_nodes,
                latest_game_details,
                latest_positive_prior,
                latest_negative_prior
            ) = self._apply_event_game(latest_local_nodes)
            latest_event_mask = self._event_mask_from_local_nodes(
                latest_local_nodes
            )
            self._current_event_mask = latest_event_mask

        self._current_event_mask = latest_event_mask

        output, prediction_details = self._predict_from_updated_nodes(
            latest_updated_nodes
        )

        fused_feature = prediction_details['fused_feature']
        gate_details = prediction_details['gate_details']
        bipolar_details = prediction_details['bipolar_details']

        (
            counterfactual_effects,
            counterfactual_loss
        ) = self._calculate_counterfactual_effects(
            latest_updated_nodes,
            output
        )

        self.last_counterfactual_effects = (
            counterfactual_effects.detach()
        )

        self.last_counterfactual_loss = counterfactual_loss

        self.last_bipolar_prediction = (
            bipolar_details['bipolar_prediction']
        )
        self.last_bipolar_gate = (
            bipolar_details['bipolar_gate'].detach()
        )

        self.last_contrastive_loss = self._calculate_contrastive_loss(
            latest_updated_nodes
        )

        self.last_graph_regularization_loss = (
            self._calculate_graph_regularization_loss(latest_graph_details)
        )

        self.last_event_regularization_loss = (
            torch.stack(event_regularization_losses).mean()
        )

        if return_nodes:
            ranges = self._event_ranges()

            returned_nodes = {
                'GT': latest_updated_nodes[:, 0, :],
                'GA': latest_updated_nodes[:, 1, :],
                'GV': latest_updated_nodes[:, 2, :],
                'Q': latest_updated_nodes[:, 3, :],
                'Q_graph': gate_details['query_graph'],

                'LT_events': latest_updated_nodes[:, list(ranges['text']), :],
                'LA_events': latest_updated_nodes[:, list(ranges['audio']), :],
                'LV_events': latest_updated_nodes[:, list(ranges['vision']), :],

                'LT': self._masked_local_mean(
                    latest_updated_nodes[:, list(ranges['text']), :],
                    latest_local_nodes['text_active_mask']
                ),
                'LA': self._masked_local_mean(
                    latest_updated_nodes[:, list(ranges['audio']), :],
                    latest_local_nodes['audio_active_mask']
                ),
                'LV': self._masked_local_mean(
                    latest_updated_nodes[:, list(ranges['vision']), :],
                    latest_local_nodes['vision_active_mask']
                ),

                'global_graph': gate_details['global_graph'],
                'local_graph': gate_details['local_graph'],
                'fusion_feature': fused_feature,
                'graph_output': prediction_details['graph_output'],
                'graph_weight': gate_details['graph_weight'],
                'local_event_weight': gate_details['local_event_weight'],
                'counterfactual_effects': counterfactual_effects,
                'graph_encoder_query': latest_graph_feature,
                'iteration_num': torch.full(
                    (latest_updated_nodes.size(0), 1),
                    float(self.iteration_num),
                    device=latest_updated_nodes.device,
                    dtype=latest_updated_nodes.dtype
                ),
                'event_refresh_gate': latest_local_nodes.get(
                    'event_refresh_gate',
                    torch.zeros(
                        latest_updated_nodes.size(0),
                        3 * self.event_slot_num,
                        1,
                        device=latest_updated_nodes.device,
                        dtype=latest_updated_nodes.dtype
                    )
                ),

                'text_prior': bipolar_details['text_prior'],
                'positive_strength': bipolar_details['positive_strength'],
                'negative_strength': bipolar_details['negative_strength'],
                'conflict_strength': bipolar_details['conflict_strength'],
                'positive_context': bipolar_details['positive_context'],
                'negative_context': bipolar_details['negative_context'],
                'positive_evidence_weight': (
                    bipolar_details['positive_evidence_weight']
                ),
                'negative_evidence_weight': (
                    bipolar_details['negative_evidence_weight']
                ),
                'bipolar_prediction': (
                    bipolar_details['bipolar_prediction']
                ),
                'bipolar_gate': bipolar_details['bipolar_gate'],

                'text_event_attention': latest_local_nodes['text_event_attention'],
                'audio_event_attention': latest_local_nodes['audio_event_attention'],
                'vision_event_attention': latest_local_nodes['vision_event_attention'],
                'text_active_mask': latest_local_nodes['text_active_mask'],
                'audio_active_mask': latest_local_nodes['audio_active_mask'],
                'vision_active_mask': latest_local_nodes['vision_active_mask'],
                'text_active_count': (
                    latest_local_nodes['text_active_count'].unsqueeze(-1)
                ),
                'audio_active_count': (
                    latest_local_nodes['audio_active_count'].unsqueeze(-1)
                ),
                'vision_active_count': (
                    latest_local_nodes['vision_active_count'].unsqueeze(-1)
                ),
                'text_change_prior': latest_local_nodes['text_change_prior'],
                'audio_change_prior': latest_local_nodes['audio_change_prior'],
                'vision_change_prior': latest_local_nodes['vision_change_prior'],
                'text_dispersion': (
                    latest_local_nodes['text_dispersion'].unsqueeze(-1)
                ),
                'audio_dispersion': (
                    latest_local_nodes['audio_dispersion'].unsqueeze(-1)
                ),
                'vision_dispersion': (
                    latest_local_nodes['vision_dispersion'].unsqueeze(-1)
                ),

                'event_game_support': latest_game_details['support'],
                'event_game_veto': latest_game_details['veto'],
                'event_game_reserve': latest_game_details['reserve'],
                'event_game_consensus_strength': (
                    latest_game_details['consensus_strength']
                ),
                'event_game_conflict_strength': (
                    latest_game_details['conflict_strength']
                ),
                'event_game_reserve_strength': (
                    latest_game_details['reserve_strength']
                ),
                'event_game_complexity': latest_game_details['game_complexity'],
                'event_game_consensus_event': (
                    latest_game_details['consensus_event']
                ),
                'event_game_specific_event': (
                    latest_game_details['specific_event']
                ),
                'event_game_conflict_event': (
                    latest_game_details['conflict_event']
                ),
                'event_game_update_gate': latest_game_details['game_update_gate'],
                'event_game_positive_prior': latest_positive_prior,
                'event_game_negative_prior': latest_negative_prior
            }

            return output, returned_nodes

        return output


def build_model(args):
    return Gate_fusion(args)