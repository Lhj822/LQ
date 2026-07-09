import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bert import BertTextEncoder
from .main_layer import ConvModulOperationSpatialAttention, Transformer


class QueryGuidedEventSlotExtractor(nn.Module):
    """
    查询引导的无窗口情感事件槽提取器。

    输入完整模态序列，不使用均值池化、固定窗口、滑动窗口或硬Top-K。
    每个可学习事件槽在查询节点和对应全局模态节点的引导下，对完整
    序列产生软分配权重，并加权聚合为一个局部情感事件节点。
    """

    def __init__(self, dim=128, event_slot_num=3, dropout=0.1):
        super().__init__()

        self.dim = dim
        self.event_slot_num = event_slot_num

        self.event_slots = nn.Parameter(
            torch.empty(event_slot_num, dim)
        )
        nn.init.normal_(self.event_slots, mean=0.0, std=0.02)

        self.query_condition = nn.Linear(dim, dim)
        self.global_condition = nn.Linear(dim, dim)
        self.previous_event_condition = nn.Linear(dim, dim)

        self.change_prior_scale = nn.Parameter(torch.tensor(1.0))
        self.repulsion_scale = nn.Parameter(torch.tensor(0.5))

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.output = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _normalized_entropy(attention, eps=1e-8):
        length = attention.size(-1)
        entropy = -(
            attention
            * torch.log(attention.clamp_min(eps))
        ).sum(dim=-1)
        return entropy / math.log(max(length, 2))

    def _regularization_loss(self, attention):
        """
        attention: [B, S, L]

        diversity_loss: 减少不同事件槽重复关注同一区域。
        compactness_loss: 避免单个事件槽关注分布过于分散。
        continuity_loss: 鼓励注意力沿时间轴相对连续。
        """

        batch_size, slot_num, _ = attention.shape

        overlap = torch.bmm(
            attention,
            attention.transpose(1, 2)
        )

        identity = torch.eye(
            slot_num,
            device=attention.device,
            dtype=attention.dtype
        ).unsqueeze(0)

        diversity_loss = (
            (overlap * (1.0 - identity))
            .pow(2)
            .sum(dim=(1, 2))
            / max(slot_num * (slot_num - 1), 1)
        ).mean()

        compactness_loss = self._normalized_entropy(
            attention
        ).mean()

        continuity_loss = torch.abs(
            attention[:, :, 1:] - attention[:, :, :-1]
        ).mean()

        return {
            'diversity': diversity_loss,
            'compactness': compactness_loss,
            'continuity': continuity_loss,
            'total': (
                diversity_loss
                + 0.1 * compactness_loss
                + 0.1 * continuity_loss
            )
        }

    def _change_prior(self, sequence, eps=1e-8):
        diff = torch.norm(
            sequence[:, 1:, :] - sequence[:, :-1, :],
            p=2,
            dim=-1
        )

        diff = F.pad(diff, (1, 0), value=0.0)
        diff = diff / diff.mean(dim=-1, keepdim=True).clamp_min(eps)
        change_prior = F.softmax(diff, dim=-1)

        entropy = -(
            change_prior
            * torch.log(change_prior.clamp_min(eps))
        ).sum(dim=-1)
        dispersion = entropy / math.log(max(sequence.size(1), 2))

        active_count = torch.where(
            dispersion < 0.45,
            torch.ones_like(dispersion, dtype=torch.long),
            torch.where(
                dispersion < 0.75,
                torch.full_like(dispersion, 3, dtype=torch.long),
                torch.full_like(dispersion, 5, dtype=torch.long)
            )
        )

        slot_index = torch.arange(
            self.event_slot_num,
            device=sequence.device
        ).unsqueeze(0)
        active_mask = (
            slot_index < active_count.unsqueeze(1)
        ).to(sequence.dtype)

        return change_prior, dispersion, active_count, active_mask

    def forward(
        self,
        sequence,
        query_node,
        global_node,
        previous_event_nodes=None
    ):
        """
        sequence:    [B, L, D]
        query_node:  [B, D]
        global_node: [B, D]
        previous_event_nodes: [B, S, D] or None

        return:
            event_nodes: [B, S, D]
            attention:   [B, S, L]
            reg_loss:    dict
            active_mask: [B, S]
            active_count:[B]
            change_prior:[B, L]
        """

        batch_size = sequence.size(0)

        if previous_event_nodes is None:
            previous_event_nodes = torch.zeros(
                batch_size,
                self.event_slot_num,
                self.dim,
                device=sequence.device,
                dtype=sequence.dtype
            )

        change_prior, dispersion, active_count, active_mask = (
            self._change_prior(sequence)
        )

        slots = self.event_slots.unsqueeze(0).expand(
            batch_size,
            -1,
            -1
        )

        conditioned_slots = (
            slots
            + self.query_condition(query_node).unsqueeze(1)
            + self.global_condition(global_node).unsqueeze(1)
            + self.previous_event_condition(previous_event_nodes)
        )

        query = self.to_q(conditioned_slots)
        key = self.to_k(sequence)
        value = self.to_v(sequence)

        change_bias = torch.log(change_prior.clamp_min(1e-8))
        coverage = torch.zeros_like(change_prior)
        attention_list = []

        for slot_idx in range(self.event_slot_num):
            content_logits = torch.matmul(
                query[:, slot_idx:slot_idx + 1, :],
                key.transpose(1, 2)
            ).squeeze(1) / math.sqrt(self.dim)

            logits = (
                content_logits
                + self.change_prior_scale * change_bias
                - F.softplus(self.repulsion_scale) * coverage
            )

            slot_attention = F.softmax(logits, dim=-1)
            slot_attention = self.dropout(slot_attention)
            slot_attention = (
                slot_attention
                * active_mask[:, slot_idx:slot_idx + 1]
            )

            attention_list.append(slot_attention)
            coverage = coverage + slot_attention.detach()

        attention = torch.stack(attention_list, dim=1)

        event_nodes = torch.matmul(attention, value)
        event_nodes = self.norm(
            event_nodes + self.output(event_nodes)
        )
        event_nodes = event_nodes * active_mask.unsqueeze(-1)

        reg_loss = self._regularization_loss(attention)

        return (
            event_nodes,
            attention,
            reg_loss,
            active_mask,
            active_count,
            change_prior,
            dispersion
        )

class LocalFusionClassifier(nn.Module):
    """
    局部情感事件节点提取器。

    本模块不再对完整序列做均值池化，而是为文本、音频、视频分别
    设置多个可学习情感事件槽。事件槽在全局查询节点Q和对应全局
    模态节点的引导下，从完整序列中软选择局部情感区域，并生成
    多个局部情感事件节点。
    """

    def __init__(self, args):
        super().__init__()

        args = args.model

        self.event_slot_num = getattr(
            args,
            'event_slot_num',
            3
        )

        self.bertmodel = BertTextEncoder(
            use_finetune=True,
            transformers='bert',
            pretrained=args.bert_pretrained
        )

        self.proj_l = nn.Sequential(
            nn.Linear(args.l_proj_dim, args.proj_dst_dim),
            Transformer(
                num_frames=args.l_proj_length,
                save_hidden=False,
                token_len=args.token_length,
                dim=args.proj_input_dim,
                depth=args.proj_depth,
                heads=args.proj_heads,
                mlp_dim=args.proj_mlp_dim
            )
        )

        self.proj_a = nn.Sequential(
            nn.Linear(args.a_proj_dim, args.proj_dst_dim),
            Transformer(
                num_frames=args.a_proj_length,
                save_hidden=False,
                token_len=args.token_length,
                dim=args.proj_input_dim,
                depth=args.proj_depth,
                heads=args.proj_heads,
                mlp_dim=args.proj_mlp_dim
            )
        )

        self.proj_v = nn.Sequential(
            nn.Linear(args.v_proj_dim, args.proj_dst_dim),
            Transformer(
                num_frames=args.v_proj_length,
                save_hidden=False,
                token_len=args.token_length,
                dim=args.proj_input_dim,
                depth=args.proj_depth,
                heads=args.proj_heads,
                mlp_dim=args.proj_mlp_dim
            )
        )

        self.conv_att_v = ConvModulOperationSpatialAttention(
            args.v_proj_dim,
            kernel_size=3
        )

        self.text_event_extractor = QueryGuidedEventSlotExtractor(
            dim=args.token_dim,
            event_slot_num=self.event_slot_num
        )
        self.audio_event_extractor = QueryGuidedEventSlotExtractor(
            dim=args.token_dim,
            event_slot_num=self.event_slot_num
        )
        self.vision_event_extractor = QueryGuidedEventSlotExtractor(
            dim=args.token_dim,
            event_slot_num=self.event_slot_num
        )

        self.last_event_regularization_loss = torch.tensor(0.0)
        self.last_event_regularization_details = {}

    def _combine_regularization(self, *regularization_dicts):
        total = sum(
            item['total']
            for item in regularization_dicts
        ) / len(regularization_dicts)

        details = {
            'text': regularization_dicts[0],
            'audio': regularization_dicts[1],
            'vision': regularization_dicts[2],
            'total': total
        }

        return total, details

    def encode_from_text_embedding(
        self,
        video_feat,
        audio_feat,
        text_embedding
    ):
        """复用全局路径已经计算过的 BERT 文本编码，只做局部路径投影。"""

        video_feat = video_feat.unsqueeze(-1)
        video_feat = video_feat.permute(0, 2, 1, 3)
        video_feat = self.conv_att_v(video_feat)
        video_feat = video_feat.squeeze(-1)
        video_feat = video_feat.permute(0, 2, 1)

        text_feat = self.proj_l(text_embedding)
        audio_feat = self.proj_a(audio_feat)
        video_feat = self.proj_v(video_feat)

        return text_feat, audio_feat, video_feat

    def extract_events_from_encoded(
        self,
        text_feat,
        audio_feat,
        video_feat,
        query_node=None,
        global_nodes=None,
        return_nodes=False,
        previous_event_nodes=None
    ):
        if query_node is None or global_nodes is None:
            raise ValueError(
                'LocalFusionClassifier 需要 query_node 和 global_nodes '
                '来执行查询引导的情感事件节点提取。'
            )

        previous_text = None
        previous_audio = None
        previous_vision = None

        if previous_event_nodes is not None:
            previous_text = previous_event_nodes.get('text', None)
            previous_audio = previous_event_nodes.get('audio', None)
            previous_vision = previous_event_nodes.get('vision', None)

        (
            text_events,
            text_attention,
            text_reg,
            text_active_mask,
            text_active_count,
            text_change_prior,
            text_dispersion
        ) = self.text_event_extractor(
            text_feat,
            query_node,
            global_nodes['GT'],
            previous_text
        )

        (
            audio_events,
            audio_attention,
            audio_reg,
            audio_active_mask,
            audio_active_count,
            audio_change_prior,
            audio_dispersion
        ) = self.audio_event_extractor(
            audio_feat,
            query_node,
            global_nodes['GA'],
            previous_audio
        )

        (
            vision_events,
            vision_attention,
            vision_reg,
            vision_active_mask,
            vision_active_count,
            vision_change_prior,
            vision_dispersion
        ) = self.vision_event_extractor(
            video_feat,
            query_node,
            global_nodes['GV'],
            previous_vision
        )

        (
            self.last_event_regularization_loss,
            self.last_event_regularization_details
        ) = self._combine_regularization(
            text_reg,
            audio_reg,
            vision_reg
        )

        local_nodes = {
            'LT_events': text_events,
            'LA_events': audio_events,
            'LV_events': vision_events,

            # 兼容性聚合节点，仅用于可视化或旧测试，不再作为图中唯一局部节点。
            'LT': self._masked_mean(text_events, text_active_mask),
            'LA': self._masked_mean(audio_events, audio_active_mask),
            'LV': self._masked_mean(vision_events, vision_active_mask),

            'text_event_attention': text_attention,
            'audio_event_attention': audio_attention,
            'vision_event_attention': vision_attention,

            'text_active_mask': text_active_mask,
            'audio_active_mask': audio_active_mask,
            'vision_active_mask': vision_active_mask,
            'text_active_count': text_active_count.float(),
            'audio_active_count': audio_active_count.float(),
            'vision_active_count': vision_active_count.float(),
            'text_change_prior': text_change_prior,
            'audio_change_prior': audio_change_prior,
            'vision_change_prior': vision_change_prior,
            'text_dispersion': text_dispersion,
            'audio_dispersion': audio_dispersion,
            'vision_dispersion': vision_dispersion
        }

        if return_nodes:
            return local_nodes

        return local_nodes

    @staticmethod
    def _masked_mean(event_nodes, active_mask):
        denominator = active_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (event_nodes * active_mask.unsqueeze(-1)).sum(dim=1) / denominator


    def forward(
        self,
        video_feat,
        audio_feat,
        text_feat,
        query_node=None,
        global_nodes=None,
        return_nodes=False
    ):
        text_embedding = self.bertmodel(text_feat)

        encoded_features = self.encode_from_text_embedding(
            video_feat,
            audio_feat,
            text_embedding
        )

        return self.extract_events_from_encoded(
            *encoded_features,
            query_node=query_node,
            global_nodes=global_nodes,
            return_nodes=return_nodes
        )


def build_model(args):
    model = LocalFusionClassifier(args)
    return model