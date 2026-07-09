import torch
from torch import nn
from einops import repeat

from .bert import BertTextEncoder
from .main_layer import Transformer


class GlobalFusionClassifier(nn.Module):
    """
    全局节点提取器。

    本模块只保留三模态编码、统一维度投影和查询节点生成，
    不再执行原 DGLQA 跨模态融合，也不再使用 CrossTransformer
    生成全局融合特征。全局—局部跨模态关系建模交给后续的
    显式有符号异构图完成。

    输出节点：
        GT: global text node
        GA: global audio node
        GV: global vision node
        Q:  sample-conditioned query node
    """

    def __init__(self, args):
        super().__init__()

        args = args.model

        self.token_len = args.token_len
        self.token_dim = args.token_dim

        self.LearnableQuery = nn.Parameter(
            torch.ones(1, args.token_len, args.token_dim)
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

        self.global_norm = nn.LayerNorm(args.token_dim)

        self.query_condition = nn.Sequential(
            nn.Linear(args.token_dim * 4, args.token_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(args.token_dim, args.token_dim)
        )

        self.query_norm = nn.LayerNorm(args.token_dim)

    def encode_text(self, x_text):
        """只运行一次 BERT，供全局路径和局部路径共享文本编码。"""

        return self.bertmodel(x_text)

    def forward_from_text_embedding(
        self,
        x_visual,
        x_audio,
        text_embedding,
        return_sequences=False
    ):
        batch_size = x_visual.size(0)

        h_t = self.proj_l(text_embedding)
        h_a = self.proj_a(x_audio)
        h_v = self.proj_v(x_visual)

        GT = self.global_norm(h_t.mean(dim=1))
        GA = self.global_norm(h_a.mean(dim=1))
        GV = self.global_norm(h_v.mean(dim=1))

        base_query = repeat(
            self.LearnableQuery,
            '1 n d -> b n d',
            b=batch_size
        ).mean(dim=1)

        query_input = torch.cat(
            [base_query, GT, GA, GV],
            dim=-1
        )

        Q = self.query_norm(
            base_query + self.query_condition(query_input)
        )

        global_nodes = {
            'GT': GT,
            'GA': GA,
            'GV': GV,
            'Q': Q
        }

        if return_sequences:
            return global_nodes, {
                'text': h_t,
                'audio': h_a,
                'vision': h_v
            }

        return global_nodes

    def forward(self, x_visual, x_audio, x_text, return_nodes=False):
        text_embedding = self.encode_text(x_text)

        global_nodes = self.forward_from_text_embedding(
            x_visual,
            x_audio,
            text_embedding,
            return_sequences=False
        )

        if return_nodes:
            return global_nodes

        return global_nodes


def build_model(args):
    model = GlobalFusionClassifier(args)
    return model