import argparse
import yaml
import torch

from core.dataset import MMDataLoader
from core.utils import dict_to_namespace, setup_seed
from models.Gate_fusion import build_model


parser = argparse.ArgumentParser()
parser.add_argument(
    '--config_file',
    type=str,
    default='configs/mosi.yaml'
)
parser.add_argument(
    '--seed',
    type=int,
    default=1111
)
args_cmd = parser.parse_args()


# 1. 读取配置
with open(args_cmd.config_file, 'r', encoding='utf-8') as f:
    args = yaml.load(f, Loader=yaml.FullLoader)

args = dict_to_namespace(args)

setup_seed(args_cmd.seed)

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

print('Device:', device)


# 2. 创建数据和模型
data_loader = MMDataLoader(args)
model = build_model(args).to(device)

batch = next(iter(data_loader['train']))

video = batch['vision'].to(device)
audio = batch['audio'].to(device)
text = batch['text'].to(device)
label = batch['labels']['M'].to(device).view(-1, 1)

print('\nInput shapes:')
print('video:', video.shape)
print('audio:', audio.shape)
print('text:', text.shape)
print('label:', label.shape)


# 3. 测试原始普通前向传播
model.eval()

with torch.no_grad():
    normal_output = model(
        video,
        audio,
        text
    )

print('\nNormal forward output:')
print('output:', normal_output.shape)

assert normal_output.dim() == 2, (
    f'普通输出维度错误，当前为 {normal_output.shape}'
)

assert normal_output.size(0) == video.size(0), (
    '普通输出的batch维度与输入不一致'
)

assert normal_output.size(1) == 1, (
    f'普通输出最后一维应为1，当前为 {normal_output.size(1)}'
)


# 4. 测试节点返回模式
with torch.no_grad():
    node_output, graph_nodes = model(
        video,
        audio,
        text,
        return_nodes=True
    )

print('\nNode forward output:')
print('output:', node_output.shape)

expected_nodes = [
    'GT',
    'GA',
    'GV',
    'Q',
    'LT',
    'LA',
    'LV',
    'Q_graph',
    'global_graph',
    'local_graph',
    'fusion_feature',
    'graph_encoder_query'
]

expected_bipolar_nodes = [
    'graph_output',
    'text_prior',
    'positive_strength',
    'negative_strength',
    'conflict_strength',
    'positive_context',
    'negative_context',
    'bipolar_prediction',
    'bipolar_gate'
]

expected_event_nodes = [
    'LT_events',
    'LA_events',
    'LV_events'
]

expected_event_attentions = [
    'text_event_attention',
    'audio_event_attention',
    'vision_event_attention'
]

for node_name in expected_nodes:
    assert node_name in graph_nodes, (
        f'缺少节点：{node_name}'
    )

    node = graph_nodes[node_name]

    print(f'{node_name}: {node.shape}')

    assert node.dim() == 2, (
        f'{node_name} 应为二维张量 [B, D]，'
        f'当前为 {node.shape}'
    )

    assert node.size(0) == video.size(0), (
        f'{node_name} 的batch维度错误，'
        f'当前为 {node.size(0)}，'
        f'输入batch为 {video.size(0)}'
    )

    assert node.size(1) == 128, (
        f'{node_name} 的特征维度应为128，'
        f'当前为 {node.size(1)}'
    )

for node_name in expected_bipolar_nodes:
    assert node_name in graph_nodes, (
        f'缺少双极性证据节点：{node_name}'
    )

    node = graph_nodes[node_name]

    print(f'{node_name}: {node.shape}')

    assert node.dim() == 2, (
        f'{node_name} 应为二维张量 [B, D] 或 [B, 1]，'
        f'当前为 {node.shape}'
    )

    assert node.size(0) == video.size(0), (
        f'{node_name} 的batch维度与输入不一致'
    )

    if node_name in [
        'positive_context',
        'negative_context'
    ]:
        assert node.size(1) == 128, (
            f'{node_name} 的特征维度应为128，'
            f'当前为 {node.size(1)}'
        )
    else:
        assert node.size(1) == 1, (
            f'{node_name} 的最后一维应为1，'
            f'当前为 {node.size(1)}'
        )


for node_name in expected_event_nodes:
    assert node_name in graph_nodes, (
        f'缺少事件节点：{node_name}'
    )

    event_node = graph_nodes[node_name]
    print(f'{node_name}: {event_node.shape}')

    assert event_node.dim() == 3, (
        f'{node_name} 应为三维张量 [B, S, D]，'
        f'当前为 {event_node.shape}'
    )
    assert event_node.size(0) == video.size(0), (
        f'{node_name} 的batch维度与输入不一致'
    )
    assert event_node.size(2) == 128, (
        f'{node_name} 的特征维度应为128，'
        f'当前为 {event_node.size(2)}'
    )

for attention_name in expected_event_attentions:
    assert attention_name in graph_nodes, (
        f'缺少事件注意力：{attention_name}'
    )

    attention = graph_nodes[attention_name]
    print(f'{attention_name}: {attention.shape}')

    assert attention.dim() == 3, (
        f'{attention_name} 应为三维张量 [B, S, L]，'
        f'当前为 {attention.shape}'
    )
    assert attention.size(0) == video.size(0), (
        f'{attention_name} 的batch维度与输入不一致'
    )

assert 'counterfactual_effects' in graph_nodes, (
    '缺少软反事实事件贡献：counterfactual_effects'
)
counterfactual_effects = graph_nodes['counterfactual_effects']
print('counterfactual_effects:', counterfactual_effects.shape)
assert counterfactual_effects.dim() == 2, (
    'counterfactual_effects 应为二维张量 [B, event_num]'
)
assert counterfactual_effects.size(0) == video.size(0), (
    'counterfactual_effects 的batch维度与输入不一致'
)
assert not torch.isnan(counterfactual_effects).any(), (
    'counterfactual_effects 中出现NaN'
)
assert not torch.isinf(counterfactual_effects).any(), (
    'counterfactual_effects 中出现Inf'
)

assert 'graph_weight' in graph_nodes, '缺少图增强门控权重：graph_weight'
graph_weight = graph_nodes['graph_weight']
print('graph_weight:', graph_weight.shape)
assert graph_weight.dim() == 2, (
    f'graph_weight 应为二维张量 [B, 1]，当前为 {graph_weight.shape}'
)
assert graph_weight.size(0) == video.size(0), (
    'graph_weight 的batch维度与输入不一致'
)
assert graph_weight.size(1) == 1, (
    f'graph_weight 最后一维应为1，当前为 {graph_weight.size(1)}'
)
assert torch.all((graph_weight >= 0) & (graph_weight <= 1)), (
    'graph_weight 应位于[0, 1]区间'
)

assert 'positive_evidence_weight' in graph_nodes, (
    '缺少正向证据注意力：positive_evidence_weight'
)
assert 'negative_evidence_weight' in graph_nodes, (
    '缺少负向证据注意力：negative_evidence_weight'
)

positive_evidence_weight = graph_nodes['positive_evidence_weight']
negative_evidence_weight = graph_nodes['negative_evidence_weight']

print('positive_evidence_weight:', positive_evidence_weight.shape)
print('negative_evidence_weight:', negative_evidence_weight.shape)

for weight_name, evidence_weight in [
    ('positive_evidence_weight', positive_evidence_weight),
    ('negative_evidence_weight', negative_evidence_weight)
]:
    assert evidence_weight.dim() == 2, (
        f'{weight_name} 应为二维张量 [B, node_num]，'
        f'当前为 {evidence_weight.shape}'
    )
    assert evidence_weight.size(0) == video.size(0), (
        f'{weight_name} 的batch维度与输入不一致'
    )
    assert torch.all(evidence_weight >= 0), (
        f'{weight_name} 应为非负权重'
    )
    assert torch.allclose(
        evidence_weight.sum(dim=-1),
        torch.ones_like(evidence_weight.sum(dim=-1)),
        atol=1e-4
    ), (
        f'{weight_name} 每个样本的权重和应接近1'
    )

assert torch.all(
    (graph_nodes['bipolar_gate'] >= 0)
    & (graph_nodes['bipolar_gate'] <= 1)
), 'bipolar_gate 应位于[0, 1]区间'

assert torch.all(
    graph_nodes['positive_strength'] >= 0
), 'positive_strength 应为非负值'

assert torch.all(
    graph_nodes['negative_strength'] >= 0
), 'negative_strength 应为非负值'

# 5. 检查普通输出与节点模式输出是否一致
max_difference = torch.max(
    torch.abs(normal_output - node_output)
).item()

print('\nOutput consistency:')
print('max difference:', max_difference)

assert max_difference < 1e-5, (
    '普通模式与节点模式的预测结果不一致，'
    f'最大差值为 {max_difference}'
)


# 6. 测试反向传播
model.train()
model.zero_grad()

train_output, train_nodes = model(
    video,
    audio,
    text,
    return_nodes=True
)

loss_fn = torch.nn.MSELoss()
loss = loss_fn(train_output, label)

print('\nBackward test:')
print('loss:', loss.item())

loss.backward()


# 7. 检查是否有梯度
has_gradient = False
gradient_count = 0

for name, parameter in model.named_parameters():
    if parameter.grad is not None:
        has_gradient = True
        gradient_count += 1

assert has_gradient, '模型反向传播后没有任何参数梯度'

print('parameters with gradients:', gradient_count)


# 8. 检查节点数值
for node_name, node in train_nodes.items():
    assert not torch.isnan(node).any(), (
        f'{node_name} 中出现NaN'
    )

    assert not torch.isinf(node).any(), (
        f'{node_name} 中出现Inf'
    )

print('\n====================================')
print('所有测试通过')
print('普通前向传播正常')
print('图节点、情感事件节点和图增强门控输出正常')
print('普通模式与节点模式结果一致')
print('反向传播正常')
print('节点中没有NaN或Inf')
print('====================================')