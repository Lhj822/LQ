import os
import torch
import argparse
import torch.nn.functional as F
from tqdm import tqdm

from core.dataset import MMDataLoader
from core.scheduler import get_scheduler
from core.utils import (
	AverageMeter,
	setup_seed,
	results_recorder,
	dict_to_namespace
)
from tensorboardX import SummaryWriter
from models.Gate_fusion import build_model
from core.metric import MetricsTop
import yaml

parser = argparse.ArgumentParser()
parser.add_argument(
	'--config_file',
	type=str,
	default='configs/mosi.yaml'
)
parser.add_argument(
	'--seed',
	type=int,
	default=-1
)
parser.add_argument(
	'--gpu_id',
	type=int,
	default=0
)

opt = parser.parse_args()
print(opt)

with open(opt.config_file) as f:
	args = yaml.load(
		f,
		Loader=yaml.FullLoader
	)

args = dict_to_namespace(args)
print(args)

seed = (
	args.base.seed
	if opt.seed == -1
	else opt.seed
)

gpu_id = (
	args.base.gpu_id
	if opt.gpu_id == -1
	else opt.gpu_id
)

print('-----------------args-----------------')
print(args)
print('-------------------------------------')

gpu_id = str(gpu_id)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

USE_CUDA = torch.cuda.is_available()

device = torch.device(
	"cuda" if USE_CUDA else "cpu"
)

print(f"Device: {device} ({gpu_id})")


def is_better_score(
		current_score,
		best_score,
		mode,
		min_delta
):
	if best_score is None:
		return True

	if mode == "max":
		return current_score > best_score + min_delta

	return current_score < best_score - min_delta


def main():
	print(f"seed{seed}")

	setup_seed(seed)

	log_path = os.path.join(
		".",
		"log",
		args.base.project_name
	)

	if not os.path.exists(log_path):
		os.makedirs(log_path)

	log_path = "/root/tf-logs"

	save_path = os.path.join(
		args.base.ckpt_root,
		args.base.project_name
	)

	if not os.path.exists(save_path):
		os.makedirs(save_path)

	model = build_model(args).to(device)

	dataLoader = MMDataLoader(args)

	optimizer = torch.optim.AdamW(
		model.parameters(),
		lr=args.base.lr,
		weight_decay=args.base.weight_decay
	)

	scheduler_warmup = get_scheduler(
		optimizer,
		args
	)

	loss_fn = torch.nn.MSELoss()

	metrics_fn = MetricsTop().getMetics(
		args.dataset.datasetName
	)

	training_results_recorder = results_recorder()
	validation_results_recorder = results_recorder()
	test_results_recorder = results_recorder()

	writer = SummaryWriter(
		logdir=log_path
	)

	early_stop_patience = getattr(
		args.base,
		"early_stop_patience",
		0
	)
	early_stop_metric = getattr(
		args.base,
		"early_stop_metric",
		"MAE"
	)
	early_stop_mode = getattr(
		args.base,
		"early_stop_mode",
		"min"
	)
	early_stop_min_delta = getattr(
		args.base,
		"early_stop_min_delta",
		0.0
	)

	best_valid_score = None
	best_epoch = 0
	best_valid_results = {}
	best_test_results_at_valid = {}
	best_cpk_path = ""
	early_stop_counter = 0

	best_cpk_dir = "./best_cpk"

	os.makedirs(
		best_cpk_dir,
		exist_ok=True
	)

	for epoch in range(
			1,
			args.base.n_epochs + 1
	):
		training_ret = train(
			model,
			dataLoader['train'],
			optimizer,
			loss_fn,
			metrics_fn,
			epoch
		)

		validation_ret = evaluate(
			model,
			dataLoader['valid'],
			loss_fn,
			metrics_fn,
			epoch,
			mode='Valid'
		)

		test_ret = evaluate(
			model,
			dataLoader['test'],
			loss_fn,
			metrics_fn,
			epoch,
			mode='Test'
		)

		training_results_recorder.update(
			training_ret['results'],
			epoch
		)

		validation_results_recorder.update(
			validation_ret['results'],
			epoch
		)

		test_results_recorder.update(
			test_ret['results'],
			epoch
		)

		best_validation_results = (
			validation_results_recorder
			.get_best_results()
		)

		best_test_results = (
			test_results_recorder
			.get_best_results()
		)

		current_valid_score = (
			validation_ret['results']
			.get(
				early_stop_metric,
				validation_ret['loss_recorder'].value_avg
			)
		)

		improved = is_better_score(
			current_valid_score,
			best_valid_score,
			early_stop_mode,
			early_stop_min_delta
		)

		if improved:
			# delete previous best checkpoint
			if os.path.exists(best_cpk_path):
				os.remove(best_cpk_path)

			best_valid_score = current_valid_score
			best_epoch = epoch
			best_valid_results = validation_ret['results']
			best_test_results_at_valid = test_ret['results']
			early_stop_counter = 0

			# save best model
			cpk_name = (
				f"valid_{early_stop_metric}_"
				f"{best_valid_score:.4f}_"
				f"epoch_{best_epoch}_"
				f"seed_{seed}.pth"
			)

			best_cpk_path = os.path.join(
				best_cpk_dir,
				cpk_name
			)

			torch.save(
				{
					'epoch': best_epoch,
					'model_state_dict': (
						model.state_dict()
					),
					'optimizer_state_dict': (
						optimizer.state_dict()
					),
					'valid_metric': early_stop_metric,
					'valid_score': (
						best_valid_score
					),
					'valid_results': best_valid_results,
					'test_results_at_valid_best': (
						best_test_results_at_valid
					),
					'seed': seed
				},
				best_cpk_path
			)
		elif early_stop_patience > 0:
			early_stop_counter += 1

		print(
			f'\n----------------- Results Epoch '
			f'{epoch} -----------------'
		)

		print(
			f'Learning Rate: '
			f'{optimizer.state_dict()["param_groups"][0]["lr"]}'
		)

		print(
			f'Training Results: '
			f'{training_ret["results"]}'
		)

		print(
			f'Validation Results: '
			f'{validation_ret["results"]}'
		)

		print(
			f'Test Results: '
			f'{test_ret["results"]}\n'
		)

		print(
			f'Best Validation Results across All Epochs: '
			f'{best_validation_results["best_results_all_epochs"]}'
		)

		print(
			f'Best Validation Results of One Epoch: '
			f'{best_validation_results["best_results_one_epoch"]}\n'
		)

		print(
			f'Best Test Results across All Epochs: '
			f'{best_test_results["best_results_all_epochs"]}'
		)

		print(
			f'Best Test Results of One Epoch: '
			f'{best_test_results["best_results_one_epoch"]}'
		)

		print(
			f'Best Checkpoint by Validation '
			f'{early_stop_metric}: '
			f'epoch={best_epoch}, '
			f'score={best_valid_score}, '
			f'valid={best_valid_results}, '
			f'test_at_best_valid={best_test_results_at_valid}'
		)

		if early_stop_patience > 0:
			print(
				f'EarlyStopping: counter='
				f'{early_stop_counter}/'
				f'{early_stop_patience}, '
				f'monitor={early_stop_metric}, '
				f'mode={early_stop_mode}'
			)

		print(
			'----------------------------------------------------------\n'
		)

		writer.add_scalar(
			'train/MAE',
			training_ret[
				'loss_recorder'
			].value_avg,
			epoch
		)

		writer.add_scalar(
			'valid/MAE',
			validation_ret[
				'loss_recorder'
			].value_avg,
			epoch
		)

		writer.add_scalar(
			'test/MAE',
			test_ret[
				'loss_recorder'
			].value_avg,
			epoch
		)

		# Note: For MOSI dataset, the best model parameters
		# are achieved at epoch 31; for MOSEI, at epoch 4.
		# Consider implementing early stopping based on
		# validation performance.
		scheduler_warmup.step()

		if (
				early_stop_patience > 0
				and early_stop_counter >= early_stop_patience
		):
			print(
				f'Early stopping triggered at epoch {epoch}. '
				f'Best epoch is {best_epoch} with validation '
				f'{early_stop_metric}={best_valid_score}.'
			)
			break

	# add to best.txt
	with open('best.txt', 'a') as f:
		f.write(
			f'Seed: {seed}, '
			f'Epoch: {best_epoch}, '
			f'Valid {early_stop_metric}: {best_valid_score}, '
			f'Valid Results: {best_valid_results}, '
			f'Test Results at Best Valid: '
			f'{best_test_results_at_valid}\n'
		)

	writer.close()


def train(
		model,
		data_loader,
		optimizer,
		loss_fn,
		metrics_fn,
		epoch
):
	loss_recorder = AverageMeter()

	y_pred, y_true = [], []

	model.train()

	train_progress = tqdm(
		data_loader,
		total=len(data_loader),
		desc=f"Epoch {epoch} Train",
		ncols=150,
		leave=False
	)

	for cur_iter, data in enumerate(
			train_progress
	):
		img = data['vision'].to(device)
		audio = data['audio'].to(device)
		text = data['text'].to(device)

		label = data['labels']['M'].to(
			device
		)

		label = label.view(-1, 1)

		batchsize = img.shape[0]

		output = model(
			img,
			audio,
			text
		)

		# 情感回归任务损失
		task_loss = loss_fn(
			output,
			label
		)

		# 全局—局部跨粒度对比损失
		contrastive_loss = getattr(
			model,
			"last_contrastive_loss",
			0.0
		)

		# 正负边互斥正则损失
		graph_regularization_loss = getattr(
			model,
			"last_graph_regularization_loss",
			0.0
		)

		# 情感事件槽多样性、紧凑性和连续性正则
		event_regularization_loss = getattr(
			model,
			"last_event_regularization_loss",
			0.0
		)

		# 每次训练前向都计算的软反事实事件贡献监测损失
		counterfactual_loss = getattr(
			model,
			"last_counterfactual_loss",
			0.0
		)

		# 文本锚定双极性证据分支辅助回归损失
		bipolar_prediction = getattr(
			model,
			"last_bipolar_prediction",
			None
		)

		if bipolar_prediction is None:
			bipolar_loss = torch.zeros(
				(),
				device=label.device
			)
		else:
			bipolar_loss = loss_fn(
				bipolar_prediction,
				label
			)

			# Has0二分类辅助损失：
		# 直接约束回归输出在0边界两侧的符号，
		# 用于提升包含中性/零值样本的Has0 Acc-2和F1。
		has0_classification_loss = (
			F.binary_cross_entropy_with_logits(
				output,
				(label >= 0).float()
			)
		)

		# 前10个epoch进行对比损失预热：
		# epoch 1  -> 0.001
		# epoch 5  -> 0.005
		# epoch 10 -> 0.010
		contrastive_weight_max = getattr(
			args.model,
			"contrastive_weight",
			0.01
		)

		contrastive_weight = (
				contrastive_weight_max
				* min(
			1.0,
			epoch / 10.0
		)
		)

		graph_regularization_weight = getattr(
			args.model,
			"graph_regularization_weight",
			0.001
		)
		event_regularization_weight = getattr(
			args.model,
			"event_regularization_weight",
			0.001
		)
		counterfactual_weight = getattr(
			args.model,
			"counterfactual_weight",
			0.001
		)
		bipolar_weight = getattr(
			args.model,
			"bipolar_weight",
			0.05
		)

		has0_classification_weight = getattr(
			args.model,
			"has0_classification_weight",
			0.0
		)

		loss = (
				task_loss
				+ contrastive_weight
				* contrastive_loss
				+ graph_regularization_weight
				* graph_regularization_loss
				+ event_regularization_weight
				* event_regularization_loss
				+ counterfactual_weight
				* counterfactual_loss
				+ bipolar_weight
				* bipolar_loss
				+ has0_classification_weight
				* has0_classification_loss
		)

		loss_recorder.update(
			loss.item(),
			batchsize
		)

		loss.backward()

		optimizer.step()

		optimizer.zero_grad()

		y_pred.append(
			output.cpu()
		)

		y_true.append(
			label.cpu()
		)

		if torch.is_tensor(
				contrastive_loss
		):
			contrastive_loss_value = (
				contrastive_loss
				.detach()
				.item()
			)
		else:
			contrastive_loss_value = float(
				contrastive_loss
			)

		if torch.is_tensor(
				graph_regularization_loss
		):
			graph_regularization_loss_value = (
				graph_regularization_loss
				.detach()
				.item()
			)
		else:
			graph_regularization_loss_value = float(
				graph_regularization_loss
			)

		if torch.is_tensor(
				event_regularization_loss
		):
			event_regularization_loss_value = (
				event_regularization_loss
				.detach()
				.item()
			)
		else:
			event_regularization_loss_value = float(
				event_regularization_loss
			)

		if torch.is_tensor(
				counterfactual_loss
		):
			counterfactual_loss_value = (
				counterfactual_loss
				.detach()
				.item()
			)
		else:
			counterfactual_loss_value = float(
				counterfactual_loss
			)

		if torch.is_tensor(
				bipolar_loss
		):
			bipolar_loss_value = (
				bipolar_loss
				.detach()
				.item()
			)
		else:
			bipolar_loss_value = float(
				bipolar_loss
			)
		has0_classification_loss_value = (
			has0_classification_loss
			.detach()
			.item()
		)

		train_progress.set_postfix(
			{
				'task': (
					f'{task_loss.item():.4f}'
				),
				'contrast': (
					f'{contrastive_loss_value:.4f}'
				),
				'cw': (
					f'{contrastive_weight:.4f}'
				),
				'graph': (
					f'{graph_regularization_loss_value:.6f}'
				),
				'event': (
					f'{event_regularization_loss_value:.6f}'
				),
				'cf': (
					f'{counterfactual_loss_value:.6f}'
				),
				'bipolar': (
					f'{bipolar_loss_value:.4f}'
				),
				'has0': (
					f'{has0_classification_loss_value:.4f}'
				),
				'h0w': (
					f'{has0_classification_weight:.4f}'
				),
				'total': (
					f'{loss.item():.4f}'
				),
				'avg': (
					f'{loss_recorder.value_avg:.4f}'
				)
			}
		)

	train_progress.close()

	pred = torch.cat(y_pred)
	true = torch.cat(y_true)

	results = metrics_fn(
		pred,
		true
	)

	return {
		'results': results,
		'loss_recorder': loss_recorder
	}


def evaluate(
		model,
		data_loader,
		loss_fn,
		metrics_fn,
		epoch,
		mode='Valid'
):
	loss_recorder = AverageMeter()

	y_pred, y_true = [], []

	model.eval()

	eval_progress = tqdm(
		data_loader,
		total=len(data_loader),
		desc=f"Epoch {epoch} {mode}",
		ncols=120,
		leave=False
	)

	for cur_iter, data in enumerate(
			eval_progress
	):
		img = data['vision'].to(device)
		audio = data['audio'].to(device)
		text = data['text'].to(device)

		label = data['labels']['M'].to(
			device
		)

		label = label.view(-1, 1)

		batchsize = img.shape[0]

		with torch.no_grad():
			output = model(
				img,
				audio,
				text
			)

		# 验证集和测试集只计算原始MSE损失
		loss = loss_fn(
			output,
			label
		)

		y_pred.append(
			output.cpu()
		)

		y_true.append(
			label.cpu()
		)

		loss_recorder.update(
			loss.item(),
			batchsize
		)

		eval_progress.set_postfix(
			{
				'loss': (
					f'{loss.item():.4f}'
				),
				'avg': (
					f'{loss_recorder.value_avg:.4f}'
				)
			}
		)

	eval_progress.close()

	pred = torch.cat(y_pred)
	true = torch.cat(y_true)

	results = metrics_fn(
		pred,
		true
	)

	return {
		'results': results,
		'loss_recorder': loss_recorder
	}


if __name__ == '__main__':
	main()