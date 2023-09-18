import os
import time
from tqdm import tqdm
from datetime import datetime
import wandb

import torch
from torchvision.utils import make_grid, save_image

from utils import *
from utils.metrics import *
# from skimage.measure import compare_psnr
from data import getLoader

Train_test = False

class Generic_train_test():
	def __init__(self, opts, accelerator, net, optimizer, scheduler, train_loader, val_loaders, datasets_name, metrics):
		self.opts = opts
		self.net = net
		self.optimizer = optimizer
		self.scheduler = scheduler

		self.l1_loss = torch.nn.L1Loss()
		self.stop_l1 = opts['train']['stop_l1'] if 'stop_l1' in opts['train'] else opts['train']['epochs']
		self.ms_ssim = MS_SSIM(accelerator)
		# self.ssim = SSIM()
		self.loss_funs = opts['train']['loss_funs']
		self.loss_weights = opts['train']['loss_weights']

		self.train_loader = train_loader
		self.val_loaders = val_loaders
		self.datasets_name = datasets_name

		# metrics
		self.best_loss = metrics['val_loss']
		self.best_ssim = metrics['val_ssim']
		self.best_psnr = metrics['val_psnr']

		# dirs
		self.checkpoint_dir = opts['Experiment']['checkpoint_dir']
		self.result_dir = opts['Experiment']['result_dir']
		self.lambda_gray = 0.5

		self.sar_trans = opts['sar_trans'] if 'sar_trans' in opts else False
		self.use_id = opts['use_id'] if 'use_id' in opts else False
		self.change_dataset = opts['train']['change_dataset'] if 'change_dataset' in opts['train'] else None


	def decode_input(self, data):
		return data
		# raise NotImplementedError()

	def train(self, accelerator, run, start_epoch, end_epoch):
		if accelerator.is_local_main_process:
			wandb.watch(self.net)
			wandb.define_metric("epoch")
			wandb.define_metric("lr", step_metric="epoch")
			metrics = ['train_loss', 'train_ssim', 'train_psnr', 'val_loss', 'val_ssim', 'val_psnr']
			for metric in metrics:
				wandb.define_metric(metric, step_metric="epoch")

		accelerator.print(f"#Train dataset: {self.datasets_name[0]}")
		accelerator.print('#Train image nums: ', len(self.train_loader)*self.opts['datasets']['train']['batch_size'])
		for epoch in range(start_epoch+1, end_epoch+1):
			cureent_epoch = f'epoch_{epoch}'
			if self.change_dataset!=None and cureent_epoch in self.change_dataset.keys():
				change_opt = self.change_dataset[cureent_epoch]
				self.train_loader = accelerator.prepare(getLoader(change_opt))
				accelerator.print('#Change dataet and image nums: ',
								  len(self.train_loader) * self.opts['datasets']['train']['batch_size'])

			batch_time = AverageMeter('Time', ':6.3f')
			data_time = AverageMeter('Data', ':6.3f')
			m_l1_loss = AverageMeter('Loss', ':.4e')
			m_ssim = AverageMeter('SSIM', ':6.2f')
			m_psnr = AverageMeter('PSNR', ':6.2f')

			if accelerator.is_local_main_process:
				wandb.log({'lr': self.optimizer.param_groups[0]["lr"], 'epoch':epoch})
			self.net.train()
			end = time.time()
			with tqdm(total=len(self.train_loader), desc=f'[Epoch {epoch}/{end_epoch}]', unit='batch',
					  disable=not accelerator.is_local_main_process) as train_pbar:
				for step, batch in enumerate(self.train_loader):
					with accelerator.accumulate(self.net):
						data_time.update(time.time() - end)

						image = batch['opt_cloudy']
						sar = batch['sar']
						label = batch['opt_clear']
						if self.use_id:
							image_id = batch['image_id']
						self.optimizer.zero_grad()
						loss_all = 0
						if self.use_id:
							pred = self.net(image, sar, image_id, accelerator)
						elif self.sar_trans:
							pred = self.net(sar)
						else:
							pred = self.net(image, sar)

						if 'pixel' in self.loss_funs.keys() and epoch < self.stop_l1:
							loss_l1 = self.l1_loss(pred, label)
							loss_all += loss_l1 * self.loss_weights[0]
						if 'ssim' in self.loss_funs.keys():
							if self.loss_funs['ssim'] == 'ms_ssim':
								loss_ssim = 1 - self.ms_ssim(pred, label)
							else:
								loss_ssim = 1 - SSIM(pred, label)
							loss_all += loss_ssim * self.loss_weights[1]

						accelerator.backward(loss_all)

						self.optimizer.step()
						# self.scheduler.step()

						# metrics
						ssim = SSIM(pred, label).item() # ?
						psnr = PSNR(pred, label)

						# loss_v = torch.mean(accelerator.gather_for_metrics(loss)).item()
						m_l1_loss.update(loss_l1.item(), image.size(0)) # average ?
						m_ssim.update(ssim, image.size(0))
						m_psnr.update(psnr, image.size(0))

						batch_time.update(time.time() - end)
						end = time.time()
						if accelerator.is_local_main_process:
							# =========== visualize results ============#
							if step % self.opts['log_step_freq'] == 0:
								total_steps = len(self.train_loader) * (epoch-1) + step + 1
								wandb.log({'loss': m_l1_loss.avg, 'ssim': m_ssim.avg, 'psnr': m_psnr.avg, 'step':total_steps})

							if step % self.opts['visual_step_freq']==0 or step==len(self.train_loader)-1:
								# figure
								img_sample = torch.cat([image.data, pred.data, label.data], -1)  # 按宽拼接
								grid = make_grid(img_sample, nrow=1, normalize=True) # 每一行显示的图像列数
								save_image(grid, os.path.join(self.result_dir, 'train_images', f'img_epoch_{epoch}_step_{step}.png'))

							# 后缀信息
							train_pbar.set_postfix(ordered_dict={'loss': m_l1_loss.avg, 'ssim': m_ssim.avg, 'psnr': m_psnr.avg})
							train_pbar.update()

						if Train_test:
							break

			# if epoch > self.opts['train']['scheduler']['lr_start_epoch_decay'] - self.opts['train']['scheduler']['lr_step']:
			self.scheduler.step()

			if accelerator.is_local_main_process:
				wandb.log({'train_loss': m_l1_loss.avg, 'train_ssim': m_ssim.avg, 'train_psnr': m_psnr.avg})

			accelerator.wait_for_everyone()

			valid_epoch_freq = self.opts['valid_epoch_freq'] if 'valid_epoch_freq' in self.opts else 1
			if epoch==start_epoch+1 or (epoch % valid_epoch_freq == 0):
				val_loss, val_ssim, val_psnr = self.validate(epoch, accelerator, run)
				metrics_dict = {'val_loss':val_loss, 'val_ssim':val_ssim, 'val_psnr':val_psnr}

			checkpoint_dict = {'epoch': epoch, 'model': accelerator.unwrap_model(self.net).state_dict(),
							   'optimizer': self.optimizer.state_dict(), 'lr_scheduler': self.scheduler.state_dict(), 'metrics': metrics_dict}

			if epoch % self.opts['save_epoch_freq'] == 0:
				accelerator.save(checkpoint_dict, os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'))

			if epoch == start_epoch or (epoch % valid_epoch_freq == 0) or (end_epoch - epoch < 5):
				update_best = val_ssim > self.best_ssim
				if update_best:
					self.best_ssim = val_ssim
					accelerator.print(f'Best valid ssim {self.best_ssim} saved at epoch {epoch}')
					accelerator.save(checkpoint_dict, os.path.join(self.checkpoint_dir, f'checkpoint_best.pth'))
			# save last
			accelerator.save(checkpoint_dict, os.path.join(self.checkpoint_dir, f'checkpoint_last.pth'))

		if accelerator.is_local_main_process:
			wandb.finish()

	@torch.no_grad()
	def validate(self, epoch, accelerator, run):
		self.net.eval()
		batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
		m_l1_loss = AverageMeter('Loss', ':.4e', Summary.NONE)
		m_ssim = AverageMeter('SSIM', ':6.2f',  Summary.AVERAGE)
		m_psnr = AverageMeter('PSNR', ':6.2f',  Summary.AVERAGE)

		end = time.time()
		for idx, val_loader in enumerate(self.val_loaders):
			correct_count = 0
			# accelerator.print(f'Validation on dataset {datasets_name[idx + 1]}:')
			#
			# with tqdm(total=len(val_loader.dataset), desc=f'Val on {self.datasets_name[idx + 1]}', unit='img',
			with tqdm(total=len(val_loader), desc=f'Val on {self.datasets_name[idx + 1]}', unit='batch',
					  disable=not accelerator.is_local_main_process) as val_pbar:
				for step, batch in enumerate(val_loader):
					image = batch['opt_cloudy']
					sar = batch['sar']
					label = batch['opt_clear']
					if self.use_id:
						image_id = batch['image_id']

					if self.use_id:
						pred = self.net(image, sar, image_id, accelerator)
					elif self.sar_trans:
						pred = self.net(sar)
					else:
						pred = self.net(image, sar)

					# Gathers tensor and potentially drops duplicates in the last batch
					all_pred, all_label = accelerator.gather_for_metrics((pred, label))

					loss_l1 = self.l1_loss(all_pred, all_label)
					# metrics
					ssim = SSIM(all_pred, all_label).item()  # ?
					# ssim = self.ssim(all_pred, all_label).item()
					psnr = PSNR(all_pred, all_label)

					m_l1_loss.update(loss_l1.item() , image.size(0))  # average ?
					m_ssim.update(ssim, image.size(0))
					m_psnr.update(psnr, image.size(0))

					batch_time.update(time.time() - end)
					end = time.time()

					if accelerator.is_local_main_process:
						# figure
						if step==0 or step % self.opts['valid_visual_step_freq'] == 0:
							img_sample = torch.cat([image.data, pred.data, label.data], -1)  # 按宽拼接
							grid = make_grid(img_sample, nrow=1, normalize=True) # 每一行显示的图像列数
							save_image(grid, os.path.join(self.result_dir, 'valid_images', f'img_epoch_{epoch}_step_{step}.png'))

						# val_pbar.update(all_label.shape[0])
						val_pbar.set_postfix(
							ordered_dict={'loss': m_l1_loss.avg, 'ssim': m_ssim.avg, 'psnr': m_psnr.avg})
						val_pbar.update()

		accelerator.wait_for_everyone()
		if accelerator.is_local_main_process:
			wandb.log({'val_loss': m_l1_loss.avg, 'val_ssim': m_ssim.avg, 'val_psnr': m_psnr.avg})

		accelerator.print(f'val_loss: {m_l1_loss.avg}, val_ssim: {m_ssim.avg}, val_ssim: {m_psnr.avg}')
		return m_l1_loss.avg, m_ssim.avg, m_psnr.avg

