import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
import hydra
import pickle
from mnh.dataset import load_datasets
from mnh.model_teacher import *
from mnh.stats import StatsLogger
from mnh.utils import *
from teacher_forward import *

from torch.utils.tensorboard import SummaryWriter

CURRENT_DIR = os.path.realpath('.')
CONFIG_DIR = os.path.join(CURRENT_DIR, 'configs')
CHECKPOINT_DIR = os.path.join(CURRENT_DIR, 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

@hydra.main(config_path=CONFIG_DIR)
def main(cfg: DictConfig):
    # Set random seed for reproduction
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Set tensorboard SummaryWriter
    TB_LOG_DIR = os.path.join(CURRENT_DIR, cfg.data.path, 'tboard_logs_teacher')
    if not os.path.exists(TB_LOG_DIR):
        os.makedirs(TB_LOG_DIR)

    # Make a separate directory for each run
    n_runs = 0
    while os.path.exists(os.path.join(TB_LOG_DIR, f'run_{n_runs}')):
        n_runs += 1
    TB_RUN_DIR = os.path.join(TB_LOG_DIR, f'run_{n_runs}')
    
    tboard_logger = SummaryWriter(TB_RUN_DIR)

    # Set device for training
    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(cfg.cuda))
    else:
        device = torch.device('cpu')

    # set DataLoader objects
    train_dataset, valid_dataset = load_datasets(os.path.join(CURRENT_DIR, cfg.data.path), cfg)
    train_loader = DataLoader(train_dataset, collate_fn=lambda x: x, shuffle=False)
    valid_loader = DataLoader(valid_dataset, collate_fn=lambda x: x, shuffle=False)

    model = get_model_from_config(cfg)
    model.to(device)

    # load checkpoints
    stats_logger = None
    optimizer_state = None   
    start_epoch = 0 

    checkpoint_path = os.path.join(CHECKPOINT_DIR, cfg.checkpoint.teacher)
    
    if cfg.train.resume and os.path.isfile(checkpoint_path):
        print('Resume from checkpoint: {}'.format(checkpoint_path))
        loaded_data = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(loaded_data['model'])
        stats_logger = pickle.loads(loaded_data['stats'])
        start_epoch = stats_logger.epoch
        optimizer_state = loaded_data['optimizer']
    else:
        # initialize plane position, rotation and size
        print('[Init] initialize plane geometry ...')
        points = train_dataset.dense_points.to(device)
        print('#points= {}'.format(points.size(0)))
        if 'replica' in cfg.data.path:
            model.plane_geo.initialize_with_box(
                points, 
                lrf_neighbors=cfg.model.init.lrf_neighbors,
                wh=cfg.model.init.wh,
                box_factor=cfg.model.init.box_factor, 
                random_rate=cfg.model.init.random_rate,
            )
        else:
            model.plane_geo.initialize(
                points,
                lrf_neighbors=cfg.model.init.lrf_neighbors,
                wh=cfg.model.init.wh,
            )
        del points 
        torch.cuda.empty_cache()

    # set optimizer 
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=cfg.optimizer.lr
    )
    if optimizer_state != None:
        optimizer.load_state_dict(optimizer_state)
        optimizer.last_epoch = start_epoch
    
    def lr_lambda(epoch):
        return cfg.optimizer.lr_scheduler_gamma ** (
            epoch / cfg.optimizer.lr_scheduler_step_size
        )

    # The learning rate scheduling is implemented with LambdaLR PyTorch scheduler.
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda, last_epoch=start_epoch - 1, verbose=False
    )

    # set StatsLogger objects
    if stats_logger == None:
        stats_logger = StatsLogger()
    
    img_folder = os.path.join(CURRENT_DIR, 'output_images', cfg.name, 'teacher', 'output')
    os.makedirs(img_folder, exist_ok=True)
    print('[Traing Teacher]')
    for epoch in range(start_epoch, cfg.train.epoch.teacher):
        model.train()
        stats_logger.new_epoch()

        # Average the losses for each epoch
        mse_color = 0
        mse_point2plane = 0
        loss_area = 0
        psnr = 0 
        ssim = 0

        for i, data in enumerate(train_loader):
            data = data[0]

            train_stats, train_images = forward_pass(
                data, 
                model,
                device,
                cfg, 
                optimizer,
                training=True,
            )
            mse_color += train_stats['mse_color']
            mse_point2plane += train_stats['mse_point2plane']
            loss_area += train_stats['loss_area']
            psnr += train_stats['psnr']
            ssim += train_stats['ssim']
            stats_logger.update('train', train_stats)

        # Add results to tensorboard
        tboard_logger.add_scalar('loss/mse_color', mse_color / len(train_loader), epoch)
        tboard_logger.add_scalar('loss/mse_point2plane', mse_point2plane / len(train_loader), epoch)
        tboard_logger.add_scalar('loss/loss_area', loss_area / len(train_loader), epoch)
        tboard_logger.add_scalar('eval/psnr', psnr / len(train_loader), epoch)
        tboard_logger.add_scalar('eval/ssim', ssim / len(train_loader), epoch)
        
        stats_logger.print_info('train')
        lr_scheduler.step()

        # Checkpoint
        if (epoch+1) % cfg.train.epoch.checkpoint == 0:
            print('store checkpoints ...')
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'stats': pickle.dumps(stats_logger)
            }
            torch.save(checkpoint, checkpoint_path)

        # validation
        if (epoch+1) % cfg.train.epoch.validation == 0:
            model.eval()
            for i, data in enumerate(valid_loader):
                data = data[0]
                valid_stats, valid_images = forward_pass(
                    data, 
                    model,
                    device,
                    cfg,
                    training=False,
                )
                stats_logger.update('valid', valid_stats)

                for key, img in valid_images.items():
                    if 'depth' in key:
                        img = img / img.max()
                    img = tensor2Image(img)
                    path = os.path.join(img_folder, 'valid-{:0>5}-{}.png'.format(i, key))
                    img.save(path)
                
                # Add image to tensorboard
                if i == 0:
                    pred_img = valid_images['color_pred']
                    pred_img = np.asarray(tensor2Image(pred_img))
                    print(f'##### pred_img.shape: {pred_img.shape}')

                    gt_img = valid_images['color_gt']
                    gt_img = np.asarray(tensor2Image(gt_img))

                    tboard_logger.add_image('pred_image', pred_img, epoch, dataformats='HWC')
                    tboard_logger.add_image('gt_image', gt_img, epoch, dataformats='HWC')
                    

            stats_logger.print_info('valid')

if __name__ == '__main__':
    main()