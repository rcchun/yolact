from Yolact_res101.data import *
from Yolact_res101.utils.augmentations import SSDAugmentation, BaseTransform, Resize, RandomRot90
from Yolact_res101.utils.functions import MovingAverage, SavePath
from Yolact_res101.utils.logger import Log
from Yolact_res101.utils import timer
from Yolact_res101.layers.modules import MultiBoxLoss
from Yolact_res101.yolact import Yolact
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
import json
import os
import sys
import time
import math, random
from pathlib import Path
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import datetime

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from sqlalchemy import create_engine, MetaData, select, and_

db = create_engine('postgresql://postgres:postgres@192.168.10.22:5433/AIMaps_2.0')
metadata = MetaData()
metadata.reflect(bind=db)
# Oof
import Yolact_res101.eval as eval_script

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Yolact Training Script')
parser.add_argument('--batch_size', default=4, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from. If this is "interrupt"' \
                         ', the model will resume training from the interrupt file.')
parser.add_argument('--start_iter', default=-1, type=int,
                    help='Resume training at this iter. If this is -1, the iteration will be' \
                         'determined from the file name.')
parser.add_argument('--num_workers', default=1, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning_rate', default=0.001, type=float,
                    help='Initial learning rate. Leave as None to read this from the config.')
parser.add_argument('--momentum', default=None, type=float,
                    help='Momentum for SGD. Leave as None to read this from the config.')
parser.add_argument('--decay', '--weight_decay', default=None, type=float,
                    help='Weight decay for SGD. Leave as None to read this from the config.')
parser.add_argument('--gamma', default=None, type=float,
                    help='For each lr step, what to multiply the lr by. Leave as None to read this from the config.')
parser.add_argument('--save_folder', default='Yolact_res101/weight/',
                    help='Directory for saving checkpoint models.')
parser.add_argument('--log_folder', default='Yolact_res101/logs/',
                    help='Directory for saving logs.')
parser.add_argument('--config', default='yolact_base_config',
                    help='The config object to use.')
parser.add_argument('--save_interval', default=20, type=int,
                    help='The number of iterations between saving the model.')
parser.add_argument('--validation_size', default=5000, type=int,
                    help='The number of images to use for validation.')
parser.add_argument('--validation_epoch', default=10, type=int,
                    help='Output validation information every n iterations. If -1, do no validation.')
parser.add_argument('--keep_latest', dest='keep_latest', action='store_true',
                    help='Only keep the latest checkpoint instead of each one.')
parser.add_argument('--keep_latest_interval', default=100000, type=int,
                    help='When --keep_latest is on, don\'t delete the latest file at these intervals. This should be a multiple of save_interval or 0.')
parser.add_argument('--dataset', default=None, type=str,
                    help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
parser.add_argument('--no_log', dest='log', action='store_false',
                    help='Don\'t log per iteration information into log_folder.')
parser.add_argument('--log_gpu', dest='log_gpu', action='store_true',
                    help='Include GPU information in the logs. Nvidia-smi tends to be slow, so set this with caution.')
parser.add_argument('--no_interrupt', dest='interrupt', action='store_false',
                    help='Don\'t save an interrupt when KeyboardInterrupt is caught.')
parser.add_argument('--batch_alloc', default=None, type=str,
                    help='If using multiple GPUS, you can set this to be a comma separated list detailing which GPUs should get what local batch size (It should add up to your total batch size).')
parser.add_argument('--no_autoscale', dest='autoscale', action='store_false',
                    help='YOLACT will automatically scale the lr and the number of iterations depending on the batch size. Set this if you want to disable that.')

parser.set_defaults(keep_latest=False, log=True, log_gpu=False, interrupt=True, autoscale=True)
args = parser.parse_args()

if args.config is not None:
    set_cfg(args.config)

if args.dataset is not None:
    set_dataset(args.dataset)

if args.autoscale and args.batch_size != 8:
    factor = args.batch_size / 8
    if __name__ == '__main__':
        print('Scaling parameters by %.2f to account for a batch size of %d.' % (factor, args.batch_size))

    cfg.lr *= factor
    cfg.max_iter //= factor
    cfg.lr_steps = [x // factor for x in cfg.lr_steps]


# Update training parameters from the config if necessary
def replace(name):
    if getattr(args, name) == None: setattr(args, name, getattr(cfg, name))


replace('lr')
replace('decay')
replace('gamma')
replace('momentum')

# This is managed by set_lr
cur_lr = args.lr

if torch.cuda.device_count() == 0:
    print('No GPUs detected. Exiting...')
    exit(-1)

if args.batch_size // torch.cuda.device_count() < 6:
    if __name__ == '__main__':
        print('Per-GPU batch size is less than the recommended limit for batch norm. Disabling batch norm.')
    cfg.freeze_bn = True

loss_types = ['B', 'C', 'M', 'P', 'D', 'E', 'S', 'I']

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


class NetLoss(nn.Module):
    """
    A wrapper for running the network and computing the loss
    This is so we can more efficiently use DataParallel.
    """

    def __init__(self, net: Yolact, criterion: MultiBoxLoss):
        super().__init__()

        self.net = net
        self.criterion = criterion

    def forward(self, images, targets, masks, num_crowds):
        preds = self.net(images)
        losses = self.criterion(self.net, preds, targets, masks, num_crowds)
        return losses


class CustomDataParallel(nn.DataParallel):
    """
    This is a custom version of DataParallel that works better with our training data.
    It should also be faster than the general case.
    """

    def scatter(self, inputs, kwargs, device_ids):
        # More like scatter and data prep at the same time. The point is we prep the data in such a way
        # that no scatter is necessary, and there's no need to shuffle stuff around different GPUs.
        devices = ['cuda:' + str(x) for x in device_ids]
        splits = prepare_data(inputs[0], devices, allocation=args.batch_alloc)

        return [[split[device_idx] for split in splits] for device_idx in range(len(devices))], \
               [kwargs] * len(devices)

    def gather(self, outputs, output_device):
        out = {}

        for k in outputs[0]:
            out[k] = torch.stack([output[k].to(output_device) for output in outputs])

        return out


def get_coco_train_format(img_dir):
    json_file = img_dir + '/' + "datasets_train.json"
    record = OrderedDict()
    annot_id = 0
    with open(json_file) as f:
        imgs_anns = json.load(f)
        record = {}
        record["images"] = []
        record["annotations"] = []
        for i in range(len(imgs_anns)):
            filename = imgs_anns[i]["file_name"]
            height, width = cv2.imread(filename).shape[:2]
            record["images"].append(
                {
                    "id": i,
                    "width": width,
                    "height": height,
                    "file_name": filename.split('/')[-1]
                },
            )
            annos = imgs_anns[i]["annotations"]
            for j in range(len(annos)):
                category_id = annos[j]["category_id"]
                segmentation = annos[j]["segmentation"]
                bbox = annos[j]['bbox']
                record["annotations"].append(
                    {
                        "id": annot_id,
                        "image_id": i,
                        "category_id": category_id + 1,
                        "segmentation": segmentation,
                        "bbox": bbox,
                        "iscrowd": 0,
                    }
                )
                annot_id += 1
        with open(img_dir + '/datasets_train_coco.json', 'w', encoding="utf-8") as make_file:
            json.dump(record, make_file, ensure_ascii=False, indent="\t")
    return record


def get_coco_test_format(img_dir):
    json_file = img_dir + '/' + "datasets_test.json"
    record = OrderedDict()
    annot_id = 0
    with open(json_file) as f:
        imgs_anns = json.load(f)
        record = {}
        record["images"] = []
        record["annotations"] = []
        for i in range(len(imgs_anns)):
            filename = imgs_anns[i]["file_name"]
            height, width = cv2.imread(filename).shape[:2]
            record["images"].append(
                {
                    "id": i,
                    "width": width,
                    "height": height,
                    "file_name": filename.split('/')[-1]
                },
            )
            annos = imgs_anns[i]["annotations"]
            for j in range(len(annos)):
                category_id = annos[j]["category_id"]
                segmentation = annos[j]["segmentation"]
                bbox = annos[j]['bbox']
                record["annotations"].append(
                    {
                        "id": annot_id,
                        "image_id": i,
                        "category_id": category_id + 1,
                        "segmentation": segmentation,
                        "bbox": bbox,
                        "iscrowd": 0,
                    }
                )
                annot_id += 1
        with open(img_dir + '/datasets_test_coco.json', 'w', encoding="utf-8") as make_file:
            json.dump(record, make_file, ensure_ascii=False, indent="\t")
    return record


def get_balloon_dicts(img_dir, dats_id):
    pf_train_dataset = metadata.tables['pf_train_dataset']
    stmt_dir = select([pf_train_dataset.c.annotation_json]).where(pf_train_dataset.c.tr_dats_id == dats_id)
    conn = db.connect()
    json_file = conn.execute(stmt_dir).fetchone()[0]

    dataset_dicts = []
    for idx, v in enumerate(json_file.values()):
        record = {}

        filename = img_dir + '/' + v["filename"]
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = v["regions"]
        objs = []
        for i in range(len(annos)):
            class_ = annos[i]["region_attributes"]
            anno = annos[i]["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "segmentation": [poly],
                "iscrowd": 0,
                "category_id": int(class_['classification']) - 1,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def train(tr_wk_id, batch, epoch, lr, dats_id, dats_rate, angle, resolution, NUM_CLASSES, data_directory):
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    if not os.path.exists(args.save_folder + '{}/'.format(tr_wk_id)):
        os.mkdir(args.save_folder + '{}/'.format(tr_wk_id))

    writer = SummaryWriter()
    # parameter application (batch / learning rate)

    args.batch_size = batch
    args.lr = lr
    args.save_folder = args.save_folder + '/{}'.format(tr_wk_id)

    # dataset directory application
    conn = db.connect()
    data_directory = data_directory
    cfg.num_classes = NUM_CLASSES + 1

    # train dataset enrollment(train/test ratio)

    obj = get_balloon_dicts(data_directory, dats_id)
    datasets = {}
    val_split = dats_rate
    datasets['train'], datasets['test'] = train_test_split(obj, test_size=val_split)
    file_data_train = datasets['train']
    file_data_test = datasets['test']
    with open(data_directory + 'datasets_train.json', 'w', encoding="utf-8") as make_file_train:
        json.dump(file_data_train, make_file_train, ensure_ascii=False, indent="\t")
    with open(data_directory + 'datasets_test.json', 'w', encoding="utf-8") as make_file_test:
        json.dump(file_data_test, make_file_test, ensure_ascii=False, indent="\t")
    get_coco_train_format(data_directory)
    get_coco_test_format(data_directory)

    # parameter application (epoch / trainset / testset)
    cfg.max_size = resolution
    cfg.max_iter = epoch
    cfg.dataset.train_images = data_directory
    cfg.dataset.train_info = data_directory + 'datasets_train_coco.json'
    cfg.dataset.valid_images = data_directory
    cfg.dataset.valid_info = data_directory + 'datasets_test_coco.json'

    dataset = COCODetection(image_path=cfg.dataset.train_images,
                            info_file=cfg.dataset.train_info,
                            transform=SSDAugmentation(MEANS))

    if args.validation_epoch > 0:
        setup_eval()
        val_dataset = COCODetection(image_path=cfg.dataset.valid_images,
                                    info_file=cfg.dataset.valid_info,
                                    transform=BaseTransform(MEANS))
    # angle 처리
    if angle is None:
        cfg.augment_random_rot90 = False
        cfg.augment_random_rot180 = False
        cfg.augment_random_rot270 = False
    ang_list = []
    len_ang = len(angle.split(','))
    for i in range(len_ang):
        ang_list.append(int(angle.split(',')[i]))
    for ang in ang_list:
        if ang == 90:
            cfg.augment_random_rot90 = True
        if ang == 180:
            cfg.augment_random_rot180 = True
        if ang == 270:
            cfg.augment_random_rot270 = True
    print(cfg.augment_random_rot90, cfg.augment_random_rot180, cfg.augment_random_rot270)

    # Parallel wraps the underlying module, but when saving and loading we don't want that
    yolact_net = Yolact()
    net = yolact_net
    net.train()

    if args.log:
        log = Log(cfg.name, args.log_folder, dict(args._get_kwargs()),
                  overwrite=(args.resume is None), log_gpu_stats=args.log_gpu)

    # I don't use the timer during training (I use a different timing method).
    # Apparently there's a race condition with multiple GPUs, so disable it just to be safe.
    timer.disable_all()

    # Both of these can set args.resume to None, so do them before the check
    if args.resume == 'interrupt':
        args.resume = SavePath.get_interrupt(args.save_folder)
    elif args.resume == 'latest':
        args.resume = SavePath.get_latest(args.save_folder, cfg.name)

    if args.resume is not None:
        print('Resuming training, loading {}...'.format(args.resume))
        yolact_net.load_weights(args.resume)

        if args.start_iter == -1:
            args.start_iter = SavePath.from_str(args.resume).iteration
    else:
        print('Initializing weights...')
        yolact_net.init_weights(backbone_path='Yolact_res101/weight/' + cfg.backbone.path)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.decay)
    criterion = MultiBoxLoss(num_classes=cfg.num_classes,
                             pos_threshold=cfg.positive_iou_threshold,
                             neg_threshold=cfg.negative_iou_threshold,
                             negpos_ratio=cfg.ohem_negpos_ratio)

    if args.batch_alloc is not None:
        args.batch_alloc = [int(x) for x in args.batch_alloc.split(',')]
        if sum(args.batch_alloc) != args.batch_size:
            print('Error: Batch allocation (%s) does not sum to batch size (%s).' % (args.batch_alloc, args.batch_size))
            exit(-1)

    net = CustomDataParallel(NetLoss(net, criterion))
    if args.cuda:
        net = net.cuda()

    # Initialize everything
    if not cfg.freeze_bn: yolact_net.freeze_bn()  # Freeze bn so we don't kill our means
    yolact_net(torch.zeros(1, 3, cfg.max_size, cfg.max_size).cuda())
    if not cfg.freeze_bn: yolact_net.freeze_bn(True)

    # loss counters
    loc_loss = 0
    conf_loss = 0
    iteration = max(args.start_iter, 0)
    last_time = time.time()

    epoch_size = len(dataset) // args.batch_size
    num_epochs = math.ceil(cfg.max_iter / epoch_size)
    print('***num_epochs : ', num_epochs, ' ***')
    # Which learning rate adjustment step are we on? lr' = lr * gamma ^ step_index
    step_index = 0

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=False, collate_fn=detection_collate,
                                  pin_memory=True)

    save_path = lambda iteration: SavePath(cfg.name, iteration).get_path(root=args.save_folder)
    time_avg = MovingAverage()

    global loss_types  # Forms the print order
    loss_avgs = {k: MovingAverage(100) for k in loss_types}

    print('Begin training!')
    print()
    # try-except so you can use ctrl+c to save early and stop training
    try:
        for epoch in range(num_epochs):
            # Resume from start_iter
            if (epoch + 1) * epoch_size < iteration:
                continue

            for datum in data_loader:
                # Stop if we've reached an epoch if we're resuming from start_iter
                if iteration == (epoch + 1) * epoch_size:
                    break

                # Stop at the configured number of iterations even if mid-epoch
                if iteration == cfg.max_iter:
                    break

                # Change a config setting if we've reached the specified iteration
                changed = False
                for change in cfg.delayed_settings:
                    if iteration >= change[0]:
                        changed = True
                        cfg.replace(change[1])

                        # Reset the loss averages because things might have changed
                        for avg in loss_avgs:
                            avg.reset()

                # If a config setting was changed, remove it from the list so we don't keep checking
                if changed:
                    cfg.delayed_settings = [x for x in cfg.delayed_settings if x[0] > iteration]

                # Warm up by linearly interpolating the learning rate from some smaller value
                if cfg.lr_warmup_until > 0 and iteration <= cfg.lr_warmup_until:
                    set_lr(optimizer,
                           (args.lr - cfg.lr_warmup_init) * (iteration / cfg.lr_warmup_until) + cfg.lr_warmup_init)

                # Adjust the learning rate at the given iterations, but also if we resume from past that iteration
                while step_index < len(cfg.lr_steps) and iteration >= cfg.lr_steps[step_index]:
                    step_index += 1
                    set_lr(optimizer, args.lr * (args.gamma ** step_index))

                # Zero the grad to get ready to compute gradients
                optimizer.zero_grad()

                # Forward Pass + Compute loss at the same time (see CustomDataParallel and NetLoss)
                losses = net(datum)

                losses = {k: (v).mean() for k, v in losses.items()}  # Mean here because Dataparallel
                loss = sum([losses[k] for k in losses])

                # no_inf_mean removes some components from the loss, so make sure to backward through all of it
                # all_loss = sum([v.mean() for v in losses.values()])

                # Backprop
                loss.backward()  # Do this to free up vram even if loss is not finite
                if torch.isfinite(loss).item():
                    optimizer.step()

                # Add the loss to the moving average for bookkeeping
                for k in losses:
                    loss_avgs[k].add(losses[k].item())

                cur_time = time.time()
                elapsed = cur_time - last_time
                last_time = cur_time

                # Exclude graph setup from the timing information
                if iteration != args.start_iter:
                    time_avg.add(elapsed)

                # if iteration % 10 == 0:
                eta_str = str(datetime.timedelta(seconds=(cfg.max_iter - iteration) * time_avg.get_avg())).split('.')[0]

                total = sum([loss_avgs[k].get_avg() for k in losses])
                loss_labels = sum([[k, loss_avgs[k].get_avg()] for k in loss_types if k in losses], [])

                print(('[%3d] %7d ||' + (' %s: %.3f |' * len(losses)) + ' T: %.3f || ETA: %s || timer: %.3f')
                      % tuple([epoch, iteration] + loss_labels + [total, eta_str, elapsed]), flush=True)

                # if args.log:
                precision = 5
                loss_info = {k: round(losses[k].item(), precision) for k in losses}
                loss_info['T'] = round(loss.item(), precision)

                if args.log_gpu:
                    log.log_gpu_stats = (iteration % 10 == 0)  # nvidia-smi is sloooow

                log.log('train', loss=loss_info, epoch=epoch, iter=iteration,
                        lr=round(cur_lr, 10), elapsed=elapsed)
                print('Loss : ', loss.item())
                avg_loss = loss.item()
                if loss.item() > 99999:
                    avg_loss = 99999

                progress_rate = int((iteration / cfg.max_iter) * 100) + 1
                pf_train_progress = metadata.tables['pf_train_progress']
                pf_train = metadata.tables['pf_train']
                udt = pf_train.update().where(pf_train.c.tr_wk_id == tr_wk_id).values(
                    iterator=iteration,
                    progress_rate=progress_rate,
                    avg_loss=avg_loss,
                    remaining_time=eta_str
                )
                ins = pf_train_progress.insert().values(tr_wk_id=tr_wk_id,
                                                        iterator=iteration,
                                                        progress_rate=progress_rate,
                                                        avg_loss=avg_loss,
                                                        remaining_time=eta_str)
                conn = db.connect()
                conn.execute(ins)
                conn.execute(udt)

                log.log_gpu_stats = args.log_gpu
                # writer.add_scalar("Loss/train", loss.item(), iteration)
                if iteration % args.validation_epoch == 0 and iteration > 0:
                    mAP_ = compute_validation_map(epoch, iteration, tr_wk_id, yolact_net, val_dataset,
                                           log if args.log else None)
                    # writer.add_scalar("mAP/train", mAP_, iteration)
                iteration += 1

                if iteration % args.save_interval == 0 and iteration != args.start_iter:
                    if args.keep_latest:
                        latest = SavePath.get_latest(args.save_folder, cfg.name)

                    print('Saving state, iter:', iteration)
                    yolact_net.save_weights(save_path(iteration))

                    if args.keep_latest and latest is not None:
                        if args.keep_latest_interval <= 0 or iteration % args.keep_latest_interval != args.save_interval:
                            print('Deleting old save...')
                            os.remove(latest)

            # force-pause
            pf_train = metadata.tables['pf_train']
            stmt = select([pf_train.c.force_pause]).where(pf_train.c.tr_wk_id == tr_wk_id)
            conn = db.connect()
            res = conn.execute(stmt).fetchone()
            if res[0]:
                ins = pf_train.update().where(pf_train.c.tr_wk_id == tr_wk_id).values(
                    status=3)
                conn = db.connect()
                conn.execute(ins)
                break
            # This is done per epoch
            # if args.validation_epoch > 0:
                # if epoch % args.validation_epoch == 0 and epoch > 0:
                    # compute_validation_map(epoch, iteration, tr_wk_id, yolact_net, val_dataset, log if args.log else None)

        # Compute validation mAP after training is finished
        compute_validation_map(epoch, iteration, tr_wk_id, yolact_net, val_dataset, log if args.log else None)

    except KeyboardInterrupt:
        if args.interrupt:
            print('Stopping early. Saving network...')

            # Delete previous copy of the interrupted network so we don't spam the weights folder
            SavePath.remove_interrupt(args.save_folder)

            yolact_net.save_weights(save_path(repr(iteration) + '_interrupt'))
        exit()

    yolact_net.save_weights(save_path(iteration))

    # weight_file 경로 insert
    pf_train = metadata.tables['pf_train']
    weight_dir_upd = pf_train.update().where(pf_train.c.tr_wk_id == tr_wk_id).values(
        weight_path='Yolact_res101/weight/{}/yolact_base_{}.pth'.format(tr_wk_id, cfg.max_iter))
    conn.execute(weight_dir_upd)

def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

    global cur_lr
    cur_lr = new_lr


def gradinator(x):
    x.requires_grad = False
    return x


def prepare_data(datum, devices: list = None, allocation: list = None):
    with torch.no_grad():
        if devices is None:
            devices = ['cuda:0'] if args.cuda else ['cpu']
        if allocation is None:
            allocation = [args.batch_size // len(devices)] * (len(devices) - 1)
            allocation.append(args.batch_size - sum(allocation))  # The rest might need more/less

        images, (targets, masks, num_crowds) = datum

        cur_idx = 0
        for device, alloc in zip(devices, allocation):
            for _ in range(alloc):
                images[cur_idx] = gradinator(images[cur_idx].to(device))
                targets[cur_idx] = gradinator(targets[cur_idx].to(device))
                masks[cur_idx] = gradinator(masks[cur_idx].to(device))
                cur_idx += 1

        if cfg.preserve_aspect_ratio:
            # Choose a random size from the batch
            _, h, w = images[random.randint(0, len(images) - 1)].size()

            for idx, (image, target, mask, num_crowd) in enumerate(zip(images, targets, masks, num_crowds)):
                images[idx], targets[idx], masks[idx], num_crowds[idx] \
                    = enforce_size(image, target, mask, num_crowd, w, h)

        cur_idx = 0
        split_images, split_targets, split_masks, split_numcrowds \
            = [[None for alloc in allocation] for _ in range(4)]
        for device_idx, alloc in enumerate(allocation):
            split_images[device_idx] = torch.stack(images[cur_idx:cur_idx + alloc], dim=0)
            split_targets[device_idx] = targets[cur_idx:cur_idx + alloc]
            split_masks[device_idx] = masks[cur_idx:cur_idx + alloc]
            split_numcrowds[device_idx] = num_crowds[cur_idx:cur_idx + alloc]
            cur_idx += alloc

        return split_images, split_targets, split_masks, split_numcrowds


def no_inf_mean(x: torch.Tensor):
    """
    Computes the mean of a vector, throwing out all inf values.
    If there are no non-inf values, this will return inf (i.e., just the normal mean).
    """

    no_inf = [a for a in x if torch.isfinite(a)]

    if len(no_inf) > 0:
        return sum(no_inf) / len(no_inf)
    else:
        return x.mean()


def compute_validation_loss(net, data_loader, criterion):
    global loss_types

    with torch.no_grad():
        losses = {}

        # Don't switch to eval mode because we want to get losses
        iterations = 0
        for datum in data_loader:
            images, targets, masks, num_crowds = prepare_data(datum)
            out = net(images)

            wrapper = ScatterWrapper(targets, masks, num_crowds)
            _losses = criterion(out, wrapper, wrapper.make_mask())

            for k, v in _losses.items():
                v = v.mean().item()
                if k in losses:
                    losses[k] += v
                else:
                    losses[k] = v

            iterations += 1
            if args.validation_size <= iterations * args.batch_size:
                break

        for k in losses:
            losses[k] /= iterations

        loss_labels = sum([[k, losses[k]] for k in loss_types if k in losses], [])
        print(('Validation ||' + (' %s: %.3f |' * len(losses)) + ')') % tuple(loss_labels), flush=True)


def compute_validation_map(epoch, iteration, tr_wk_id, yolact_net, dataset, log: Log = None):
    with torch.no_grad():
        yolact_net.eval()

        start = time.time()
        print()
        print("Computing validation mAP (this may take a while)...", flush=True)
        val_info = eval_script.evaluate_train(tr_wk_id, yolact_net, dataset, train_mode=True)
        end = time.time()
        print('val_info : ', val_info)
        print('mAP : ', val_info['box'][50])
        _mAP = val_info['box'][50] / 100
        pf_train_progress = metadata.tables['pf_train_progress']
        pf_train = metadata.tables['pf_train']
        conn = db.connect()

        iter_ = iteration - 1
        udt = pf_train.update().where(pf_train.c.tr_wk_id == tr_wk_id).values(map=_mAP)
        ins = pf_train_progress.update().where(and_(pf_train_progress.c.tr_wk_id == tr_wk_id,
                                                    pf_train_progress.c.iterator == iter_)).values(map=_mAP)
        conn.execute(ins)
        conn.execute(udt)

        if log is not None:
            log.log('val', val_info, elapsed=(end - start), epoch=epoch, iter=iteration)

        yolact_net.train()
        return val_info['box'][50]

def setup_eval():
    eval_script.parse_args(['--no_bar', '--max_images=' + str(args.validation_size)])

# if __name__ == '__main__':
#     train()
