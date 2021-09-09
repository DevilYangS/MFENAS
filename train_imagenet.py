import argparse, logging, collections
import random, time, sys,os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as dset

from utils import dagnode,create__dir, count_parameters_in_MB
import utils
from Node import NetworkImageNet
from autoaugment import  ImageNetPolicy

import torch.distributed as dist


from folder2lmdb import ImageFolderLMDB
from prefetch_generator import BackgroundGenerator



os.environ['CUDA_VISIBLE_DEVICES']='0,1'



class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())





class individual():
    def __init__(self, dec):
        #dec
        #dag
        #num_node
        self.dec = dec
        self.re_duplicate()
        #self.trans2bin()# if dec is (int10,op)
        self.trans2dag()

    # def trans2bin(self):
    #     self.bin_dec = []
    #     self.conv_bin_dec = []
    #     self.redu_bin_dec =[]
    #
    #     for i in range(2):
    #         temp_dec = []
    #         for j in range(int(len(self.dec[i])/2)):
    #             bin_value = bin(self.dec[i][2*j])
    #             temp_list = [int(i) for i in bin_value[2:] ]
    #             if len(temp_list)<j+2:
    #                 A = [0]*(j+2 - len(temp_list))
    #                 A.extend(temp_list)
    #                 temp_list = A.copy()
    #             temp_list.extend([self.dec[i][2*j+1]])
    #             temp_dec.append(temp_list)
    #         self.bin_dec.append(temp_dec)
    #
    #     temp = [self.conv_bin_dec.extend(i) for i in self.bin_dec[0]]
    #     del temp
    #     temp = [self.redu_bin_dec.extend(i) for i in self.bin_dec[1]]
    #     del temp
    def re_duplicate(self):
        #used for deleting the nodes not actived

        for i,cell_dag in enumerate(self.dec):
            L = 0
            j = 0
            zero_index = []
            temp_dec = []
            while L <len(cell_dag):
                S = L
                L +=3+j
                node_j_A = np.array(cell_dag[S:L]).copy()
                node_j = node_j_A[:-1]
                if node_j.sum()- node_j[zero_index].sum()==0:
                    zero_index.extend([j+2])
                else:
                    temp_dec.extend(np.delete(node_j_A, zero_index))
                j+=1

            self.dec[i] = temp_dec.copy()

    def trans2dag(self):
        self.dag = []
        self.num_node = []
        for i in range(2):
            dag = collections.defaultdict(list)
            dag[-1] = dagnode(-1, [], None)
            dag[0] = dagnode(0, [0], None)


            j = 0
            L = 0
            while L < len(self.dec[i]):
                S = L
                L += 3+j
                node_j = self.dec[i][S:L]
                dag[j+1] = dagnode(j+1,node_j[:-1],node_j[-1])
                j+=1
            self.num_node.extend([j])
            self.dag.append(dag)
            del dag

class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss

def train(train_queue, model, train_criterion, optimizer, args,epoch,global_step,since_time):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()
    total = len(train_queue)
    data_time = utils.AvgrageMeter()
    batch_time = utils.AvgrageMeter()
    end = time.time()

    for step, (inputs, targets) in enumerate(train_queue):

        data_time.update(time.time() - end)


        # inputs, targets = inputs.to(args.device,non_blocking=True), targets.to(args.device,non_blocking=True)
        inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
        optimizer.zero_grad()

        outputs = model(inputs,global_step[0])
        global_step[0] += 1

        if args.use_aux_head:
            outputs, outputs_aux = outputs[0], outputs[1]

        loss = train_criterion(outputs, targets)
        if args.use_aux_head:
            loss_aux = train_criterion(outputs_aux, targets)
            loss += args.auxiliary_weight * loss_aux

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
        n = inputs.size(0)
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)

        batch_time.update(time.time() - end)
        end = time.time()

        # print('\r  Epoch{0:>2d}/250, Training {1:>2d}/{2:>2d}, data time:{3:.4f}s, batch time:{4:.4f}s, total_used_time {5:.3f}min]'.
        #       format(epoch,step + 1, total, data_time.avg,batch_time.avg,(time.time() - since_time)/60 ),end='')
        print('\r  Epoch{0:>2d}/250, Training {1:>2d}/{2:>2d}, data time:{3}s, batch time:{4}s, total_used_time {5:.3f}min]'.
              format(epoch,step + 1, total, data_time._print,batch_time._print,(time.time() - since_time)/60 ),end='')

    return top1.avg, top5.avg, objs.avg

def valid(valid_queue, model, eval_criterion,args):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    with torch.no_grad():
        model.eval()
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda()

            outputs = model(input)

            if args.use_aux_head:
                outputs, outputs_aux = outputs[0], outputs[1]

            loss = eval_criterion(outputs, target)

            prec1, prec5 = utils.accuracy(outputs, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)

    return top1.avg, top5.avg, objs.avg



def adjust_lr(optimizer, epoch):
    # Smaller slope for the last 5 epochs because lr * 1/250 is relatively large
    if args.epochs -  epoch > 5:
        lr = args.lr * (args.epochs - 5 - epoch) / (args.epochs - 5)
    else:
        lr = args.lr * (args.epochs - epoch) / ((args.epochs - 5) * 5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def adjust_lr_(optimizer, epoch):
    # Smaller slope for the last 55 epochs because lr * 1/250 is relatively large
    if args.epochs -  epoch > 55:
        lr = args.lr * (args.epochs - 5 - epoch) / (args.epochs - 5)
    else:
        lr = args.lr * (args.epochs - epoch) / ((args.epochs - 5) * 5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr



def build_imagenet(model_state_dict, optimizer_state_dict, **kwargs):

    solution = kwargs.pop('solution')
    epoch = kwargs.pop('epoch')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if args.autoaugment:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            ImageNetPolicy(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2),
            transforms.ToTensor(),
            normalize,
        ])

    valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    if args.load=='lmdb':

        logging.info('Loading data from lmdb file')
        traindir = os.path.join(args.data, 'train.lmdb')
        validdir = os.path.join(args.data, 'val.lmdb')
        print('LMDB data is used :'
              'https://github.com/xunge/pytorch_lmdb_imagenet')

        train_data = ImageFolderLMDB(traindir, train_transform)
        valid_data = ImageFolderLMDB(validdir, valid_transform)

    elif args.load=='original':

        logging.info('Loading data from directory')
        traindir = os.path.join(args.data, 'train')
        validdir = os.path.join(args.data, 'val')

        train_data = dset.ImageFolder(traindir, train_transform)
        valid_data = dset.ImageFolder(validdir, valid_transform)

    elif args.load=='memory':

        logging.info('Loading data into memory')
        traindir = os.path.join(args.data, 'train')
        validdir = os.path.join(args.data, 'val')
        train_data = utils.InMemoryDataset(traindir, train_transform, num_workers=args.num_workers)
        valid_data = utils.InMemoryDataset(validdir, valid_transform, num_workers=args.num_workers)

    logging.info('Found %d in training data', len(train_data))
    logging.info('Found %d in validation data', len(valid_data))

    #------------------------------------------------ steps -------------------------------------------------
    args.steps = int(np.ceil(len(train_data) / (args.batch_size))) * torch.cuda.device_count() * args.epochs
    #---------------------------------------------------------------------------------------------------------



    train_queue = DataLoaderX(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    valid_queue = DataLoaderX(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    # 0.00005s for DataLoaderX each batch=256
    # 0.0003s  for DataLoader  each batch=256

    # train_queue = torch.utils.data.DataLoader(
    #     train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    # valid_queue = torch.utils.data.DataLoader(
    #     valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)



    model = NetworkImageNet(args, args.classes, args.layers, args.channels, solution.dag, args.use_aux_head,
                            args.keep_prob,args.steps,args.drop_path_keep_prob,args.channel_double)

    logging.info("param size = %fMB", count_parameters_in_MB(model))
    print('Model Parameters: {} MB'.format(count_parameters_in_MB(model)))

    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)


    if torch.cuda.device_count() > 1:
        logging.info("Use %d %s", torch.cuda.device_count(), "GPUs !")
        model = nn.DataParallel(model)
        # model = torch.nn.parallel.DistributedDataParallel(model)


    model = model.cuda()

    train_criterion = CrossEntropyLabelSmooth(args.classes, args.label_smooth).cuda()
    eval_criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.l2_reg,
    )
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    if not args.warm_up:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, args.gamma, epoch)
    else:
        # linear or cosine is used for warm up training
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs,last_epoch=epoch)

    return train_queue, valid_queue, model, train_criterion, eval_criterion, optimizer, scheduler


def main(args):
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True


    # solution = individual([[1, 0, 3, 1, 1, 0, 5, 1, 0, 1, 0, 3, 1, 1, 1, 1, 0, 2, 0, 1, 0, 1, 0, 0, 3, 1, 1, 1, 1, 0, 0, 1, 11,
    #                         1, 1, 0, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 0, 1, 1, 0, 0, 3, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 8, 1, 1, 0, 0, 1,
    #                         0, 1, 0, 0, 0, 0, 3, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0
    #                            , 3, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0], [1, 1, 8, 0, 0, 1, 8, 1, 1, 0, 0, 9, 0, 0, 0, 0, 1, 5,
    #                                                                             1, 1, 0, 0, 1, 0, 3]] )
    solution = individual(args.solution)

    _, model_state_dict, epoch, step, optimizer_state_dict, best_acc_top1 = utils.load(args.resume)

    train_queue, valid_queue, model, train_criterion, eval_criterion, optimizer, scheduler = \
        build_imagenet(model_state_dict, optimizer_state_dict,epoch=epoch-1, solution=solution)


    global_step = [0]
    global_step[0] = step
    # epoch = 0
    # best_acc_top1= 0
    since_time = time.time()

    while epoch<args.epochs:

        logging.info('epoch %d lr %e', epoch+1, scheduler.get_last_lr()[0]) #
        print('epoch:{}, lr:{}, '.format(epoch+1, scheduler.get_last_lr()[0]))
        #====================================================training===================================================
        train_acc, top5_avg, train_obj = train(train_queue, model, train_criterion, optimizer, args, epoch,global_step,since_time)
        logging.info('train_accuracy: %f, top5_avg: %f, loss: %f', train_acc, top5_avg, train_obj)
        print('\n       train_accuracy: {}, top5_avg: {}, loss: {}'.format(train_acc, top5_avg, train_obj))
        # ====================================================validate===================================================
        valid_acc_top1, valid_acc_top5, valid_obj = valid(valid_queue, model, eval_criterion, args)
        logging.info('valid_accuracy: %f, valid_top5_accuracy: %f', valid_acc_top1,valid_acc_top5)
        print('         valid_accuracy: {}, valid_top5_accuracy: {}'.format(valid_acc_top1,valid_acc_top5))
        # ====================================================saving===================================================
        epoch += 1
        is_best = False
        if valid_acc_top1 > best_acc_top1:
            best_acc_top1 = valid_acc_top1
            is_best = True

            utils.save(args.save, args, model, epoch, global_step[0], optimizer, best_acc_top1, is_best)


        scheduler.step()


def main_warmup(args):
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True


    # solution = individual([[1, 0, 3, 1, 1, 0, 5, 1, 0, 1, 0, 3, 1, 1, 1, 1, 0, 2, 0, 1, 0, 1, 0, 0, 3, 1, 1, 1, 1, 0, 0, 1, 11,
    #                         1, 1, 0, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 0, 1, 1, 0, 0, 3, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 8, 1, 1, 0, 0, 1,
    #                         0, 1, 0, 0, 0, 0, 3, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0
    #                            , 3, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0], [1, 1, 8, 0, 0, 1, 8, 1, 1, 0, 0, 9, 0, 0, 0, 0, 1, 5,
    #                                                                             1, 1, 0, 0, 1, 0, 3]] )

    solution = individual(args.solution)

    _, model_state_dict, epoch, step, optimizer_state_dict, best_acc_top1 = utils.load(args.resume)

    train_queue, valid_queue, model, train_criterion, eval_criterion, optimizer, scheduler = \
        build_imagenet(model_state_dict, optimizer_state_dict,epoch=epoch-1, solution=solution)


    global_step = [0]
    global_step[0] = step
    # epoch = 0
    # best_acc_top1= 0
    since_time = time.time()



    while epoch<args.epochs:
        # ====================================================scheduler setting ===================================================
        if args.lr_scheduler == 'cosine':
            current_lr = scheduler.get_last_lr()[0]
        elif args.lr_scheduler == 'linear':
            current_lr = adjust_lr(optimizer, epoch)
        else:
            print('Wrong lr type, exit')
            sys.exit(1)

        logging.info('epoch %d lr %e', epoch+1, current_lr) #
        print('epoch:{}, lr:{}, '.format(epoch+1, current_lr))
        # ====================================================warm-up===================================================
        # warm up is used for large batch size training, usually 256 or more
        # if epoch < 5 and args.batch_size > 256:
        if epoch < 5 and args.batch_size >= args.warm_up_batch:
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr * (epoch + 1) / 5.0
            print('Warming-up Epoch: %d, LR: %e', epoch, current_lr * (epoch + 1) / 5.0)
            logging.info('Warming-up Epoch: %d, LR: %e', epoch, current_lr * (epoch + 1) / 5.0)
        #------------------the following are not considered in current version due to  drop_path_prob==0  -------------
        # if torch.cuda.device_count() > 1:
        #     model.module.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        # else:
        #     model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        #====================================================training===================================================
        train_acc, top5_avg, train_obj = train(train_queue, model, train_criterion, optimizer, args, epoch,global_step,since_time)
        logging.info('train_accuracy: %f, top5_avg: %f, loss: %f', train_acc, top5_avg, train_obj)
        print('\n       train_accuracy: {}, top5_avg: {}, loss: {}'.format(train_acc, top5_avg, train_obj))

        # ====================================================validate===================================================
        valid_acc_top1, valid_acc_top5, valid_obj = valid(valid_queue, model, eval_criterion, args)
        logging.info('valid_accuracy: %f, valid_top5_accuracy: %f', valid_acc_top1,valid_acc_top5)
        print('         valid_accuracy: {}, valid_top5_accuracy: {}'.format(valid_acc_top1,valid_acc_top5))
        # ====================================================saving===================================================
        epoch += 1
        is_best = False
        if valid_acc_top1 > best_acc_top1:
            best_acc_top1 = valid_acc_top1
            is_best = True
            utils.save(args.save, args, model, epoch, global_step[0], optimizer, best_acc_top1, is_best)

        # ====================================================scheduler step ===================================================
        # since scheduler.step() should be placed  after each training, scheduler.step() will change the lr value
        # the self-designed 'adjust_lr' would not change the lr at epoch = 0
        if args.lr_scheduler == 'cosine':
            scheduler.step()




if __name__=='__main__':

    parser = argparse.ArgumentParser(description='training on imagenet')
    # ***************************  common setting******************
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('-save', type=str, default='result_imagenet')
    parser.add_argument('-resume', type=str, default=None, help='The path to the dir you want resume')
    parser.add_argument('-device', type=str, default='cuda')
    # ***************************  DDP setting **********************

    # ***************************  dataset setting******************
    parser.add_argument('-data', type=str, default="/path/imagenet")
    parser.add_argument('-classes', type=int, default=1000)
    parser.add_argument('-autoaugment', action='store_true', default=False)  # True
    parser.add_argument('-load', type=str, default='original',choices=['original','lmdb','memory'])  # True
    parser.add_argument('-num_workers', type=int, default=16)  # 16
    parser.add_argument('-data_prefetch', action='store_true', default=True,help='Accelerate DataLoader by prefetch_generator')

    # ***************************  optimization setting******************
    parser.add_argument('-batch_size', type=int, default=256)
    parser.add_argument('-eval_batch_size', type=int, default=500)

    parser.add_argument('-epochs', type=int, default=250)
    parser.add_argument('--lr', type=float, default=0.1, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
    parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
    parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
    parser.add_argument('--l2_reg', type=float, default=3e-5)
    # ***************************  warm up setting******************
    parser.add_argument('--warm_up',action='store_true', default=True,help='warm_up, where linear or cosine lr scheduler is used!!!')
    parser.add_argument('--warm_up_batch', type=int, default=256, help='batch size 256 or more is suggested, ')
    parser.add_argument('--lr_scheduler', type=str, default='linear', help='lr scheduler, linear or cosine')

    # ***************************  structure setting******************

    parser.add_argument('-channel_double', action='store_true', default=True)

    parser.add_argument('-use_aux_head', action='store_true', default=True)
    parser.add_argument('-auxiliary_weight', type=float, default=0.4)

    parser.add_argument('-keep_prob', type=float, default=1.0)
    parser.add_argument('-drop_path_keep_prob', type=float,
                        default=1.0)
    parser.add_argument('-channels', type=int, default=40)
    parser.add_argument('-layers', type=int, default=4)
    args = parser.parse_args()

    #=====================================setting=======================================
    args.resume = 'result_imagenet/train_2020-12-22-15-17-33/'
    args.data = '/home/severuspeng/AppendDisk/ImageNet/'
    args.channel_double = False
    args.channels = 76 # 24 for double_channel


    args.num_workers = 12

    args.warm_up = True #True

    args.lr = 0.1
    args.batch_size = 240
    args.warm_up_batch=100
    args.autoaugment = True

    args.solution = [[1, 1, 0, 1, 0, 0, 3, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 6, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1,
       1, 1, 0, 0, 0, 0, 0, 0, 3, 1, 0, 0, 0, 1, 0, 0, 1, 0, 8, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 10, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 3],
      [1, 0, 8, 0, 1, 0, 9, 0, 1, 0, 0, 8, 1, 1, 0, 0, 0, 7, 1, 0, 0, 0, 0, 0, 8, 1, 1, 0, 0, 0, 1, 0, 3]]



    # #------ testing arguments for warm up----------------
    # args.channel_double = True
    # args.num_workers = 0
    # args.channels=24
    # args.load = 'lmdb'
    # args.data = 'D:/数据集/imageNet/ImageNet2012/'
    # args.batch_size=30
    # args.warm_up_batch=20
    # args.warm_up=True
    #==================================== Creating Dir ========================================

    if args.resume is None:
        args.save = '{}/train_{}'.format(args.save, time.strftime("%Y-%m-%d-%H-%M-%S"))
        create__dir(args.save)
    else:

        args.save = args.resume
        print('resume from the dir: {file}'.format(file=args.resume))

    # ===================================  logging  ===================================
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(filename='{}/logs.log'.format(args.save),
                        level=logging.INFO, format=log_format, datefmt='%Y-%m-%d %I:%M:%S %p')

    if args.resume is None:
        logging.info("[Experiments Setting]\n" + "".join(
            ["[{0}]: {1}\n".format(name, value) for name, value in args.__dict__.items()]))
    else:
        logging.info('resume from the dir: {file}'.format(file=args.resume))


    print(
        'NAO ImageNet setting: args.lr = 0.4  for 4 card, batch size = 512\n                      args.lr = 0.1  for 1 card, batch size = 128\n'
        'refer to https://github.com/renqianluo/NAO_pytorch/tree/master/NAO_V2')

    print(
        '\nWarm up ImageNet setting: args.lr = 0.5  for 4 card, batch size = 1024\n                          args.lr = 0.1  for 1 card, batch size = 256\n'
        'reference: https://github.com/yuhuixu1993/PC-DARTS/blob/master/train_imagenet.py, https://github.com/chenxin061/pdarts/blob/master/train_imagenet.py')

    print('\nThe current batch size: {}, the current lr rate: {}\n'.format(args.batch_size, args.lr))

    if not args.warm_up:

        print('The basic training process (NAO training model) is used, where StepLR scheduler is used')

        # lr=0.1 for batch 128, lr=0.2 for 256, lr = k*0.1 for k*128
        if args.batch_size>128 and args.lr!=args.batch_size/128*0.1:
            args.lr = args.batch_size/128*0.1
            print('adjusting lr to {}'.format(args.lr))

        main(args)
    else:
        print('The warm_up_batch size: {}, the batch size: {}'.format(args.warm_up_batch, args.batch_size))

        if args.batch_size<args.warm_up_batch:
            print('Not reaching warm up batch size, please Increase the batch size'
                  ' or Decrease warm_up_batch size (256 is suggested) to used the warm up training\n'
                  'or Change to NAO training model'
                  ', the process will be kiiled!!!')
            sys.exit(1)

        else:
            print('The warm_up training process (Warm Up) is used, where {} scheduler is used'.format(args.lr_scheduler))

            main_warmup(args)





