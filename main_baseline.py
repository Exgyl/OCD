import random,os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from utils import Parser as parser
from utils.get_datasets import get_datasets, get_class_splits
from config import dino_pretrain_path
import vision_transformer as vits
from data.argumentations import get_transform
from utils.general_utils import get_mean_lr
from utils.cluster_and_log_utils import log_accs_from_preds
from torch.utils.data import DataLoader
from torch.optim import SGD, lr_scheduler
from tqdm import tqdm
from utils.general_utils import AverageMeter
from Contrastive import *
def seed_torch(seed):
    random.seed(seed) #设置random模块的种子
    os.environ['PYTHONHASHSEED'] = str(seed) #控制python哈希函数的种子值，使得进行与哈希相关的操作时结果可复现
    np.random.seed(seed)#设置了numpy随机数种子
    torch.manual_seed(seed) #cpu seed
    torch.cuda.manual_seed(seed) #gup seed
    torch.cuda.manual_seed_all(seed) #mutil gpu seed
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train(projection_head, model, train_loader, test_loader, unlabelled_train_loader, args):
    optimizer = SGD(list(projection_head.parameters()) + list(model.parameters()), lr=args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 1e-3
    )

    sup_con_crit = SupConLoss()
    best_test_acc_lab = 0

    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        train_acc_record = AverageMeter()

        projection_head.train()
        model.train()

        for batch_id, batch in enumerate(tqdm(train_loader)):
            images, class_labels, uq_idsx = batch

            class_labels = class_labels.to(device)
            images = torch.cat(images, dim=0).to(device)
            features = model(images)
            features, _, _ = projection_head(features)
            features = torch.nn.functional.normalize(features, dim=-1)

            f1, f2 = [f for f in features.chunk(2)]

            sup_con_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            sup_con_labels = class_labels
            sup_con_loss = sup_con_crit(sup_con_feats, labels=sup_con_labels)

            loss = sup_con_loss
            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Train Epoch: {} Avg Loss: {:.4f} | Seen Class Acc: {:.4f} '.format(epoch, loss_record.avg,
                                                                                  train_acc_record.avg))

        with torch.no_grad():
            all_acc, old_acc, new_acc = test_on_the_fly(model, projection_head, unlabelled_train_loader,
                                                        epoch=epoch, save_name='Train ACC Unlabelled',
                                                        args=args)
        #保存数据
        args.writer.add_scalar('Loss', loss_record.avg, epoch)
        args.writer.add_scalar('Train Acc Labelled Data', train_acc_record.avg, epoch)
        args.writer.add_scalar('LR', get_mean_lr(optimizer), epoch)
        print('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc,
                                                                              new_acc))
        exp_lr_scheduler.step()
        torch.save(model.state_dict(), args.model_path)
        print("model saved to {}.".format(args.model_path))

        torch.save(projection_head.state_dict(), args.model_path[:-3] + '_proj_head.pt')
        print("projection head saved to {}.".format(args.model_path[:-3] + '_proj_head.pt'))

        if old_acc > best_test_acc_lab:

            print(f'Best ACC on old Classes on disjoint test set: {old_acc:.4f}...')
            print('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc,
                                                                                  new_acc))
            torch.save(model.state_dict(), args.model_path[:-3] + f'_best.pt')
            print("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))
            torch.save(projection_head.state_dict(), args.model_path[:-3] + f'_proj_head_best.pt')
            print("projection head saved to {}.".format(args.model_path[:-3] + f'_proj_head_best.pt'))
            best_test_acc_lab = old_acc



def test_on_the_fly(model, projection_head, test_loader, epoch, save_name, args):

    model.eval()
    projection_head.eval()

    all_feats = []
    targets = np.array([])
    mask = np.array([])

    print('Collating features...')
    #提取所有特征
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda()

        feats = model(images)
        _, feats, _ = projection_head(feats)

        feats = torch.nn.functional.normalize(feats, dim=-1)[:, :]
        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))

    print('-' * 20 + 'On-The-Fly' + '-' * 20)
    all_feats = np.concatenate(all_feats)
    feats_hash = torch.Tensor(all_feats > 0).float().tolist()
    preds = []
    hash_dict = []
    for feat in feats_hash:
        if not feat in hash_dict:
            hash_dict.append(feat)
        preds.append(hash_dict.index(feat))
    preds = np.array(preds)
    print(len(list(set(preds))), len(preds))
    print('-' * 20 + 'EVALUATE' + '-' * 20)

    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                        T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                        writer=args.writer)

    return all_acc, old_acc, new_acc

if __name__ == '__main__':

    for run in range(0,10):
        print('-' * 20 + 'INIT' + '-' * 20)
        seed_torch(run)
        args = parser.parse_args()
        device = torch.device('cuda:0')
        args = get_class_splits(args)
        args.writer = SummaryWriter(log_dir='./logs')
        args.num_labeled_classes = len(args.train_classes)
        args.num_unlabeled_classes = len(args.unlabeled_classes)

        print(f'Labeled_classes {args.num_labeled_classes}, Unlabeled_classes {args.num_unlabeled_classes}')
        print(f'Using evaluation function {args.eval_funcs[0]} to print results')

        print('-' * 20 + 'BASE MODEL' + '-' * 20)
        if args.base_model == 'vit_dino':
            args.interpolation = 3
            args.crop_pct = 0.875
            pretrain_path = dino_pretrain_path

            model = vits.__dict__['vit_base']()

            if args.warmup_model_dir is not None:
                print(f'Loading weights from {args.warmup_model_dir}')
                model.load_stat_dict(args.warmup_model_dir, map_location='cpu')
            else:
                state_dict = torch.load(pretrain_path, map_location='cpu')  # 加载模型的预训练权重
                model.load_state_dict(state_dict)  # 将权重加到模型参数中

            model.to(device)

            #硬编码整个模型，因为没有对Vit进行微调
            args.image_size = 224
            args.feat_dim = 768
            args.num_mlp_layers = 3
            args.code_dim = 12
            args.mlp_out_dim = None


            for m in model.parameters(): #遍历所有参数，将所有参数的梯度设置为不可训练
                m.requires_grad = False
            for name, m in model.named_parameters(): #只训练block_num（一个数字）之后的参数
                if 'block' in name:
                    block_num = int(name.split('.')[1])
                    if block_num >= args.grad_from_block:
                        m.requires_grad = True

        else:
            raise NotImplementedError

        print('-' * 20 + 'CONTRASTIVE TRANSFORM' + '-' * 20)
        train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
        train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)

        print('-' * 20 + 'DataSets' + '-' * 20)
        train_dataset, test_dataset, unlabelled_train_examples_test, datasets, labelled_dataset = get_datasets(args.dataset_name,
                                                                                                               train_transform,
                                                                                                               test_transform,
                                                                                                               args)
        print('-' * 20 + 'DATALOADERS' + '-' * 20)
        labelled_train_loader = DataLoader(labelled_dataset, num_workers=args.num_workers, batch_size=args.batch_size,
                                           shuffle=True, drop_last=True)
        unlabelled_train_loader = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                             batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, num_workers=args.num_workers,
                                 batch_size=args.batch_size,shuffle=False)

        print('-' * 20 + 'PROJECTION HEAD' + '-' * 20)
        projection_head = vits.__dict__['BASEHead'](in_dim=args.feat_dim, out_dim=args.mlp_out_dim,
                                                    nlayers=args.num_mlp_layers, code_dim=args.code_dim,
                                                    class_num=args.num_labeled_classes)
        projection_head.to(device)

        print('-' * 20 + 'TRAIN' + '-' * 20)
        train(projection_head, model, labelled_train_loader, test_loader, unlabelled_train_loader, args)


