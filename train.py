# torch and visulization
from tqdm             import tqdm
import torch.nn as nn
# import torch.optim    as optim
from torch.optim      import lr_scheduler
from torchvision      import transforms
from torch.utils.data import DataLoader
from model.parse_args_train import  parse_args

# metric, loss .etc
from model.utils import *
from model.metric import *
from model.loss import *
from model.load_param_data import  load_dataset, load_param

# model
from model.model_DNANet import  Res_CBAM_block
from model.model_DNANet import  DNANet

from model.build_model import build_model
# import segmentation_models_pytorch as smp
# import torch_optimizer as optim

class Trainer(object):
    def __init__(self, args):
        # Initial
        self.args = args
        self.ROC  = ROCMetric(1, 10)
        self.mIoU = mIoU(1)
        self.PD_FA = PD_FA(1,10)
        self.save_prefix = '_'.join([args.model, args.dataset])
        self.save_dir    = args.save_dir
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)
        
        # Read image index from TXT
        if args.mode == 'TXT':
            dataset_dir = args.root + '/' + args.dataset
            self.train_img_ids, self.val_img_ids, test_txt = load_dataset(args.root, args.dataset, args.split_method)
        
        # Preprocess and load data
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

       
        trainset        = TrainSetLoader(dataset_dir,img_id=self.train_img_ids,base_size=args.base_size,crop_size=args.crop_size,transform=input_transform,suffix=args.suffix)
        testset         = TestSetLoader (dataset_dir,img_id=self.val_img_ids,base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix)
        self.train_data = DataLoader(dataset=trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers,drop_last=True)
        self.test_data  = DataLoader(dataset=testset,  batch_size=args.test_batch_size, num_workers=args.workers,drop_last=False)

        # Choose and load model (this paper is finished by one GPU)
       
        model = build_model(is_train=True)
        # model.load_state_dict(torch.load("./result/NUAA-SIRST_MyModel_02_07_2022_09_43_08_wDS/mIoU__MyModel_NUAA-SIRST_epoch.pth.tar")['state_dict'])
      
       
        model  = model.cuda()
        # model.apply(weights_init_xavier)
        # print("Model Initializing")
        self.model      = model
        # Optimizer and lr scheduling
        if args.optimizer   == 'AdamW':
            # self.optimizer = optim.Lamb(model.parameters(), lr=args.lr)
            self.optimizer  = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
            # self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9)
        elif args.optimizer == 'Adagrad':
            self.optimizer  = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        if args.scheduler   == 'CosineAnnealingLR':
            self.scheduler  = lr_scheduler.CosineAnnealingLR( self.optimizer, T_max=args.T_max, eta_min=args.min_lr)
            # self.scheduler  = lr_scheduler.ReduceLROnPlateau( self.optimizer, factor=0.9,patience=100)
        
        self.DiceLoss = DiceLoss()
        # Evaluation metrics
        self.best_iou  = 0
        self.best_recall    = [0,0,0,0,0,0,0,0,0,0,0]
        self.best_precision = [0,0,0,0,0,0,0,0,0,0,0]
   
    # Training
    def training(self,epoch):

        tbar = tqdm(self.train_data)
        self.model.train()
        losses = AverageMeter()
        for i, ( data, labels) in enumerate(tbar):
            data   = data.cuda()
            labels = labels.cuda()
           
            rough_logits, pred = self.model(data)
            
            Iou_loss = SoftIoULoss(pred, labels)
            # Iou_loss = self.DiceLoss(pred, labels)
            Bce_loss = BCELoss(rough_logits,labels)
            loss = 0.9*Iou_loss + 0.1*Bce_loss
       
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.update(loss.item(), pred.size(0))
            # tbar.set_description('Epoch %d, training loss %.4f' % (epoch, losses.avg))
            tbar.set_description('Epoch %d, training loss %.4f, Iou_loss %.4f, Bce_loss %.4f, lr %.7f' % (epoch, losses.avg,Iou_loss,Bce_loss, self.optimizer.param_groups[0]['lr']))
        self.train_loss = losses.avg
        

    # Testing
    def testing (self, epoch):
        tbar = tqdm(self.test_data)
        self.model.eval()
        self.mIoU.reset()
        # self.PD_FA.reset()
        losses = AverageMeter()

        with torch.no_grad():
            for i, ( data, labels) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()
  
                rough_logits, pred = self.model(data)
                
                Iou_loss = SoftIoULoss(pred, labels)
                # Iou_loss = self.DiceLoss(pred, labels)
                Bce_loss = BCELoss(rough_logits,labels)
                loss = 0.9*Iou_loss + 0.1*Bce_loss
                # pred = torch.sigmoid(pred)
                # pred = torch.where(pred>0.5,torch.ones_like(pred),torch.zeros_like(pred))
                # pred = pred.sigmoid()
                # pred = (pred >0.5).float()
                losses.update(loss.item(), pred.size(0))
                self.ROC .update(pred, labels)
                self.mIoU.update(pred, labels)
                # self.PD_FA.update(pred, labels)
                ture_positive_rate, false_positive_rate, recall, precision = self.ROC.get()
                _, mean_IOU = self.mIoU.get()
                tbar.set_description('Epoch %d, test loss %.4f, mean_IoU: %.4f' % (epoch, losses.avg, mean_IOU))
            test_loss=losses.avg
        # FA, PD = self.PD_FA.get(len(self.val_img_ids))
        # print('FA %.7f, PD %.7f' % (FA[0]*1000000,PD[0]*100))
        self.scheduler.step()
        if mean_IOU >= 0.74:
            torch.save(self.model.state_dict(), "./result/mIoU{}model_parameter.pth.tar".format(int(mean_IOU*1000)))
        # save high-performance model
        best_iou = save_model(mean_IOU, self.best_iou, self.save_dir, self.save_prefix,
                   self.train_loss, test_loss, recall, precision, epoch, self.model.state_dict())
        self.best_iou = best_iou

def init_seeds(seed=0):
    torch.manual_seed(seed) # sets the seed for generating random numbers.
    torch.cuda.manual_seed(seed) # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed_all(seed) # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.

    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main(args):
    init_seeds(1024)
    trainer = Trainer(args)
    for epoch in range(args.start_epoch, args.epochs):
        trainer.training(epoch)
        trainer.testing(epoch)


if __name__ == "__main__":
    args = parse_args()
    main(args)





