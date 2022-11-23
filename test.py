# Basic module
from tqdm             import tqdm
from model.parse_args_test import  parse_args
import scipy.io as scio

# Torch and visulization
import torchvision
from torchvision      import transforms
from torch.utils.data import DataLoader

# Metric, loss .etc
from model.utils import *
from model.metric import *
from model.loss import *
from model.load_param_data import  load_dataset, load_param

# Model
from model.build_model import build_model
from model.model_DNANet import  Res_CBAM_block
from model.model_DNANet import  DNANet

class Trainer(object):
    def __init__(self, args):

        # Initial
        self.args  = args
        self.ROC   = ROCMetric(1, args.ROC_thr)
        self.PD_FA = PD_FA(1,args.ROC_thr)
        self.mIoU  = mIoU(1)
        self.save_prefix = '_'.join([args.model, args.dataset])
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)

        # Read image index from TXT
        if args.mode    == 'TXT':
            dataset_dir = args.root + '/' + args.dataset
            train_img_ids, val_img_ids, test_txt=load_dataset(args.root, args.dataset,args.split_method)

        # Preprocess and load data
        input_transform = transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        testset         = TestSetLoader (dataset_dir,img_id=val_img_ids,base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix)
        self.test_data  = DataLoader(dataset=testset,  batch_size=args.test_batch_size, num_workers=args.workers,drop_last=False)

        # Choose and load model (this paper is finished by one GPU)
        # if args.model   == 'DNANet':
        #     model       = DNANet(num_classes=1,input_channels=args.in_channels, block=Res_CBAM_block, num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=args.deep_supervision)
        model = build_model(is_train=True)
        model           = model.cuda()
        # model.apply(weights_init_xavier)
        # print("Model Initializing")
        self.model      = model

        # Initialize evaluation metrics
        self.best_recall    = [0,0,0,0,0,0,0,0,0,0,0]
        self.best_precision = [0,0,0,0,0,0,0,0,0,0,0]

        # Load trained model
        checkpoint        = torch.load('result/' + 'NUAA-SIRST_MyModel_02_07_2022_09_43_08_wDS/mIoU__MyModel_NUAA-SIRST_epoch.pth.tar')
        self.model.load_state_dict(checkpoint['state_dict'])

        # Test
        self.model.eval()
        total = sum([param.nelement() for param in self.model.parameters()])
        print("Number of parameter: %.2fM" % (total/1e6))
        # print(self.model)
        tbar = tqdm(self.test_data)
        losses = AverageMeter()
        with torch.no_grad():
            num = 0
            for i, ( data, labels) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()
                  
                rough_logits, pred = self.model(data)
                # print(torch.sqrt(rough_logits.max()))
                
                Iou_loss = SoftIoULoss(pred, labels)
                
                # Iou_loss = self.DiceLoss(pred, labels)
                Bce_loss = BCELoss(rough_logits,labels)
                loss = 0.9*Iou_loss + 0.1*Bce_loss
                num += 1
                 
                pred = torch.sigmoid(pred)
                pred = torch.where(pred>0.5,torch.ones_like(pred),torch.zeros_like(pred))
                # pred = pred.sigmoid()
                # pred = (pred >0.5).float()
                losses.    update(loss.item(), pred.size(0))
                self.ROC.  update(pred, labels)
                self.mIoU. update(pred, labels)
                self.PD_FA.update(pred, labels)

                ture_positive_rate, false_positive_rate, recall, precision= self.ROC.get()
                _, mean_IOU = self.mIoU.get()
            FA, PD = self.PD_FA.get(len(val_img_ids))
            # print(PD)
            print('FA %.7f, PD %.7f' % (FA[0]*1000000,PD[0]*100))
            scio.savemat(dataset_dir + '/' +  'value_result'+ '/' +args.st_model  + '_PD_FA_' + str(255),
                         {'number_record1': FA, 'number_record2': PD})

            save_result_for_test(dataset_dir, args.st_model,args.epochs, mean_IOU, recall, precision)

def main(args):
    trainer = Trainer(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)





