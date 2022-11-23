# Basic module
from tqdm             import tqdm
from model.parse_args_test import  parse_args

# Torch and visulization
from torchvision      import transforms
from torch.utils.data import DataLoader


# Metric, loss .etc
from model.utils import *
from model.loss import *
from model.load_param_data import  load_dataset, load_param

# Model
from model.model_DNANet import  Res_CBAM_block
from model.model_DNANet import  DNANet
from model.build_model import build_model

from torchvision import utils as vutils
def save_image_tensor(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为图片
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    vutils.save_image(input_tensor, filename)

class Trainer(object):
    def __init__(self, args):

        # Initial
        self.args  = args
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
        model = model.cuda()
        # model.apply(weights_init_xavier)
        # print("Model Initializing")
        self.model      = model

        # Checkpoint
        checkpoint             = torch.load('result/' + args.model_dir)
        visulization_path      = dataset_dir + '/' +'visulization_result' + '/' + args.st_model + '_visulization_result'
        visulization_fuse_path = dataset_dir + '/' +'visulization_result' + '/' + args.st_model + '_visulization_fuse'

        make_visulization_dir(visulization_path, visulization_fuse_path)

        # Load trained model
        self.model.load_state_dict(checkpoint['state_dict'])

        # Test
        self.model.eval()
        tbar = tqdm(self.test_data)
        with torch.no_grad():
            num = 0
            for i, ( data, labels) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()
               
                rough_logits, pred = self.model(data)
                
                # img = pred.sigmoid().data.cpu().numpy().squeeze()
                # pred = torch.sigmoid(pred)
                # img = pred.data.cpu().numpy().squeeze()
                # img = res
                # img[img > 0.5]=1
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # cv2.imwrite(visulization_path + '/' + '%s' % (val_img_ids[num]) + args.suffix, 255*img)
                # pred = torch.sigmoid(pred)
                # print(pred.size())
                # pred = torch.where(pred>0.5,torch.ones_like(pred),torch.zeros_like(pred))
                # cv2.imwrite(visulization_path + '/' % (val_img_ids[num]) + args.suffix, pred)
                save_Pred_GT(pred, labels,visulization_path, val_img_ids, num, args.suffix)
                # save_image_tensor(pred,visulization_path + '/' + '%s' % (val_img_ids[num]) + args.suffix)
                num += 1

            # total_visulization_generation(dataset_dir, args.mode, test_txt, args.suffix, visulization_path, visulization_fuse_path)





def main(args):
    trainer = Trainer(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)





