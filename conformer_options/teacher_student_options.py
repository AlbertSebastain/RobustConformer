import argparse
import os
from utils import utils
from .base_conformer_options import Base_conformer_Options
import yaml

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

class Teacher_Student_Options():
    def __init__(self):
        self.teacher_network = None
        self.student_network = None
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # general configuration
        self.parser.add_argument('--works_dir', help='path to work', default='.')
        self.parser.add_argument("--feat_type",type=str,default="kaldi_magspec",help="feat_type")
        self.parser.add_argument("--delta_order", type=int, default=0, help="input delta-order")
        self.parser.add_argument('--dataroot', help='path (should have subfolders train, dev, test)')
        self.parser.add_argument('--left_context_width', type=int, default=0, help='input left_context_width-width')
        self.parser.add_argument('--right_context_width', type=int, default=0, help='input right_context_width')
        self.parser.add_argument('--normalize_type', type=int, default=1, help='normalize_type') 
        self.parser.add_argument('--num_utt_cmvn', type=int, help='the number of utterances for cmvn', default=20000)
        self.parser.add_argument('--dict_dir', default='/home/bliu/SRC/workspace/e2e/data/mix_aishell/lang_1char/', help='path to dict')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='', help='name of the experiment.')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints1', help='models are saved here')  
        self.parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to latest checkpoint (default: none)')     
        self.parser.add_argument('--teacher_resume', default=None, type=str, metavar='PATH', help='path to teahcer model')
        self.parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in data-loading')
        self.parser.add_argument('--train_folder',default='train',type=str,help='name of train folder')
        self.parser.add_argument('--dev_folder',default='dev',type=str,help="name of dev folder")
        self.parser.add_argument('--exp_path',type=str,default = None,help = 'exp_dir')
        # use yaml config
        self.parser.add_argument('--config_file',default=None,type=str,action = 'append',help="use yaml file to set arguments")
        self.parser.add_argument('--student_model_config',default=None,type=str,help="configuration file for student .yam file")
        self.parser.add_argument("--transformer-lr",default=10.0,type=float,help="Initial value of learning rate",)
        self.parser.add_argument("--transformer-warmup-steps",default=25000,type=int,help="optimizer warmup steps",)

        # augmentation parameters
        # adversarial sample regularization
        # FGSM
        self.parser.add_argument("--FGSM_augmentation",default=False,type=str2bool,help="Whether to use FGSM augmentation.")
        self.parser.add_argument("--epsilon_FGSM",default=0.15,type=float, help="FGSM weight")
        self.parser.add_argument("--alpha_FGSM",default=0.3,type=float,help="regularization term weight")
        # VAT
        self.parser.add_argument("--use_vat",default=False,type= str2bool,help="Whether to use VAT augmentation")
        self.parser.add_argument("--vat_epsilon",default=5.0,type=float, help="updating step")
        self.parser.add_argument("--vat_delta_weight",default=0.3,type=float,help="VAT weight")
        self.parser.add_argument("--vat_weight",default=1.0,type=float, help="VAT regularization term weight")
        self.parser.add_argument("--vat_iter",default=1,type=int, help="Number of iterations")
        # general
        self.parser.add_argument("--p_aug",default=1,type=float, help="probability of using adversarial regularization")
        self.parser.add_argument("--start_augmentation",default=9,type=int,help="augmentation start epoch")
        # SpecAugment
        self.parser.add_argument("--use_spec_aug",default=False,type=str2bool, help="Whether to use spec augmentation")
        self.parser.add_argument("--SpecF",default=30,type=int, help="Number of masked frequency")
        self.parser.add_argument("--SpecT", default=40,type=int, help="Number of potential masked time frames")
        # shift operation
        self.parser.add_argument("--use_shift",default=False,type=str2bool,help="Whether to use shift operation")
        self.parser.add_argument("--Shift_frames",default=20,type=int, help="Number of removed time frames")
        # random_mix_operation
        self.parser.add_argument("--use_random_mix_noise",default=False,type= str2bool,help="Whether to use random-mix operation")

        # fbank model
        self.parser.add_argument('--fbank_dim', type=int, default=80, help='num_features of a frame')
        self.parser.add_argument('--fbank-opti-type', type=str, default='frozen', choices=['frozen', 'train'], help='fbank-opti-type')

        # rnnlm related         
        self.parser.add_argument('--model-unit', type=str, default='char', choices=['char', 'word', 'syllable'], help='model_unit')
        self.parser.add_argument('--space-loss-weight', default=0.1, type=float, help='space_loss_weight.')
        self.parser.add_argument('--lmtype', type=str, default=None, help='RNNLM model file to read')
        self.parser.add_argument('--rnnlm', type=str, default=None, help='RNNLM model file to read')
        self.parser.add_argument('--lm-weight', default=0.1, type=float, help='RNNLM weight.')

        # minibatch related
        self.parser.add_argument('--batch-size', '-b', default=30, type=int, help='Batch size')
        self.parser.add_argument('--maxlen-in', default=800, type=int, metavar='ML', help='Batch size is reduced if the input sequence length > ML')
        self.parser.add_argument('--maxlen-out', default=150, type=int, metavar='ML', help='Batch size is reduced if the output sequence length > ML')        
        self.parser.add_argument('--verbose', default=1, type=int, help='Verbose option')
        self.parser.add_argument('--use_delta', default=False, type=str2bool, help="use delta delta_delta features")

        # ts leanring related
        self.parser.add_argument('--tau',default=1.0,type=float,help="Temperature parameter for CTC guide")
        self.parser.add_argument('--skd_weight',default=1.0,type=float, help="Weight parameter for CTC guide")
        self.parser.add_argument('--attention_temperature',default=2.0,type=float,help="Temperature parameter for attention")
        self.parser.add_argument('--top_k',default=20, type=int, help="Number of perserved logits")
        self.parser.add_argument('--attention_logit_weight',default=0.3,type=float,help="Weight parameter for attention guide")

        # condition network
        self.parser.add_argument('--use_condition_layer',default=False,type=str2bool, help="Whether to use the condition layer")
        self.parser.add_argument('--condition_layers',default=2,type=int, help="Number of layers in condition network")
        self.parser.add_argument('--no_condition_epoch',default=12,type=int, help="the epochs of activation of the condition network")
        self.parser.add_argument('--start_condition_epoch',default=32,type=int, help="the epochs starts condition network")
        self.parser.add_argument('--condition_hidden_dim',default=256,type=int, help="the hidden dimension of the condition network")
        self.parser.add_argument('--condition_weight',default=1.0,type=float,help="Weight parameter of condition network")
        # train configuration
        self.parser.add_argument('--opt_type', default='adadelta', type=str, choices=['adadelta', 'adam'], help='Optimizer')
        self.parser.add_argument('--lr', type=float, default=0.005, help='learning rate, default=0.0002')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
        self.parser.add_argument('--eps', default=1e-8, type=float, help='Epsilon constant for optimizer')
        self.parser.add_argument('--eps-decay', default=0.01, type=float, help='Decaying ratio of epsilon')
        self.parser.add_argument('--criterion', default='acc', type=str, choices=['loss', 'acc'], help='Criterion to perform epsilon decay')
        self.parser.add_argument('--threshold', default=1e-4, type=float, help='Threshold to stop iteration')
        self.parser.add_argument('--start_epoch', default=0, type=int, help='manual iters number (useful on restarts)') 
        self.parser.add_argument('--iters', default=0, type=int, help='manual iters number (useful on restarts)')   
        self.parser.add_argument('--epochs', '-e', default=30, type=int, help='Number of maximum epochs')
        self.parser.add_argument('--shuffle_epoch', default=-1, type=int, help='Number of shuffle epochs')        
        self.parser.add_argument('--grad-clip', default=5, type=float, help='Gradient norm threshold to clip')
        self.parser.add_argument('--num-save-attention', default=3, type=int, help='Number of samples of attention to be saved')   
        self.parser.add_argument('--num-saved-specgram', default=3, type=int, help='Number of samples of specgram to be saved')               
        # debug related   
        self.parser.add_argument('--validate_freq', type=int, default=8000, help='how many batches to validate the trained model')   
        self.parser.add_argument('--print_freq', type=int, default=500, help='how many batches to print the trained model')         
        self.parser.add_argument('--best_acc', default=0, type=float, help='best_acc')
        self.parser.add_argument('--best_loss', default=float('inf'), type=float, help='best_loss')


        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.student_network = Base_conformer_Options().parse(['--config_file',self.opt.student_model_config],only_set=True)
        self.teacher_network = Base_conformer_Options().parse(['--resume', None],only_set=True)
        if self.opt.config_file != None:
            for config_file in  self.opt.config_file:
                with open(config_file,encoding = "utf-8") as f:
                    data_file = f.read()
                data = yaml.full_load(data_file)
                for key_d,val_d in data.items():
                    key_d = key_d.replace("--","")
                    key_d = key_d.replace("-","_")
                    setattr(self.opt,key_d,val_d)   
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # if self.opt.mtlalpha == 1.0:
        #     self.opt.mtl_mode = 'ctc'
        # elif self.opt.mtlalpha == 0.0:
        #     self.opt.mtl_mode = 'att'
        # else:
        #     self.opt.mtl_mode = 'mtl'
        setattr(self.opt,'student_network',self.student_network)
        setattr(self.opt,'teacher_network',self.teacher_network)
        self.opt.student_network.use_delta = self.opt.use_delta
        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        #exp_path = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        utils.mkdirs(self.opt.exp_path)
        #self.opt.exp_path = self.opt.exp_path
        if self.opt.name != '':
            file_name = os.path.join(self.opt.checkpoints_dir,self.opt.name, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt