import argparse
import os
from utils import utils
import yaml
import torch

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

class Base_conformer_Options():
    def __init__(self):
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
        self.parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in data-loading')
        self.parser.add_argument('--train_folder',default='train',type=str,help='name of train folder')
        self.parser.add_argument('--dev_folder',default='dev',type=str,help="name of dev folder")
        self.parser.add_argument('--exp_path',type=str,default = None,help = 'exp_dir')
        self.parser.add_argument("--mtlalpha",type = float, default = 0.0)
        # use yaml config
        self.parser.add_argument('--config_file',default=None,type=str,action = 'append',help="use yaml file to set arguments")
        # input features
        self.parser.add_argument("--transformer-init",type=str,default="pytorch",
            choices=[
                "pytorch",
                "xavier_uniform",
                "xavier_normal", 
                "kaiming_uniform",
                "kaiming_normal",
                ]
                ,help="how to initialize transformer parameters",)
        self.parser.add_argument("--transformer-input-layer",type=str,default="conv2d",
            choices=[
                "conv2d",
                "linear", 
                "embed"
                ]
                ,help="transformer input layer type",)
        self.parser.add_argument("--transformer-attn-dropout-rate",default=None,type=float,help="dropout in transformer attention. use --dropout-rate if None is set",)
        self.parser.add_argument("--transformer-lr",default=10.0,type=float,help="Initial value of learning rate",)
        self.parser.add_argument("--transformer-warmup-steps",default=25000,type=int,help="optimizer warmup steps",)
        self.parser.add_argument("--transformer-length-normalized-loss",default=True,type=str2bool,help="normalize loss by length",)
        self.parser.add_argument("--transformer-encoder-selfattn-layer-type",type=str,default="selfattn",
            choices=[
                "selfattn",
                "rel_selfattn",
                "lightconv",
                "lightconv2d",
                "dynamicconv",
                "dynamicconv2d",
                "light-dynamicconv2d",
                ],
                help="transformer encoder self-attention layer type",)
        self.parser.add_argument("--transformer-decoder-selfattn-layer-type",type=str,default="selfattn",
            choices=[
                "selfattn",
                "lightconv",
                "lightconv2d",
                "dynamicconv",
                "dynamicconv2d",
                "light-dynamicconv2d",
                ],
                help="transformer decoder self-attention layer type",)
        self.parser.add_argument("--wshare",default=4,type=int,help="Number of parameter shargin for lightweight convolution",)
        self.parser.add_argument("--ldconv-encoder-kernel-length",default="21_23_25_27_29_31_33_35_37_39_41_43",type=str,
            help="kernel size for lightweight/dynamic convolution: "
            'Encoder side. For example, "21_23_25" means kernel length 21 for '
            "First layer, 23 for Second layer and so on.",)
        self.parser.add_argument("--ldconv-decoder-kernel-length",default="11_13_15_17_19_21",type=str,
            help="kernel size for lightweight/dynamic convolution: "
            'Decoder side. For example, "21_23_25" means kernel length 21 for '
            "First layer, 23 for Second layer and so on.",)
        self.parser.add_argument("--ldconv-usebias",type=str2bool,default=False,help="use bias term in lightweight/dynamic convolution",)
        self.parser.add_argument("--dropout-rate",default=0.0,type=float,help="Dropout rate for the encoder",)
        self.parser.add_argument("--decoder_mode",default = None)
        self.parser.add_argument("--ctc_type",default = "warpctc",type=str)
        self.parser.add_argument("--report_cer",default = False, type = str2bool)
        self.parser.add_argument("--report_wer",default = False, type = str2bool)
        self.parser.add_argument("--elayers",default=4,type=int,
            help="Number of encoder layers (for shared recognition part "
            "in multi-speaker asr mode)",)
        self.parser.add_argument("--eunits","-u",default=300,type=int,help="Number of encoder hidden units",)
        # Attention
        self.parser.add_argument("--adim",default=320,type=int,help="Number of attention transformation dimensions",)
        self.parser.add_argument("--aheads",default=4,type=int,help="Number of heads for multi head attention",)
        # Decoder
        self.parser.add_argument("--dlayers", default=1, type=int, help="Number of decoder layers")
        self.parser.add_argument("--dunits", default=320, type=int, help="Number of decoder hidden units")
        self.parser.add_argument("--transformer-encoder-pos-enc-layer-type",type=str,default="abs_pos",choices=["abs_pos", "scaled_abs_pos", "rel_pos"],help="transformer encoder positional encoding layer type",)
        self.parser.add_argument("--transformer-encoder-activation-type",type=str,default="swish",choices=["relu", "hardtanh", "selu", "swish"],help="transformer encoder activation function type",)
        self.parser.add_argument("--macaron-style",default=False,type=str2bool,help="Whether to use macaron style for positionwise layer",)
        # CNN module
        self.parser.add_argument("--use-cnn-module",default=False,type=str2bool,help="Use convolution module or not",)
        self.parser.add_argument("--cnn-module-kernel",default=31,type=int,help="Kernel size of convolution module.",)

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
        
        # model (parameter) related
        self.parser.add_argument('--sche-samp-rate', default=0.0, type=float, help='scheduled sampling rate')
        self.parser.add_argument('--sche-samp-final-rate', default=0.6, type=float, help='scheduled sampling final rate')
        self.parser.add_argument('--sche-samp-start-epoch', default=5, type=int, help='scheduled sampling start epoch')
        self.parser.add_argument('--sche-samp-final_epoch', default=15, type=int, help='scheduled sampling start epoch')
        
        # rnnlm related         
        self.parser.add_argument('--model-unit', type=str, default='char', choices=['char', 'word', 'syllable'], help='model_unit')
        self.parser.add_argument('--space-loss-weight', default=0.1, type=float, help='space_loss_weight.')
        self.parser.add_argument('--lmtype', type=str, default=None, help='RNNLM model file to read')
        self.parser.add_argument('--rnnlm', type=str, default=None, help='RNNLM model file to read')
        #self.parser.add_argument('--kenlm', type=str, default=None, help='KENLM model file to read')
        #self.parser.add_argument('--word-rnnlm', type=str, default=None, help='Word RNNLM model file to read')
        #self.parser.add_argument('--word-dict', type=str, default=None, help='Word list to read')
        self.parser.add_argument('--lm-weight', default=0.1, type=float, help='RNNLM weight.')
        
        # FSLSTMLM training configuration
        # self.parser.add_argument('--fast_cell_size', type=int, default=400, help='fast_cell_size')
        # self.parser.add_argument('--slow_cell_size', type=int, default=400, help='slow_cell_size')
        # self.parser.add_argument('--fast_layers', type=int, default=2, help='fast_layers')
        # self.parser.add_argument('--zoneout_keep_c', type=float, default=0.5, help='zoneout_c')
        # self.parser.add_argument('--zoneout_keep_h', type=float, default=0.9, help='zoneout_h')
    
        # minibatch related
        self.parser.add_argument('--batch-size', '-b', default=30, type=int, help='Batch size')
        self.parser.add_argument('--maxlen-in', default=800, type=int, metavar='ML', help='Batch size is reduced if the input sequence length > ML')
        self.parser.add_argument('--maxlen-out', default=150, type=int, metavar='ML', help='Batch size is reduced if the output sequence length > ML')        
        self.parser.add_argument('--verbose', default=1, type=int, help='Verbose option')
        self.parser.add_argument('--use_delta', default=False, type=str2bool, help="use delta delta_delta features")
        self.initialized = True

    def parse(self, pass_args=None, only_set=False):
        if not self.initialized:
            self.initialize()
        if pass_args is None:
            self.opt = self.parser.parse_args()
        else:
            self.opt = self.parser.parse_args(pass_args)
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

        if self.opt.mtlalpha == 1.0:
            self.opt.mtl_mode = 'ctc'
        elif self.opt.mtlalpha == 0.0:
            self.opt.mtl_mode = 'att'
        else:
            self.opt.mtl_mode = 'mtl'

        if only_set:
            return self.opt
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

