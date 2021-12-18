import argparse
import os
from utils import utils
import torch


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # general configuration
        self.parser.add_argument("--works_dir", help="path to work", default=".")
        self.parser.add_argument("--dataroot", default="data", help="path (should have subfolders train, dev, test)")
        self.parser.add_argument("--dict_dir", default="data/lang_1char", help="path to dict")
        self.parser.add_argument("--dict_file", default="data/lang_1char/train_units.txt", help="path to dict file")
        self.parser.add_argument("--gpu_ids", type=str, default="0", help="gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU")
        self.parser.add_argument("--name", type=str, default="testing", help="name of the experiment.")
        self.parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="models are saved here")
        self.parser.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")
        self.parser.add_argument("--enhance_resume", default="", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")
        self.parser.add_argument("--asr_resume", default="", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")
        self.parser.add_argument("--joint_resume", default="", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")
        self.parser.add_argument("--num_workers", default=4, type=int, help="Number of workers used in data-loading")

        # input features
        self.parser.add_argument("--feat_type", type=str, default="kaldi_magspec", help="feat_type")
        self.parser.add_argument("--left_context_width", type=int, default=0, help="input left_context_width-width")
        self.parser.add_argument("--right_context_width", type=int, default=0, help="input right_context_width")
        self.parser.add_argument("--delta_order", type=int, default=0, help="input delta-order")
        self.parser.add_argument("--normalize_type", type=int, default=1, help="normalize_type")
        self.parser.add_argument("--ob_compute_cmvn", type=int, default=1, help="Do you want to compute a cmvn?")

        self.parser.add_argument("--num_utt_cmvn", type=int, help="the number of utterances for cmvn", default=20000)
        self.parser.add_argument("--num_utt_per_loading", type=int, help="the number of utterances one loading", default=200)
        self.parser.add_argument("--mix_noise", dest="mix_noise", action="store_true", help="mix_noise")
        self.parser.add_argument("--lowSNR", type=float, default=5, help="lowSNR")
        self.parser.add_argument("--upSNR", type=float, default=30, help="upSNR")

        # transformer
        # general
        self.parser.add_argument("--", default="")
        self.parser.add_argument("--", default="")
        self.parser.add_argument("--", default="")
        self.parser.add_argument("--", default="")
        self.parser.add_argument("--", default="")
        self.parser.add_argument("--", default="")
        self.parser.add_argument("--", default="")
        self.parser.add_argument("--", default="")
        self.parser.add_argument("--", default="")
        self.parser.add_argument("--", default="")
        self.parser.add_argument("--", default="")

        # encoder
        self.parser.add_argument("--dropout_rate", default=0.0)
        self.parser.add_argument("--transformer_attn_dropout_rate", default=None)
        self.parser.add_argument("--", default="")
        self.parser.add_argument("--", default="")
        self.parser.add_argument("--", default="")
        self.parser.add_argument("--transformer_encoder_selfattn_layer_type", default="selfattn")
        self.parser.add_argument("--adim", default=256)
        self.parser.add_argument("--aheads", default=4)
        self.parser.add_argument("--wshare", default=4)
        self.parser.add_argument("--ldconv_encoder_kernel_length", default=11)
        self.parser.add_argument("--ldconv_usebias", default=False)
        self.parser.add_argument("--eunits", default=2048)
        self.parser.add_argument("--elayers", default=6)
        self.parser.add_argument("--transformer_input_layer", default="embed")
        self.parser.add_argument("--dropout_rate", default=0.1)
        self.parser.add_argument("--transformer_attn_dropout_rate", default=0.0)
        self.parser.add_argument("--", default="")
        self.parser.add_argument("--", default="")
        self.parser.add_argument("--", default="")
        self.parser.add_argument("--", default="")

        # decoder
        self.parser.add_argument("--transformer_decoder_selfattn_layer_type", default="")
        # self.parser.add_argument("--adim", default="")
        # self.parser.add_argument("--aheads", default="")
        # self.parser.add_argument("--wshare", default="")
        self.parser.add_argument("--ldconv_decoder_kernel_length", default="")
        # self.parser.add_argument("--ldconv_usebias", default="")
        self.parser.add_argument("--dunits", default=2048)
        self.parser.add_argument("--dlayers", default="")
        self.parser.add_argument("--decoder_mode", default=None, help="maskctc or None")

        # LabelSmoothingLoss
        self.parser.add_argument("--lsm_weight", default="")
        self.parser.add_argument("--transformer_length_normalized_loss", default=False)
        self.parser.add_argument("--subsample", default="1_1_1_1")
        self.parser.add_argument("--sym_space", default="<space>")
        self.parser.add_argument("--sym_blank", default="<black>")
        self.parser.add_argument("--report_cer", default=False)
        self.parser.add_argument("--report_wer", default=False)

        # attention--
        self.parser.add_argument("--", default="")
        self.parser.add_argument("--", default="")
        self.parser.add_argument("--", default="")
        self.parser.add_argument("--", default="")
        self.parser.add_argument("--", default="")

        # recog
        self.parser.add_argument("--beam_size", default=12)
        self.parser.add_argument("--penalty", default=0.0)
        self.parser.add_argument("--ctc_weight", default=0.2)
        self.parser.add_argument("--maxlenratio", default=10.0)
        self.parser.add_argument("--minlenratio", default=0.0)
        self.parser.add_argument("--lm_weight", default=0.0)
        self.parser.add_argument("--nbest", default=1)
        self.parser.add_argument("--maskctc_probability_threshold", default=0.0)
        self.parser.add_argument("--maskctc_n_iterations", default=1)
        self.parser.add_argument("--", default="")
        self.parser.add_argument("--", default="")
        self.parser.add_argument("--", default="")
        self.parser.add_argument("--", default="")
        self.parser.add_argument("--", default="")
        self.parser.add_argument("--", default="")
        # minibatch related
        self.parser.add_argument("--batch_size", "-b", default=30, type=int, help="Batch size")
        self.parser.add_argument("--maxlen-in", default=800, type=int, metavar="ML", help="Batch size is reduced if the input sequence length > ML")
        self.parser.add_argument("--maxlen-out", default=150, type=int, metavar="ML", help="Batch size is reduced if the output sequence length > ML")
        self.parser.add_argument("--verbose", default=1, type=int, help="Verbose option")
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        str_ids = self.opt.gpu_ids.split(",")
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        if self.opt.mtlalpha == 1.0:
            self.opt.mtl_mode = "ctc"
        elif self.opt.mtlalpha == 0.0:
            self.opt.mtl_mode = "att"
        else:
            self.opt.mtl_mode = "mtl"

        args = vars(self.opt)

        print("------------ Options -------------")
        for k, v in sorted(args.items()):
            print("%s: %s" % (str(k), str(v)))
        print("-------------- End ----------------")

        # save to the disk
        exp_path = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        utils.mkdirs(exp_path)
        self.opt.exp_path = exp_path
        file_name = os.path.join(exp_path, "arguments.txt")
        with open(file_name, "wt") as opt_file:
            opt_file.write("------------ Options -------------\n")
            for k, v in sorted(args.items()):
                opt_file.write("%s: %s\n" % (str(k), str(v)))
            opt_file.write("-------------- End ----------------\n")
        return self.opt