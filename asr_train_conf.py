#from data.labelparse import Labelparse

import math
import random


import itertools
import numpy as np
import torch
import torch.optim as optim
import os
import torch.nn.functional as F
from data import data_loader

from utils.visualizer import Visualizer
from utils.utils import ScheSampleRampup, save_checkpoint, adadelta_eps_decay
from tqdm import tqdm
from librosa.feature import delta

from transformer.optimizer import NoamOpt
from transformer.nets_utils import pad_list
from e2e_asr_conformer import E2E
from conformer_options.train_conformer_options import Train_conformer_Options
import fake_opt
from model.feat_model import FbankModel
from data.SpecAugment import spec_augmentation
from model.VAT import VAT
import datetime

import os 
def save_opt(arg,file_path):
    dict_arg = take_args(arg=arg)
    file_name = os.path.join(file_path,"opt.txt")
    with open(file_name,mode = "w",encoding = "utf-8") as name:
        for var, val in dict_arg.items():
            name.write("{}:{}\n".format(var, val))
            
def take_args(prefix='',arg=None):
    dict_arg = {}
    for var, val in arg.__dict__.items():
        if(hasattr(val, "__dict__")):
            dict_val = take_args(var+'.',val)
            dict_arg.update(dict_val)
        else:
            dict_arg[prefix+var] = val
    return dict_arg
SEED = random.randint(1, 10000)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
def train():
    opt = Train_conformer_Options().parse()
    #save_opt(arg=opt, file_path=opt.exp_path)
    if opt.use_random_mix_noise:
        from data.rand_cleandataloader import SequentialDataset, SequentialDataLoader, BucketingSampler
    else:
        from data.data_loader import SequentialDataset, SequentialDataLoader,BucketingSampler
    device = torch.device("cuda:{}".format(opt.gpu_ids[0]) if len(opt.gpu_ids) > 0 and torch.cuda.is_available() else "cpu")

    visualizer = Visualizer(opt)
    logging = visualizer.get_logger()
    acc_report = visualizer.add_plot_report(["train/acc", "val/acc"], "acc.png")
    loss_report = visualizer.add_plot_report(["train/loss", "val/loss"], "loss.png")

    train_fold = opt.train_folder
    dev_fold = opt.dev_folder
    train_dataset = SequentialDataset(opt, os.path.join(opt.dataroot, train_fold), os.path.join(opt.dict_dir, 'train_units.txt'),type_data = 'train') 
    val_dataset = data_loader.SequentialDataset(opt, os.path.join(opt.dataroot, dev_fold), os.path.join(opt.dict_dir, 'train_units.txt'),type_data = 'dev')    
    train_sampler = BucketingSampler(train_dataset,batch_size = opt.batch_size)
    train_loader = SequentialDataLoader(train_dataset, num_workers=opt.num_workers, batch_sampler=train_sampler)
    val_loader = data_loader.SequentialDataLoader(val_dataset, batch_size=int(opt.batch_size/2), num_workers=opt.num_workers, shuffle=False)
    # add new parameters
    opt.idim = train_dataset.get_feat_size()
    opt.odim = train_dataset.get_num_classes()
    opt.char_list = train_dataset.get_char_list()
    opt.num_speak_ids = train_dataset.num_ids
    opt.train_dataset_len = len(train_dataset)
    FGSM_augmentation = opt.FGSM_augmentation
    logging.info("#input dims : " + str(opt.idim))
    logging.info("#output dims: " + str(opt.odim))
    logging.info("Dataset ready!")
    #asr_model = E2E(opt.idim, opt.odim, opt)
    asr_model = E2E(opt)
    fbank_model = FbankModel(opt)
    use_vat = opt.use_vat
    lr = opt.lr  # default=0.005
    eps = opt.eps  # default=1e-8
    iters = opt.iters  # default=0
    start_epoch = opt.start_epoch  # default=0
    best_loss = opt.best_loss  # default=float('inf')
    best_acc = opt.best_acc  # default=0
    model_path = None
    pre_valid_acc = 0
    # convert to cuda
    #fbank_model.cuda()
    if opt.resume:
        model_path = os.path.join(opt.works_dir, opt.resume)
        if os.path.isfile(model_path):
            package = torch.load(model_path, map_location=lambda storage, loc: storage)
            lr = package.get('lr', opt.lr)
            eps = package.get('eps', opt.eps)        
            best_loss = package.get('best_loss', float('inf'))
            best_acc = package.get('best_acc', 0)
            start_epoch = int(package.get('epoch', 0))   
            iters = int(package.get('iters', 0))
            iters = iters-1
            pre_valid_acc = best_acc

            acc_report = package.get('acc_report', acc_report)
            loss_report = package.get('loss_report', loss_report)
            visualizer.set_plot_report(acc_report, 'acc.png')
            visualizer.set_plot_report(loss_report, 'loss.png')
            

            logging.info('Loading model {} and iters {}'.format(model_path, iters))
        else:
            print("no checkpoint found at {}".format(model_path))     
    if FGSM_augmentation:
        logging.info("epsilon_FGSM {},alpha_FGSM {} start epoch is {}".format(opt.epsilon_FGSM,opt.alpha_FGSM,opt.start_augmentation))
    asr_model = E2E.load_model(model_path, 'asr_state_dict',opt) 
    fbank_model = FbankModel.load_model(model_path, 'fbank_state_dict',opt) 
    parameters = filter(lambda p: p.requires_grad, itertools.chain(asr_model.parameters()))
    optimizer = torch.optim.Adam(parameters,lr = lr,betas = (opt.beta1,0.98), eps=eps)
    if opt.opt_type == 'noam':
        optimizer = NoamOpt(asr_model.adim, opt.transformer_lr, opt.transformer_warmup_steps, optimizer,iters)
    asr_model.cuda()
    print(asr_model)
    asr_model.train()
    vat_scheme = VAT(asr_model, opt.vat_epsilon)
    #sample_rampup = ScheSampleRampup(opt.sche_samp_start_iter, opt.sche_samp_final_iter, opt.sche_samp_final_rate)
    #sche_samp_rate = sample_rampup.update(iters)
    fbank_cmvn_file = os.path.join(opt.exp_path, 'fbank_cmvn.npy')
    if os.path.exists(fbank_cmvn_file):
            fbank_cmvn = np.load(fbank_cmvn_file)
    else:
        for i, (data) in enumerate(train_loader, start=0):
            utt_ids, spk_ids, inputs, log_inputs, targets, input_sizes, target_sizes = data
            fbank_cmvn = fbank_model.compute_cmvn(inputs, input_sizes)
            if fbank_model.cmvn_processed_num >= fbank_model.cmvn_num:
                #if fbank_cmvn is not None:
                fbank_cmvn = fbank_model.compute_cmvn(inputs, input_sizes)
                np.save(fbank_cmvn_file, fbank_cmvn)
                print('save fbank_cmvn to {}'.format(fbank_cmvn_file))
                break
    fbank_cmvn = torch.FloatTensor(fbank_cmvn)
    for epoch in range(start_epoch, opt.epochs):
        if epoch > opt.shuffle_epoch:
            print(">> Shuffling batches for the following epochs")
            train_sampler.shuffle(epoch)
        for i, (data) in enumerate(train_loader, start=(iters * opt.batch_size) % len(train_dataset)):
            utt_ids, spk_ids, inputs, log_inputs, targets, input_sizes, target_sizes = data
            fbank_features = fbank_model(inputs, fbank_cmvn)
            if opt.use_shift:
                batch, length, dim = fbank_features.shape
                fbank_features = fbank_features.transpose(1,2)
                shift = opt.Shift_frames
                length = length - shift
                if shift > 0:
                    offsets = torch.randint(
                        shift,
                        [batch, 1, 1], device=fbank_features.device)
                    offsets = offsets.expand(-1, dim, -1)
                    indexes = torch.arange(length, device=fbank_features.device)
                    fbank_features = fbank_features.gather(2, indexes + offsets)
                    fbank_features = fbank_features.transpose(1,2)
                    input_sizes = input_sizes-shift
            if opt.use_spec_aug:
                fbank_features = spec_augmentation(fbank_features, opt.SpecF, opt.SpecT, 2)
            if opt.use_delta:
                feature_device = fbank_features.device
                feature_delta = delta(fbank_features.cpu().numpy())
                feature_delta_delta = delta(fbank_features.cpu().numpy(),order = 2)
                feature_delta = torch.from_numpy(feature_delta).to(feature_device)
                feature_delta_delta = torch.from_numpy(feature_delta_delta).to(feature_device)
                fbank_features = torch.cat([fbank_features,feature_delta,feature_delta_delta], dim = -1)
            loss_vat = torch.tensor([0])
            loss_adv = torch.tensor([0])
            if FGSM_augmentation and epoch >= opt.start_augmentation and random.random() <= opt.p_aug:
                fbank_features.requires_grad = True
                loss, acc = asr_model(fbank_features, input_sizes, targets, target_sizes)
                loss.backward(retain_graph = True)
                grad_fbank = fbank_features.grad.data
                adv_fbank = fbank_features + opt.epsilon_FGSM*torch.sign(grad_fbank)
                loss_adv,acc_ad = asr_model(adv_fbank, input_sizes, targets, target_sizes)
                loss += opt.alpha_FGSM * loss_adv
                # FGSM_delta = torch.zeros_like(fbank_features).uniform_(-opt.epsilon_FGSM, opt.epsilon_FGSM).cuda()
                # FGSM_delta.requires_grad = True
                # loss, acc = asr_model(fbank_features+FGSM_delta, input_sizes, targets,target_sizes)
                # loss.backward()
                # grad = FGSM_delta.grad
                # FGSM_delta.data = torch.clamp(FGSM_delta+opt.alpha_FGSM*torch.sign(grad),-opt.epsilon_FGSM,opt.epsilon_FGSM)
                # FGSM_delta.detach()
                # loss, acc = asr_model(fbank_features+FGSM_delta, input_sizes, targets,target_sizes)
            elif use_vat and epoch >= opt.start_augmentation and random.random() <= opt.p_aug:
                d, loss_vat = vat_scheme.compute_vat_data([fbank_features, input_sizes, targets, target_sizes],opt.vat_iter,optimizer)
                loss,acc = asr_model(fbank_features, input_sizes, targets, target_sizes)
                loss += opt.vat_delta_weight*loss_vat
            else:
                loss, acc = asr_model(fbank_features, input_sizes, targets, target_sizes)
            optimizer.zero_grad()  # Clear the parameter gradients
            loss.backward()  # compute backwards
            grad_norm = torch.nn.utils.clip_grad_norm_(asr_model.parameters(), opt.grad_clip)
            if math.isnan(grad_norm):
                logging.warning(">> grad norm is nan. Do not update model.")
            else:
                optimizer.step()

            iters += 1
            if FGSM_augmentation:
                errors = {
                    "train/loss": loss.item(),
                    "train/FGSM_loss": loss_adv.item(),
                    "train/acc": acc,
                }
            elif use_vat:
                errors = {
                    "train/loss": loss.item(),
                    "train/vat_loss":loss_vat.item(),
                    "train/acc":acc,
                }
            else:
                errors = {
                    "train/loss": loss.item(),
                    "train/acc": acc,
                }
            visualizer.set_current_errors(errors)
    
            # print
            if iters % opt.print_freq == 0:
                visualizer.print_current_errors(epoch, iters)
                state = {
                    "asr_state_dict": asr_model.state_dict(),
                    "opt": opt,
                    "epoch": epoch,
                    "iters": iters,
                    "eps": opt.eps,
                    "lr": opt.lr,
                    "best_loss": best_loss,
                    "best_acc": best_acc,
                    "acc_report": acc_report,
                    "loss_report": loss_report,
                }
                filename = "latest"
                save_checkpoint(state, opt.exp_path, filename=filename)
            # evalutation
            if iters % opt.validate_freq == 0:
                asr_model.eval()
                torch.set_grad_enabled(False)
                pbar = tqdm(total=len(val_dataset))
                for i, (data) in enumerate(val_loader, start=0):
                    utt_ids, spk_ids, inputs, log_inputs, targets, input_sizes, target_sizes = data

                    fbank_features = fbank_model(inputs, fbank_cmvn)
                    if opt.use_delta:
                        feature_device = fbank_features.device
                        feature_delta = delta(fbank_features.cpu().numpy())
                        feature_delta_delta = delta(fbank_features.cpu().numpy(),order = 2)
                        feature_delta = torch.from_numpy(feature_delta).to(feature_device)
                        feature_delta_delta = torch.from_numpy(feature_delta_delta).to(feature_device)
                        fbank_features = torch.cat([fbank_features,feature_delta,feature_delta_delta], dim = -1)
                    loss,acc = asr_model(fbank_features, input_sizes, targets,target_sizes)

                    #loss = opt.mtlalpha * loss_ctc + (1 - opt.mtlalpha) * loss_att
                    errors = {
                        "val/loss": loss.item(),
                        "val/acc": acc,
                    }
                    visualizer.set_current_errors(errors)
                    pbar.update(opt.batch_size/2)
                pbar.close()
                asr_model.train()
                torch.set_grad_enabled(True)

                visualizer.print_epoch_errors(epoch, iters)
                acc_report = visualizer.plot_epoch_errors(epoch, iters, "acc.png")
                loss_report = visualizer.plot_epoch_errors(epoch, iters, "loss.png")
                val_loss = visualizer.get_current_errors("val/loss")
                val_acc = visualizer.get_current_errors("val/acc")
                filename = None
                if opt.criterion == "acc" and opt.mtl_mode != "ctc":
                    if val_acc < best_acc:
                        logging.info("val_acc {} > best_acc {}".format(val_acc, best_acc))
                        #opt.eps = adadelta_eps_decay(optimizer, opt.eps_decay)  # Epsilon constant for optimizer
                    else:
                        filename = "model.acc.best"
                    best_acc = max(best_acc, val_acc) 
                    logging.info("best_acc {}".format(best_acc))
                elif opt.criterion == "loss":
                    if val_loss > best_loss:
                        logging.info("val_loss {} > best_loss {}".format(val_loss, best_loss))
                        #opt.eps = adadelta_eps_decay(optimizer, opt.eps_decay)
                    else:
                        filename = "model.loss.best"
                    best_loss = min(val_loss, best_loss)
                    logging.info("best_loss {}".format(best_loss))
                state = {
                    "asr_state_dict": asr_model.state_dict(),
                    "opt": opt,
                    "epoch": epoch,
                    "iters": iters,
                    "eps": opt.eps,
                    "lr": opt.lr,
                    "best_loss": best_loss,
                    "best_acc": best_acc,
                    "acc_report": acc_report,
                    "loss_report": loss_report,
                }
                save_checkpoint(state, opt.exp_path, filename=filename)

                visualizer.reset()
                pre_valid_acc = best_acc
        if (FGSM_augmentation or use_vat) and epoch == opt.start_augmentation-1:
            filename = "model.{}".format(opt.start_augmentation-1)
            logging.info("Before start augmentation save model")
            state = {
                        "asr_state_dict": asr_model.state_dict(),
                        "opt": opt,
                        "epoch": epoch,
                        "iters": iters,
                        "eps": opt.eps,
                        "lr": opt.lr,
                        "best_loss": best_loss,
                        "best_acc": best_acc,
                        "acc_report": acc_report,
                        "loss_report": loss_report,
                    }
            save_checkpoint(state, opt.exp_path, filename=filename)


if __name__ == "__main__":
    train()
