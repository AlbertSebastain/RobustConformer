import math
import random


import itertools
import numpy as np
import torch
import torch.optim as optim
import os
import torch.nn.functional as F
from data import mix_data_loader

from utils.visualizer import Visualizer
from utils.utils import ScheSampleRampup, save_checkpoint, adadelta_eps_decay
from tqdm import tqdm
from librosa.feature import delta

from transformer.optimizer import NoamOpt
from transformer.nets_utils import pad_list
from e2e_asr_conformer import E2E
from data.SpecAugment import spec_augmentation
from model.KD_conformer import Teacher_Student_Conformer
from conformer_options.teacher_student_options import Teacher_Student_Options
import fake_opt
from model.feat_model import FbankModel
from model.VAT import VAT
#torch.cuda.set_device(1)
SEED = random.randint(1, 10000)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
# def save_opt(arg,file_path):
#     dict_arg = take_args(arg=arg)
#     file_name = os.path.join(file_path,"opt.txt")
#     with open(file_name,mode = "w",encoding = "utf-8") as name:
#         for var, val in dict_arg.items():
#             name.write("{}:{}\n".format(var, val))
            
# def take_args(prefix='',arg=None):
#     dict_arg = {}
#     for var, val in arg.__dict__.items():
#         if(hasattr(val, "__dict__")):
#             dict_val = take_args(var+'.',val)
#             dict_arg.update(dict_val)
#         else:
#             dict_arg[prefix+var] = val
#     return dict_arg
            
def add_delta(features):
    feature_device = features.device
    feature_delta = delta(features.cpu().numpy())
    feature_delta_delta = delta(features.cpu().numpy(),order = 2)
    feature_delta = torch.from_numpy(feature_delta).to(feature_device)
    feature_delta_delta = torch.from_numpy(feature_delta_delta).to(feature_device)
    features = torch.cat([features,feature_delta,feature_delta_delta], dim = -1)
    return features

def add_shift(features, input_sizes, shift_frames):
    batch, length, dim = features.shape
    features = features.transpose(1,2)
    shift = shift_frames
    length = length - shift
    if shift > 0:
        offsets = torch.randint(
            shift,
            [batch, 1, 1], device=features.device)
        offsets = offsets.expand(-1, dim, -1)
        indexes = torch.arange(length, device=features.device)
        features = features.gather(2, indexes + offsets)
        features = features.transpose(1,2)
        input_sizes = input_sizes-shift
    return features, input_sizes

def train():
    #opt = fake_opt.teacher_student_conformer()
    opt = Teacher_Student_Options().parse()
    #save_opt(arg=opt, file_path=opt.exp_path)
    device = torch.device("cuda:{}".format(opt.gpu_ids[0]) if len(opt.gpu_ids) > 0 and torch.cuda.is_available() else "cpu")
    if opt.use_random_mix_noise:
        from data.mix_rand_dataloader import MixSequentialDataset, MixSequentialDataLoader, BucketingSampler
    else:
        from data.mix_data_loader import MixSequentialDataset, MixSequentialDataLoader,BucketingSampler
    visualizer = Visualizer(opt)
    logging = visualizer.get_logger()
    acc_report = visualizer.add_plot_report(["train/acc", "val/acc"], "acc.png")
    loss_report = visualizer.add_plot_report(["train/loss", "val/loss"], "loss.png")
    train_fold = opt.train_folder
    dev_fold = opt.dev_folder
    train_dataset = MixSequentialDataset(opt, os.path.join(opt.dataroot, train_fold), os.path.join(opt.dict_dir, 'train_units.txt'),train_fold) 
    val_dataset   = mix_data_loader.MixSequentialDataset(opt, os.path.join(opt.dataroot, dev_fold), os.path.join(opt.dict_dir, 'train_units.txt'),dev_fold)
    train_sampler = BucketingSampler(train_dataset, batch_size=opt.batch_size) 
    train_loader = MixSequentialDataLoader(train_dataset, num_workers=opt.num_workers, batch_sampler=train_sampler)
    val_loader = mix_data_loader.MixSequentialDataLoader(val_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)
    opt.idim = train_dataset.get_feat_size()
    opt.student_network.idim = train_dataset.get_feat_size()
    opt.teacher_network.idim = train_dataset.get_feat_size()
    opt.odim = train_dataset.get_num_classes()
    opt.condition_speak_id_nums = train_dataset.num_ids
    opt.student_network.odim = train_dataset.get_num_classes()
    opt.teacher_network.odim = train_dataset.get_num_classes()
    opt.char_list = train_dataset.get_char_list()
    opt.train_dataset_len = len(train_dataset)
    FGSM_augmentation = opt.FGSM_augmentation
    logging.info('#input dims : ' + str(opt.idim))
    logging.info('#output dims: ' + str(opt.odim))
    logging.info("Dataset ready!")
    fbank_model = FbankModel(opt)
    use_vat = opt.use_vat
    lr = opt.lr  # default=0.005
    eps = opt.eps  # default=1e-8
    iters = opt.iters  # default=0
    start_epoch = opt.start_epoch  # default=0
    best_loss = opt.best_loss  # default=float('inf')
    best_acc = opt.best_acc  # default=0
    start_RKD_epoch = 0
    model_path = None
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
            start_RKD_epoch = package.get('start_RKD_epoch', 0)
            pre_valid_acc = best_acc

            acc_report = package.get('acc_report', acc_report)
            loss_report = package.get('loss_report', loss_report)
            visualizer.set_plot_report(acc_report, 'acc.png')
            visualizer.set_plot_report(loss_report, 'loss.png')
            teach_model = Teacher_Student_Conformer.load_model(model_path,"asr_state_dict",opt)

            logging.info('Loading model {} and iters {}'.format(model_path, iters))
        else:
            print("no checkpoint found at {}".format(model_path))
    else:
        print("no checkpoint found, so init student model")
        if opt.teacher_resume is not None:
                teacher_model_path = os.path.join(opt.works_dir, opt.teacher_resume)
                package = torch.load(teacher_model_path, map_location=lambda storage, loc: storage)
                teacher_model =  E2E.load_model(teacher_model_path, 'asr_state_dict',opt)
                teach_model = Teacher_Student_Conformer(opt, teacher_model)
                opt.teacher_network = package['opt']
        else:
            raise Exception("No teacher model")
    fbank_model = FbankModel.load_model(model_path, 'fbank_state_dict',opt)
    vat_scheme = VAT(teach_model.student_model, opt.vat_epsilon)
    teach_model = teach_model.cuda()
    teach_model.condition_weight = opt.condition_weight
    teach_model.start_condition_epoch = opt.start_condition_epoch
    teach_model.stop_condition_epoch = opt.start_condition_epoch + opt.no_condition_epoch
    #teach_model = torch.nn.DataParallel(teach_model,device_ids = [0,1])
    #teach_model = teach_model.module
    teach_model.freeze_grad_before_RKD()    
    print(teach_model.student_model)
    fbank_clean_cmvn_file = os.path.join(opt.exp_path,"fbank_clean_cmvn.npy")
    if os.path.exists(fbank_clean_cmvn_file):
            fbank_clean_cmvn = np.load(fbank_clean_cmvn_file)
    else:
        for i, (data) in enumerate(train_loader, start=0):
            _, _, clean_inputs, _, _, _, _, _, input_sizes, _, _, _, _ = data
            fbank_clean_cmvn = fbank_model.compute_cmvn(clean_inputs, input_sizes)
        
            if fbank_model.cmvn_processed_num >= fbank_model.cmvn_num:
                #if fbank_cmvn is not None:
                fbank_clean_cmvn = fbank_model.compute_cmvn(clean_inputs, input_sizes)
                np.save(fbank_clean_cmvn_file, fbank_clean_cmvn)
                print('save clean fbank_cmvn to {}'.format(fbank_clean_cmvn_file))
                break
    fbank_clean_cmvn = torch.FloatTensor(fbank_clean_cmvn)

    fbank_noise_cmvn_file = os.path.join(opt.exp_path,"fbank_noise_cmvn.npy")
    if os.path.exists(fbank_noise_cmvn_file):
            fbank_noise_cmvn = np.load(fbank_noise_cmvn_file)
    else:
        fbank_model.cmvn_processed_num = 0
        for i, (data) in enumerate(train_loader, start=0):
            _, _, _, _, mix_inputs, _, _, _, input_sizes, _, _, _, _ = data
            fbank_noise_cmvn = fbank_model.compute_cmvn(mix_inputs, input_sizes)
        
            if fbank_model.cmvn_processed_num >= fbank_model.cmvn_num:
                #if fbank_cmvn is not None:
                fbank_noise_cmvn = fbank_model.compute_cmvn(mix_inputs, input_sizes)
                np.save(fbank_noise_cmvn_file, fbank_noise_cmvn)
                print('save noise fbank_cmvn to {}'.format(fbank_noise_cmvn_file))
                break
    fbank_noise_cmvn = torch.FloatTensor(fbank_noise_cmvn)
    teach_model.release_grad_before_SKD()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, teach_model.parameters()),lr = lr,betas = (opt.beta1,0.98), eps=eps)
    if opt.opt_type == 'noam':
        optimizer = NoamOpt(teach_model.student_model.adim, opt.transformer_lr, opt.transformer_warmup_steps, optimizer,iters)
        #save_checkpoint(state, opt.exp_path, filename=filename)
    # iters = 0
    visualizer.reset()
    for epoch in range(start_epoch, opt.epochs):
        if epoch > opt.shuffle_epoch:
            print(">> Shuffling batches for the following epochs")
            train_sampler.shuffle(epoch)
        for i, (data) in enumerate(train_loader, start=(iters * opt.batch_size) % len(train_dataset)):
            utt_ids, spk_ids, clean_inputs, clean_log_inputs, mix_inputs, mix_log_inputs, cos_angles, targets, input_sizes, target_sizes, mix_angles, clean_agles, cmvn = data
            fbank_clean_features = fbank_model(clean_inputs, fbank_clean_cmvn)
            fbank_noise_features = fbank_model(mix_inputs, fbank_noise_cmvn)
            if opt.use_shift:
                fbank_noise_features, input_sizes = add_shift(fbank_noise_features, input_sizes, opt.Shift_frames)
            if opt.use_spec_aug:
                fbank_noise_features = spec_augmentation(fbank_noise_features, opt.SpecF, opt.SpecT, 2)
            if opt.use_delta:
                fbank_noise_features = add_delta(fbank_noise_features)
            if hasattr(opt.teacher_network,"use_delta") and opt.teacher_network.use_delta:
                fbank_clean_features = add_delta(fbank_clean_features)
            if FGSM_augmentation and epoch >= opt.start_augmentation and random.random() <= opt.p_aug:
                fbank_noise_features.requires_grad = True
                original_loss, loss_mix_att, loss_attention_logit, loss_skd, loss_condition, loss, acc, condition_acc = teach_model(fbank_clean_features, fbank_noise_features, input_sizes, targets, target_sizes, spk_ids, epoch)
                loss.backward(retain_graph=True)
                grad_fbank = fbank_noise_features.grad.data
                fgsm_fbank = fbank_noise_features + opt.epsilon_FGSM*torch.sign(grad_fbank)
                loss_fgsm, _ = teach_model.student_model(fgsm_fbank, input_sizes, targets, target_sizes)
                loss += opt.alpha_FGSM * loss_fgsm
            elif use_vat and epoch >= opt.start_augmentation and random.random() <= opt.p_aug:
                d, loss_vat = vat_scheme.compute_vat_data([fbank_noise_features, input_sizes, targets, target_sizes],opt.vat_iter,optimizer)
                original_loss, loss_mix_att, loss_attention_logit, loss_skd, loss_condition, loss, acc, condition_acc = teach_model(fbank_clean_features, fbank_noise_features, input_sizes, targets, target_sizes,spk_ids, epoch)
                loss += opt.vat_delta_weight*loss_vat
            else:
                original_loss, loss_mix_att, loss_attention_logit, loss_skd, loss_condition, loss, acc, condition_acc = teach_model(fbank_clean_features, fbank_noise_features,input_sizes, targets, target_sizes,spk_ids, epoch)
            optimizer.zero_grad()  # Clear the parameter gradients
            loss.backward()  # compute backwards
            if opt.use_condition_layer and epoch >= opt.start_condition_epoch and epoch <= opt.start_condition_epoch+opt.no_condition_epoch:
                teach_model.backward_with_condition()
            grad_norm = torch.nn.utils.clip_grad_norm_(teach_model.parameters(), opt.grad_clip)
            if math.isnan(grad_norm):
                logging.warning(">> grad norm is nan. Do not update model.")
            else:
                optimizer.step()
            iters += 1
            errors = {
                "train/loss": loss.item(), 
                "train/loss_mix_att(att and attention_logit)": loss_mix_att.item(),
                "train/loss attention logit": loss_attention_logit.item(),
                "train/loss_skd": loss_skd.item(),
                "train/loss_traditional att and ctc": original_loss.item(),
                "train/loss_condition": loss_condition.item(),
                "train/acc":  acc,
                "train/condition_acc": condition_acc,

            }
            visualizer.set_current_errors(errors)
            if iters % opt.print_freq == 0:
                visualizer.print_current_errors(epoch, iters)
                state = {
                    "asr_state_dict": teach_model.state_dict(),
                    "asr_student_state_dict": teach_model.student_model.state_dict(),
                    "opt": opt,
                    "student_opt": opt.student_network,
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
                teach_model.student_model.eval()
                torch.set_grad_enabled(False)
                pbar = tqdm(total=len(val_dataset))
                for i, (data) in enumerate(val_loader, start=0):
                    utt_ids, spk_ids, clean_inputs, clean_log_inputs, mix_inputs, mix_log_inputs, cos_angles, targets, input_sizes, target_sizes, mix_angles, clean_agles, cmvn = data

                    fbank_clean_features = fbank_model(clean_inputs, fbank_clean_cmvn)
                    fbank_noise_features = fbank_model(mix_inputs, fbank_noise_cmvn)
                    if opt.use_delta:
                        fbank_noise_features = add_delta(fbank_noise_features)
                    if hasattr(opt.teacher_network,"use_delta") and opt.teacher_network.use_delta:
                        fbank_clean_features = add_delta(fbank_clean_features)
                    original_loss, loss_mix_att, loss_attention_logit, loss_skd, loss_condition, loss, acc, condition_acc= teach_model(fbank_clean_features, fbank_noise_features,input_sizes, targets, target_sizes, spk_ids)
                    #loss = opt.mtlalpha * loss_ctc + (1 - opt.mtlalpha) * loss_att
                    errors = {
                        "val/loss": loss.item(),
                        "val/loss_mix_att(att and attention_logit)": loss_mix_att.item(),
                        "val/loss attention logit": loss_attention_logit.item(),
                        "val/loss_skd": loss_skd.item(),
                        "train/loss_traditional att and ctc": original_loss.item(),
                        "val/acc": acc,
                        "val/loss_condition": loss_condition.item(),
                        "val/condition_acc": condition_acc,

                    }
                    visualizer.set_current_errors(errors)
                    pbar.update(opt.batch_size)
                pbar.close()
                teach_model.student_model.train()
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
                    "asr_state_dict": teach_model.state_dict(),
                    "asr_student_dict": teach_model.student_model.state_dict(),
                    "opt": opt,
                    "student_opt": opt.student_network,
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

if __name__ == "__main__":
    train()
