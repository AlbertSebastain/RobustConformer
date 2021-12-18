from unicodedata import bidirectional
from e2e_asr_conformer import E2E
import numpy as np
import torch
import torch.optim as optim
from transformer import nets_utils
from e2e_asr_conformer import E2E
import os
import torch.nn.functional as F

class condition_layer(torch.nn.Module):
    def __init__(self, idim, layers, odim, hidden_dim, projection_dim = 256):
        super(condition_layer, self).__init__()
        #self.layer = torch.nn.LSTM(input_size = idim, hidden_size = hidden_dim, num_layers = layers, bias = True, batch_first = True, dropout = 0.1, bidirectional = True)
        #self.output_layer = torch.nn.Linear(2*hidden_dim, projection_dim)
        #self.l_layer = torch.nn.Linear(projection_dim, odim, bias = False)
        self.odim = odim

        self.layer = torch.nn.LSTM(input_size = idim, hidden_size = hidden_dim, num_layers = layers, bias = True, batch_first = True, dropout = 0.1, bidirectional = False)
        self.output_layer = torch.nn.Linear(hidden_dim, odim, bias = False)

    def forward(self, hs, ys):
        y,hc = self.layer(hs)   
        y = y[:,-1,:]
        batch = y.size(0)
        #out = torch.tanh(self.output_layer(y))
        out = self.output_layer(y)
        #out = torch.sigmoid(out)
        
        device = out.device
        ys = torch.tensor(ys).to(device)
        #ys_tensor = torch.zeros(batch,self.odim)
        #ys_tensor = ys_tensor.scatter(1,ys-1,1)
        loss = F.cross_entropy(input = out, target = ys, reduction = "none")
        acc = torch.sum(torch.argmax(out,1) == ys).item()/batch
        loss = loss.sum()/batch
        return loss, acc



class Teacher_Student_Conformer(torch.nn.Module):
    def __init__(self, opt, teacher_model = None):
        super(Teacher_Student_Conformer, self).__init__()
        if teacher_model is None:
            self.teacher_model = E2E(opt.teacher_network)
        else:
            self.teacher_model = teacher_model
        self.student_model = E2E(opt.student_network)
        self.student_model = self.student_model.cuda(0)
        self.teacher_model.eval()
        self.teacher_model = self.teacher_model.cuda(1)
        self.attention_temperature = opt.attention_temperature
        self.top_k = opt.top_k
        if opt.use_condition_layer:
            self.condition_layer_model = condition_layer(opt.student_network.adim, opt.condition_layers, opt.condition_speak_id_nums,opt.condition_hidden_dim)
            self.condition_layer_model = self.condition_layer_model.cuda(0)
            self.condition_layer_model.train()
            self.condition_weight = opt.condition_weight
            self.start_condition_epoch = opt.start_condition_epoch
            self.stop_condition_epoch = self.start_condition_epoch + opt.no_condition_epoch
        else:
            self.condition_layer_model = None
            self.condition_weight = 0.0
        for par in self.teacher_model.parameters():
            par.requires_grad = False
        self.student_model.train()
        self.teacher_adim = self.teacher_model.adim
        self.student_adim = self.student_model.adim
        self.tau = opt.tau
        self.weight = opt.skd_weight
        self.mtlalpha = opt.student_network.mtlalpha
        self.attention_logit_weight = opt.attention_logit_weight
        self.conv_layer = torch.nn.Conv1d(in_channels = self.student_adim, out_channels = self.teacher_adim, kernel_size = 1)

    def RKD_train(self, input_features, input_sizes, targets, target_sizes):
        _,_ = self.teacher_model(input_features, input_sizes, targets, target_sizes)
        student_loss, student_acc = self.student_model(input_features, input_sizes, targets, target_sizes)
        teacher_encoder_hs = nets_utils.to_device(self.student_model,self.teacher_model.hs_pad)
        student_encoder_hs = self.student_model.hs_pad
        student_encoder_hs_T = student_encoder_hs.permute(0,2,1)
        out = self.conv_layer(student_encoder_hs_T)
        out = out.permute(0,2,1)
        teacher_encoder_hs_avr = torch.mean(teacher_encoder_hs, dim = -1, keepdim = True)
        teacher_encoder_hs_avr = torch.sigmoid(teacher_encoder_hs_avr)
        M_FW = teacher_encoder_hs_avr.repeat(1,1,self.teacher_adim)
        out_student = torch.mul(M_FW, out)
        out_teacher = torch.mul(M_FW, teacher_encoder_hs)
        loss = self.l2_loss(out_student, out_teacher)
        return loss,student_loss, student_acc

    def forward(self, teacher_input, student_input, input_sizes, targets, target_sizes, speak_ids = None,  epochs = 0):
        _,_ = self.teacher_model(teacher_input, input_sizes, targets, target_sizes)
        loss, acc = self.student_model(student_input, input_sizes, targets, target_sizes)
        loss_att = self.student_model.loss_att
        loss_ctc = self.student_model.loss_ctc
        teacher_ctc_logit = nets_utils.to_device(self.student_model, self.teacher_model.ctc.ys_hat)
        student_ctc_logit = self.student_model.ctc.ys_hat
        teacher_ctc_logit = torch.softmax(teacher_ctc_logit / self.tau, -1)
        student_ctc_logit = torch.softmax(student_ctc_logit /self.tau, -1)
        teacher_attention_logit = self.teacher_model.pred_pad
        student_attention_logit = self.student_model.pred_pad
        teacher_pred = self.selection_top_k_softmax(teacher_attention_logit)
        student_pred = F.softmax(student_attention_logit,-1)
        teacher_pred = nets_utils.to_device(self.student_model, teacher_pred)
        loss_attention_logit = self.kl_div_loss(student_pred, teacher_pred)
        loss_skd = self.l2_loss(student_ctc_logit,teacher_ctc_logit)
        loss_att_sum = (1-self.attention_logit_weight) * loss_att + self.attention_logit_weight * loss_attention_logit
        hs = self.student_model.hs_pad
        if self.condition_layer_model is not None and epochs >= self.start_condition_epoch and epochs <= self.stop_condition_epoch:
            loss_condition, condition_acc = self.condition_layer_model(hs,speak_ids)

        else:
            loss_condition = torch.tensor([0.0])
            device = loss_att_sum.device
            loss_condition = loss_condition.to(device)
            condition_acc = 0.0
        #loss_sum = (1-self.mtlalpha) * loss_att_sum + self.mtlalpha * loss_ctc + self.weight * loss_skd + self.condition_weight * loss_condition
        loss_sum = (1-self.mtlalpha) * loss_att_sum + self.mtlalpha * loss_ctc + self.weight * loss_skd - self.condition_weight * loss_condition
        self.loss_sum = loss_sum
        
        return loss, loss_att_sum, loss_attention_logit, loss_skd, loss_condition, loss_sum, acc, condition_acc
    
    def backward_with_condition(self):
        for _,para in self.condition_layer_model.named_parameters():
            if para.requires_grad:
                para.grad = para.grad/(-self.condition_weight)
        


    def l2_loss(self, x, y):
        loss_dummy = F.mse_loss(x, y, reduction="none")
        loss = torch.sum(loss_dummy)/x.size(0)
        return loss

    def kl_div_loss(self,x,y):
        loss_dummy = F.kl_div(x.log(), y, reduction="none")
        loss = torch.sum(loss_dummy)/x.size(0)
        return loss

    def freeze_grad_before_RKD(self):
        for name, para in self.student_model.named_parameters():
            if "encoder" in name:
                para.requires_grad = True
            else:
                para.requires_grad = False
        self.conv_layer.requires_grad = True
    def release_grad_before_SKD(self):
        for para in self.student_model.parameters():
            para.requires_grad = True
        self.conv_layer.requires_grad = False

    def get_optim_para(self):
        para = [x for x in self.student_model.parameters()]
        for x in self.conv_layer.parameters():
            para.append(x)
        return para

    def selection_top_k_softmax(self,ys_hat):
        topk_value,topk_index = ys_hat.topk(self.top_k,-1)
        C = -600
        mod_ys = torch.ones(ys_hat.size())*C
        mod_ys = nets_utils.to_device(ys_hat,mod_ys)
        #  select top k, the others will set to C
        mod_ys.scatter_(-1,topk_index, topk_value)
        mod_ys = mod_ys/self.attention_temperature
        q_hat = F.softmax(mod_ys,-1)
        return q_hat

    @classmethod
    def load_model(cls, path, state_dict, opt=None):
        """classmethode类型，代表可以不需要实例化，直接使用该函数，并使用cls使用该类的属性等内容。
        这个函数可以读取模型。

        Args:
            path (str): 模型的地址
            state_dict (dict): 模型的参数

        Returns:
            model类: 模型
        """
        if path is not None:
            # # Load all tensors onto GPU 1
            package = torch.load(path, map_location=lambda storage, loc: storage)
            # 加载opt
            model = cls(package["opt"])
            print("model.state_dict() is", model.state_dict().keys())
            if state_dict in package and package[state_dict] is not None:
                model.load_state_dict(package[state_dict])
                print("package.state_dict() is", package[state_dict].keys())
                print("checkpoint found at {} {}".format(path, state_dict))

        print(model)
        return model

        


        