import torch
import torch.nn.functional as F

class VAT:
    def __init__(self, asr_model, vat_epsilon):
        self.asr_model = asr_model
        self.vat_epsilon = vat_epsilon

    def compute_vat_data(self, input_data, vat_iter,optimizer):
        input = input_data[0]
        device = input.device
        d = torch.randn(input.size())
        d = d.to(device)
        d = F.normalize(d,p=2,dim=-1)
        delta = self.vat_epsilon*d
        delta.requires_grad = True
        for i in range(vat_iter):
            vat_loss = self.compute_vat_loss(input_data, delta)
            vat_loss.backward()
            d = delta.grad.data
            d = F.normalize(d,p=2, dim=-1)
            delta = self.vat_epsilon*d
            optimizer.zero_grad()
            # delta.grad.data.zero_()
            delta.requires_data = True
        vat_loss = self.compute_vat_loss(input_data, delta)
        return delta, vat_loss


    def compute_vat_loss(self, input_data, delta):
        input = input_data[0]
        with torch.no_grad():
            loss,acc = self.asr_model(*input_data)
            P = F.softmax(self.asr_model.pred_pad, -1)
        fbank_hat = input+delta
        _,_, = self.asr_model(fbank_hat, *input_data[1:])
        Q = F.softmax(self.asr_model.pred_pad, -1)
        vat_loss = F.kl_div(Q.log(),P, reduction = "batchmean")
        #vat_loss = vat_loss.sum()/P.size(0)
        return vat_loss




    def update_data(self, input_data):
        self.input_data = input_data[1:]
        self.fbank_features = input_data[0]

    def update_iter(self, d,reduction = "sum", iter = True, delta_weight = 1):
        if not iter:
            delta_v = delta_weight*d
            fbank_hat = self.fbank_features+delta_v
        else:
            vat_delta = self.vat_epsilon*d
            device = self.fbank_features.device
            vat_delta = vat_delta.to(device)
            vat_delta.requires_grad = True
            fbank_hat = self.fbank_features+vat_delta
        loss,acc = self.asr_model(self.fbank_features, *self.input_data)
        self.loss = loss
        self.acc = acc
        P = F.softmax(self.asr_model.pred_pad, -1)
        _,_ = self.asr_model(fbank_hat, *self.input_data)
        Q = F.softmax(self.asr_model.pred_pad,-1)
        vat_loss = F.kl_div(P.log(), Q, reduction = reduction)
        self.vat_loss = vat_loss
        if iter:
            vat_loss.backward(retain_graph = True)
            self.vat_loss = vat_loss
            d = vat_delta.grad.data
            d = F.normalize(d, p=2, dim=-1)
        return d

    
    def get_loss(self,weight):
        self.loss += weight*self.vat_loss
        return self.loss, self.acc


