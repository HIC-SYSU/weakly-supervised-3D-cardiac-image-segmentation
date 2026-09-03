from torch.nn import CrossEntropyLoss,MSELoss,Module
from monai.losses import DiceLoss

class loss_call(Module):
    def __init__(self, pce_args,MSELoss_arg,DiceCeLoss_arg):
        super(loss_call, self).__init__()
        self.pce=CrossEntropyLoss(**pce_args)#ignore_index=pce_args['ignore_label'],reduction='mean')
        self.MseLoss=MSELoss(**MSELoss_arg)
        self.DiceLoss=DiceLoss(**DiceCeLoss_arg)


    def forward(self, pred_target,loss_fn_list,weight):
        loss_list=[]
        loss=0
        for step,(pred,target) in enumerate(pred_target):
            loss_fn=getattr(self,loss_fn_list[step])
            loss_step=loss_fn(pred,target)#.requires_grad()
            loss_list.append(loss_step)
            loss=loss+(loss_step*weight[step])#.requires_grad()

        return loss_list,loss

#d=loss_call(pce_args,MSELoss_arg,consist_loss_arg)