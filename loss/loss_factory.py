from class_loss import ClassLoss
from complex_loss import ComplexLoss

def create_loss(args,choose):
    if choose == 'class_loss':
        loss = ClassLoss(
               scale = args.loss.scale,
               lambda_g = args.loss.lambda_g
        )
    else:
        loss = ComplexLoss(
                local_loss=False,
                gather_with_grad=False,
                cache_labels=False,
                rank=0,
                world_size=1,
                use_horovod=False
            )
    return loss
