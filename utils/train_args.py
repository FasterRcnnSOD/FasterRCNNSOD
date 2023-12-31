Cuda = True
seed = 12345
train_gpu = [0, ]
fp16 = False
input_shape = [784, 784]
backbone = "swinT"
pretrained = False
anchors_size = [8, 16, 32]

Init_Epoch = 0
Freeze_Epoch = 5
Freeze_batch_size = 8
UnFreeze_Epoch = 500
Unfreeze_batch_size = 4
Freeze_Train = True

Init_lr = 1e-4
Min_lr = Init_lr * 0.01
optimizer_type = "adam"
momentum = 0.9
weight_decay = 0
lr_decay_type = 'cos'
save_period = 1e6
save_dir = 'logs/Visdrone'
eval_flag = True
eval_period = 5
num_workers = 8

model_path = 'logs/loss_2023_10_16_19_53_03_swinT/best_epoch_weights.pth'
classes_path = 'datasets/Visdrone/num_classes.txt'
train_annotation_path = 'datasets/Visdrone/train.txt'
val_annotation_path = 'datasets/Visdrone/val.txt'