python train.py \
    --in-dataset CIFAR-10 \
    --id_loc /datasets/CIFAR10 \
    --gpu 0 \
    --seed 10 \
    --model resnet18 \
    --batch-size 512 \
    --epochs 500 \
    --proto_m 0.9 \
    --w 0.1 \
    --arcface \
    --margins 0.3