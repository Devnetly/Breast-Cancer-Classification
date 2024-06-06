#max=1
#for i in `seq 1 $max`
#do
#    python3 attention.py --model=ABNN --tensors="/media/bracs/Extreme SSD/tensors" --weights-folder="/home/bracs/Breast-Cancer-Detection/models/attention1_resnet50" --histories-folder="/home/bracs/Breast-Cancer-Detection/histories/attention1_resnet50" --epochs=10 --filters-in=2048 --filters-out=64 --use-lr-decay=False
#    sleep 3600
#done

#python3 attention.py --model=ABNN --tensors="/media/bracs/Extreme SSD/tensors" --weights-folder="/home/bracs/Breast-Cancer-Detection/models/attention1_resnet50" --histories-folder="/home/bracs/Breast-Cancer-Detection/histories/attention1_resnet50" --epochs=10 --filters-in=2048 --filters-out=64 --use-lr-decay=False

python3 attention.py --model=ACMIL --tensors="/media/bracs/Extreme SSD/tensors-att2" --weights-folder="/home/bracs/Breast-Cancer-Detection/models/attention2_resnet50" --histories-folder="/home/bracs/Breast-Cancer-Detection/histories/attention2_resnet50" --epochs=100 --features=resnet50 --use-lr-decay=False
