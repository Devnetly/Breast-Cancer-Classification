max=10
for i in `seq 1 $max`
do
    python3 attention.py --model=ABNN --tensors=$1 --weights-folder=$2 --histories-folder=$3 --epochs=10 --filters-in=2048 --filters-out=64 
    sleep 3600
done