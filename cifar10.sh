for i in "SGD MSE 20 False" "SGD CrossEntropy 20 False" "Adam MSE 20 False" "Adam CrossEntropy 20 False" "SGD MSE 20 True" "SGD CrossEntropy 20 True" "Adam MSE 20 True" "Adam CrossEntropy 20 True"
do
	set -- $i
	export CUDA_VISIBLE_DEVICES=2,3 && python cifar10.py --optim $1 --loss $2 --epochs $3 --train_scheme $4 
done
