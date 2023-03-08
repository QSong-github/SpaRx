# !bin/bash

name='datasets'
yname='parameter_tune.yml'
src_data=$name'/source_data.csv'
src_lab=$name'/source_label.csv'
tar_data=$name'/target_data.csv'
tar_lab=$name'/target_label.csv'
conv='TransformerConv'

mkdir -p 'checkpoint'
mkdir -p 'results'
savemdir='./checkpoint'
savemodel='best_loss.pth'
resdir='./results'
epoch=100

for k in 2 4 6 8 10
do
    Rscript ./src/data_preprocess.R $src_data $tar_data 'NULL' 'simulation' $k  './datasets'
    src_adj=$name'/source_adj.csv'
    tar_adj=$name'/target_adj.csv'
    echo $src_adj
    echo $tar_adj
    for d1 in 128 256 512 1024
    do
	for d2 in 16 32 64 128
	do
            for momen in 0.8 0.9 0.99 0.999
            do 
		for wd in 0.00001 0.00005 0.0001  0.001 0.01
		do 
                    for gd in 1 3 5 7 9
                    do 
			nhim="$d1,$d2"
			python3  configuration.py \
				 --data_dir   $name \
				 --source_data  $src_data \
				 --source_label $src_lab \
				 --source_adj   $src_adj \
				 --target_data  $tar_data \
				 --target_label $tar_lab \
				 --target_adj   $tar_adj \
				 --input_dim 6922 \
				 --conv $conv \
				 --epochs $epoch \
				 --save_path  $savemdir \
				 --test_save_path  $savemodel \
				 --pred  $resdir \
				 --NUM_HIDDENS $nhim \
				 --ymlname $yname \
				 --momentum $momen \
				 --weight_decay $wd \
				 --grad_clip $gd
			python3  parameter_tune.py
                    done
		done
            done
	done
    done
done
