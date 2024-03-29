#!bin/bash

name='datasets'
src_data=$name'/source_data.csv'
src_lab=$name'/source_label.csv'
src_adj=$name'/source_adj.csv'
tar_data=$name'/target_data.csv'
tar_lab=$name'/target_label.csv'
tar_adj=$name'/target_adj.csv'
conv='TransformerConv'

mkdir 'checkpoint'
mkdir 'results'
savemdir='./checkpoint'
savemodel='./checkpoint/best_f1.pth'
resdir='./results'

/Users/qianqian/.pyenv/versions/3.7.9/bin/python  configuration.py \
	 --data_dir   $name \
	 --source_data  $src_data \
	 --source_label $src_lab \
	 --source_adj   $src_adj \
	 --target_data  $tar_data \
	 --target_label $tar_lab \
	 --target_adj   $tar_adj \
	 --input_dim 2000 \
	 --conv $conv \
	 --epochs 200 \
	 --save_path  $savemdir \
	 --test_save_path  $savemodel \
	 --pred  $resdir 

/Users/qianqian/.pyenv/versions/3.7.9/bin/python  main_func.py
