import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--data_dir', type=str, default='datasets')
    parser.add_argument('--source_data', type=str, default='./datasets/source_data.csv')
    parser.add_argument('--source_label', type=str, default='./datasets/source_label.csv')
    parser.add_argument('--source_adj', type=str, default='./datasets/source_adj.csv')

    parser.add_argument('--target_data', type=str, default='./datasets/target_data.csv')
    parser.add_argument('--target_label', type=str, default='./datasets/target_label.csv')
    parser.add_argument('--target_adj', type=str, default='./datasets/target_adj.csv')
    
    # model
    parser.add_argument('--modelname', type=str, default='tf_model')
    parser.add_argument('--input_dim', type=int, default=6992)
    parser.add_argument('--conv', type=str, default='TransformerConv') # GCNConv; GATConv
    parser.add_argument('--NUM_HIDDENS', type=str, default='512,64')
    parser.add_argument('--num_class', type=int, default=2)

    # train
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100) 
    parser.add_argument('--save_path', type=str, default='./checkpoint')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--grad_clip', type=float, default=5)

    # test
    parser.add_argument('--test_save_path', type=str, default='best_f1.pth')
    parser.add_argument('--pred', type=str, default='./results')
    parser.add_argument('--verbose', type=int, default=0)

    # seed
    parser.add_argument('--seed', type=int, default=42)

    # cuda
    parser.add_argument('--use_cuda', type=str,default=False)

    # yml name
    parser.add_argument('--ymlname', type=str, default='configure_default.yml')

    opt = parser.parse_args()

    yml = open(opt.ymlname, 'w')
    yml.write('Name: %s\n'%(opt.data_dir))
    
    yml.write('DATASET:\n')
    yml.write('  Source_data_root: %s\n'%(opt.source_data))
    yml.write('  Source_label_root: %s\n'%(opt.source_label))
    yml.write('  Source_adj_root: %s\n'%(opt.source_adj))

    yml.write('  target_data_root: %s\n'%(opt.target_data))
    yml.write('  target_label_root: %s\n'%(opt.target_label))
    yml.write('  target_adj_root: %s\n'%(opt.target_adj))

    yml.write('MODEL:\n')
    yml.write('  NAME: %s\n'%(opt.modelname))
    yml.write('  INPUT_DIM: %d\n'%(opt.input_dim))
    yml.write('  Conv: %s\n'%(opt.conv))

    feature_extractor = opt.NUM_HIDDENS.split(',')
    name = '_'.join(feature_extractor)

    feature_extractor = [int(x) for x in feature_extractor]
    yml.write('  NUM_HIDDENS: [%d'%(feature_extractor[0]))
    for d in feature_extractor[1:]:
        yml.write(',%d'%(d))
    yml.write(']\n')

    yml.write('  num_classes: %s\n'%(opt.num_class))

    yml.write('TRAIN:\n')
    yml.write('  lr: %f\n'%(opt.lr))
    yml.write('  Epochs: %d\n'%(opt.epochs))
    yml.write('  Save_path: %s\n'%(opt.save_path))
    yml.write('  Momentum: %f\n'%(opt.momentum))
    yml.write('  Weight_decay: %f\n'%(opt.weight_decay))
    yml.write('  grad_clip: %d\n'%(opt.grad_clip))

    yml.write('TEST:\n')
    yml.write('  Save_path: %s\n'%(opt.test_save_path))
    yml.write('  Pred: %s\n'%(opt.pred))
    yml.write('  Verbose: %s\n'%(opt.verbose))

    yml.write('SEED: %d\n'%(opt.seed))
    yml.write('Use_CUDA: %s\n'%(opt.use_cuda))


if __name__ == '__main__':
    main()
