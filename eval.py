import argparse
from train_and_test import test_single_class

#         test_single_class(test_path=data, save_path=save_path, model_path=model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--save_path', default='./save', type=str, help='Directory to save results, which will be '
                                                                        'saved in acc_classes.txt')
    parser.add_argument('--eval_path', default='./data/NWPU_RSOR', type=str, help='Data to evaluate the model')
    # parser.add_argument('--nwpu_path', default='./data/NWPU_RSOR', type=str)
    # parser.add_argument('--hrrsd_path', default='./data/HRRSD_RSOR/test', type=str)
    # parser.add_argument('--dota_path', default='./data/DOTA_RSOR/test', type=str)
    # parser.add_argument('--dior_path', default='./data/DIOR_RSOR/test', type=str)
    parser.add_argument('--model_path', default='./Ready Model/ResNet-50-TSCM.pth', type=str, help='Model to evaluate.')
    args = parser.parse_args()
    print("Eval data from {} with Model {}, Save results to {}.\n".format(args.eval_path, args.model_path,
                                                                          args.save_path))
    test_single_class(test_path=args.eval_path, save_path=args.save_path, model_path=args.model_path)
