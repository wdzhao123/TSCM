import argparse
from train_and_test import train

# train(alpha=0.0, beta=0.0, gamma=0.0, save_path='',
#       train_path='', test_path='',
#       epochs=0,
#       initial_resnet_path=''
#       )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--alpha', default=0.1, type=float, help='Alpha super-parameter which controls Inter-class '
                                                                 'Dispersion Metric')
    parser.add_argument('--gamma', default=0.5, type=float, help='Beta super-parameter which controls Intra-class '
                                                                 'Interaction Metric')
    parser.add_argument('--beta', default=0.5, type=float, help='Gamma super-parameter which controls Intra-class '
                                                                'Compactness Metric')
    parser.add_argument('--save_path', default='./save', type=str, help='Directory to save trained model')
    parser.add_argument('--train_path', default='./data/DOTA_RSOR/train', type=str, help='Data to train the model')
    parser.add_argument('--test_path', default='./data/DOTA_RSOR/test', type=str, help='Data to evaluate the model')
    parser.add_argument('--epochs', default='48', type=int, help='How many epochs to train')
    parser.add_argument('--initial_resnet_path', default='./Ready Model/resnet50-pretrained.pth', type=str,
                        help='Initial model parameters')
    args = parser.parse_args()
    print(f"Super parameters: "
          f"Alpha {args.alpha}, Beta {args.beta}, Gamma {args.gamma}.\n"
          f"Save trained model to {args.save_path}.\n"
          f"Train data from {args.train_path}, Eval data from {args.test_path}.\n"
          f"Plan to train {args.epochs} epochs.\n"
          f"Initial model parameters {args.initial_resnet_path}.")
    train(alpha=args.alpha, beta=args.beta, gamma=args.gamma, save_path=args.save_path,
          train_path=args.train_path, test_path=args.test_path,
          epochs=args.epochs,
          initial_resnet_path=args.initial_resnet_path)
