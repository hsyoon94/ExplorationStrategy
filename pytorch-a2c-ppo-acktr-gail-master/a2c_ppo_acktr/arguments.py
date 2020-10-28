import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--algo', default='a2c', help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument(
        '--gail-algo', default='standard', help='algorithm to use: standard | wasserstein | none')
    parser.add_argument(
        '--expert-algo', default='ikostrikov', help='expert demos algorithm to use: a2c | ppo | acktr | ikostrikov')
    parser.add_argument(
        '--pretrain-algo', default='none', help='pre-training algorithm to use: cvae | bc | none')
    parser.add_argument(
        '--load-algo', default='a2c', help='algorithm to use: a2c | ppo | acktr | ikostrikov')
    parser.add_argument(
        '--test-model', default='trained', help='algorithm to use: trained | pretrained')
    parser.add_argument(
        '--save-result',
        action='store_true',
        default=False,
        help='save results')
    parser.add_argument(
        '--render',
        action='store_true',
        default=False,
        help='render simulator')
    parser.add_argument(
        '--load-model',
        action='store_true',
        default=False,
        help='load trained model and obs_rms')
    parser.add_argument(
        '--gail',
        action='store_true',
        default=False,
        help='do imitation learning with gail')
    parser.add_argument(
        '--gail-batch-size',
        type=int,
        default=64,
        help='gail batch size (default: 64)')
    parser.add_argument(
        '--gail-epoch', type=int, default=5, help='gail epochs (default: 5)')
    parser.add_argument(
        '--icm',
        action='store_true',
        default=False,
        help='do imitation learning with icm')
    parser.add_argument(
        '--icm-batch-size',
        type=int,
        default=32,
        help='icm batch size (default: 32)')
    parser.add_argument(
        "--latent-dim",
        action="store",
        type=int,
        default=4,
        help="dimension of latent space")
    parser.add_argument(
        '--controller-coef',
        type=float,
        default=2.5,
        help='controller coefficient (default: 0.5)')
    parser.add_argument(
        '--intr-coef',
        type=float,
        default=0.1,
        help='icm coefficient (default: 0.1)')
    parser.add_argument(
        '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=False,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--plt-entropy-coef',
        type=float,
        default=0.005,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--plr-entropy-coef',
        type=float,
        default=0.02,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--plt-value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--plr-value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-processes',
        type=int,
        default=1,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=5,
        help='number of forward steps in A2C (default: 5)')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=4,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--icm-epoch',
        type=int,
        default=4,
        help='number of icm epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=32,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--clip-param1',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--clip-param2',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--vis-interval',
        type=int,
        default=100,
        help='visualization interval, one save per n updates (default: 10)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=25,
        help='save interval, one save per n updates (default: 50)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=1,
        # default=None,
        help='eval interval, one eval per n updates (default: 1)')
    parser.add_argument(
        '--save-episode',
        type=int,
        default=100,
        help='eval episode, n episodes per one save (default: 100)')
    parser.add_argument(
        '--eval-episode',
        type=int,
        default=10,
        # default=None,
        help='eval episode, n episodes per one eval (default: 10)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=10e6,
        help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--env-name',
        default='PongNoFrameskip-v4',
        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument(
        '--log-dir',
        # default='/tmp/gym/',
        default='./logs/',
        help='directory to save learning logs (default: /tmp/gym)')
    parser.add_argument(
        '--pre-log-dir',
        # default='/tmp/gym/',
        default='./pre_logs/',
        help='directory to save learning logs (default: /tmp/gym)')
    parser.add_argument(
        '--result-dir',
        # default='/tmp/gym/',
        default='./results/',
        help='directory to save result logs (default: /tmp/gym)')
    parser.add_argument(
        '--pre-result-dir',
        # default='/tmp/gym/',
        default='./pre_results/',
        help='directory to save result logs (default: /tmp/gym)')
    parser.add_argument(
        '--save-dir',
        default='./trained_models/',
        help='directory to save agent models (default: ./trained_models/)')
    parser.add_argument(
        '--experts-dir',
        default='./expert_models/',
        help='directory to save agent models (default: ./trained_models/)')
    parser.add_argument(
        '--gail-experts-dir',
        default='./gail_experts',
        help='directory that contains expert demonstrations for gail')
    parser.add_argument(
        '--load-dir',
        default='./trained_models/',
        help='directory to load trained models (default: ./pretrained_models/)')
    parser.add_argument(
        '--pre-load-dir',
        default='./pretrained_models/',
        help='directory to load pretrained models (default: ./pretrained_models/)')
    parser.add_argument(
        '--pretrain-dir',
        default='./pretrained_models/',
        help='directory to save pretrained models (default: ./pretrained_models/)')
    parser.add_argument(
        '--load-date',
        default='191025',
        help='pt file name for saved agent model')
    parser.add_argument(
        '--save-date',
        default='190909',
        help='pt file name for saved agent model')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--recurrent-policy',
        action='store_true',
        default=False,
        help='use a recurrent policy')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')
    parser.add_argument(
        '--egreedy',
        action='store_true',
        default=False,
        help='egreedy or not')

    parser.add_argument(
        '--extr-reward-weight',
        type=float,
        default=1.0,
        help='weight of extrinsic reward')
    parser.add_argument(
        '--expert-reward-weight',
        type=float,
        default=1.0,
        help='weight of expert reward')
    parser.add_argument(
        '--favor-zero-expert-reward',
        action='store_true',
        default=False,
        help='weight of extrinsic reward')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.algo in ['a2c', 'ppo', 'acktr']
    assert args.gail_algo in ['standard', 'wasserstein']

    if args.recurrent_policy:
        assert args.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'

    return args
