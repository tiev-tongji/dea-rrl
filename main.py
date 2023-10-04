from utils import long_num_to_string
from experiment import Experimemt
import argparse
import os
import random

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')

# experiment config
parser.add_argument('--env-name', default="Safexp-PointGoal1-v0",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--seed', type=int, default=None, metavar='N',
                    help='random seed (default: None)')
parser.add_argument('--cuda', type=int, default=None,
                    help='run on CUDA (default: False)')
parser.add_argument('--single-core', default=False, action='store_true')
parser.add_argument('--batch-size-run', type=int, default=10,
                    help='batch size while running')
parser.add_argument('--batch-size-test', type=int, default=4)
parser.add_argument('--start-cpuid', type=int, default=1)
parser.add_argument('--experiment-name', type=str, default=None)
parser.add_argument('--not-use-summary-writer',
                    default=False, action='store_true')
parser.add_argument('--pretrain-suffix', type=str, default=None)
parser.add_argument('--experiment-suffix', type=str, default=None)
parser.add_argument('--save-expert-data', default=False, action='store_true')
parser.add_argument('--save-expert-data-path',
                    type=str, default='expert_data')
parser.add_argument('--safety-gym', default=False, action='store_true')
parser.add_argument('--logdir', type=str, default='runs')
parser.add_argument('--test-only', default=False, action='store_true')
parser.add_argument('--test-recovery', default=False, action='store_true')
parser.add_argument('--save-plt', default=False, action='store_true')
parser.add_argument('--final-test-times', type=int, default=100)

# SAC config
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--hidden-size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 1000000)')
parser.add_argument('--save-model-path', type=str, default='./checkpoints',
                    help='save model path')
parser.add_argument('--load-model-path', type=str, default=None,
                    help='load model path')

# SAC training config
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--target-update-interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')

# global training config
parser.add_argument('--not-train', default=False, action='store_true',
                    help='training mode (default: False)')
parser.add_argument('--num-epochs', type=int, default=2, metavar='N')
parser.add_argument('--num-steps', type=int, default=1000000, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--start-steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--updates-per-step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')

''' Data Config '''
# data config
parser.add_argument('--load-data-path', type=str, default=None,
                    help='load offline data path')
parser.add_argument('--pretrain-data-size', type=int, default=1000000,
                    help='sample pretrain offline data size (default: 1000000)')

# sample data
parser.add_argument('--sample-data-only', default=False, action='store_true')
parser.add_argument('--sample-last-num-steps', type=int, default=0,
                    help='only sample steps before failed')
parser.add_argument('--save-data-path', type=str, default='./data',
                    help='save offline data path')
parser.add_argument('--data-name', type=str, default=None)

# not use
parser.add_argument('--sample-data', default=False, action='store_true')
parser.add_argument('--sample-by-policy', default=False, action='store_true')


''' Baseline Config '''
# LR
parser.add_argument('--use-lr', default=False, action='store_true')
parser.add_argument('--nu', type=float, default=0.01)

# SQRL
parser.add_argument('--use-sqrl', default=False, action='store_true')
parser.add_argument('--safe-samples', type=int, default=100)

# RSPO
parser.add_argument('--use-rspo', default=False, action='store_true')
parser.add_argument('--nu-start', type=float, default=1e3)
parser.add_argument('--nu-end', type=int, default=0)

# RCPO
parser.add_argument('--use-rcpo', default=False, action='store_true')
parser.add_argument('--rcpo_lambda', type=float, default=0.01)

# WCSAC
parser.add_argument('--use-wcsac', default=False, action='store_true')
parser.add_argument('--use-iqn-recovery', default=False, action='store_true')
parser.add_argument('--num-quantiles-recovery', type=int, default=32)
parser.add_argument('--tau-type-recovery', type=str, default='iqn')
parser.add_argument('--risk-type-recovery', type=str, default='ncvar')
parser.add_argument('--risk-param-recovery', type=float, default=0.3)

# RP
parser.add_argument('--use-rp', default=False, action='store_true')
parser.add_argument('--rp-lambda', type=float, default=100.)

# DSAC
parser.add_argument('--use-dsac', default=False, action='store_true')
parser.add_argument('--num-quantiles', type=int, default=32)
parser.add_argument('--tau-type', type=str, default='iqn')
parser.add_argument('--risk-type', type=str, default='cvar')
parser.add_argument('--risk-param', type=float, default=0.3)
parser.add_argument('--adaptive-risk', type=str, default='disable')

# IQL
parser.add_argument('--use-iql', default=False, action='store_true')
parser.add_argument('--iql-quantile', type=float, default=0.7)
parser.add_argument('--clip-score', type=float, default=1.)
parser.add_argument('--iql-beta', type=float, default=10.)

# DEA RRL
parser.add_argument('--use-dea-rrl', default=False, action='store_true')

# ORI RRL
parser.add_argument('--use-ori-rrl', default=False, action='store_true')

# COM RRL (composite rrl)
parser.add_argument('--use-com-rrl', default=False, action='store_true')

''' Safety Critic Config '''
parser.add_argument('--use-safety-critic', default=False, action='store_true')
parser.add_argument('--pos-fraction', type=float, default=0.5)
parser.add_argument('--updates-per-step-safety-critic', type=int, default=1)
parser.add_argument('--not-use-double-q', default=False, action='store_true')

''' Recovery Config '''
parser.add_argument('--use-recovery', default=False, action='store_true',
                    help='use safety critic and recovery policy')
parser.add_argument('--dea-recovery', default=False, action='store_true')
parser.add_argument('--recovery-policy', default="Gaussian",
                    help='Recovery Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--gamma-safe', type=float, default=0.99,
                    help='discount factor for cost (default: 0.99)')

# recovery explore
parser.add_argument('--recovery-not-explore',
                    default=False, action='store_true')

# iql for recovery
parser.add_argument('--use-iql-recovery', default=False, action='store_true')
parser.add_argument('--iql-quantile-recovery', type=float, default=0.3)
parser.add_argument('--clip-score-recovery', type=float, default=None)
parser.add_argument('--iql-beta-recovery', type=float, default=10.)
parser.add_argument('--iql-awr-recovery', default=False, action='store_true')

# pretrain config
parser.add_argument('--not-pretrain', default=False, action='store_true')
parser.add_argument('--not-pretrain-recovery', default=False, action='store_true',
                    help='pretrain model')
parser.add_argument('--pretrain-steps-recovery', type=int, default=500000,
                    help='pretrain steps (default: 500000)')
parser.add_argument('--save-pretrain-model-path', type=str, default='./pretrain_models',
                    help='save pretrain model path')
parser.add_argument('--load-pretrain-model-path', type=str, default=None,
                    help='load pretrain model path')
parser.add_argument('--pretrain-model-name', type=str, default=None)


# finetune config
parser.add_argument('--not-finetune', default=False, action='store_true')
parser.add_argument('--eps-safe', type=float, default=0.3,
                    help='safe threshold (default: 0.3)')
parser.add_argument('--updates-per-step-recovery', type=int, default=1, metavar='N',
                    help='model updates per simulator step recovery (default: 1)')

# add noise
parser.add_argument('--add-noise-on-state-during-pretraining', default=False, action='store_true')
parser.add_argument('--add-noise-on-state-during-training'   , default=False, action='store_true')
parser.add_argument('--add-noise-on-state-during-testing'    , default=False, action='store_true')

parser.add_argument('--noise-std-on-state-during-pretraining', default=0.1, type=float)
parser.add_argument('--noise-std-on-state-during-training'   , default=0.1, type=float)
parser.add_argument('--noise-std-on-state-during-testing'    , default=0.1, type=float)

parser.add_argument('--add-noise-on-action-during-training'  , default=False, action='store_true')
parser.add_argument('--add-noise-on-action-during-testing'   , default=False, action='store_true')

parser.add_argument('--noise-std-on-action-during-training'   , default=0.1, type=float)
parser.add_argument('--noise-std-on-action-during-testing'    , default=0.1, type=float)


''' Test Config '''
parser.add_argument('--not-test', default=False, action='store_true')

''' End '''

if __name__ == '__main__':
    args = parser.parse_args()
    
    if args.env_name == 'StaticEnv-v0':
        args.num_steps = 200000
    if args.env_name == 'DynamicEnv-v0':
        args.num_steps = 500000
        
    args.safety_gym = True
    args.recovery_not_explore = True
        
    if args.seed == None:
        args.seed = random.randint(0, 1000000)

    if args.use_ori_rrl:
        args.use_recovery = True
    
    if args.use_dea_rrl:
        args.use_recovery = True
        args.use_iql_recovery = True
        args.not_finetune = True

    if args.use_com_rrl:
        args.use_recovery = True
        args.com_recovery = True
    else:
        args.com_recovery = False

    # check if safety critic is needed
    if args.use_wcsac:
        args.use_lr = True
        args.ise_dsac = True
        args.use_iqn_recovery = True

    if args.use_iqn_recovery:
        args.use_safety_critic = True
        args.safety_critic_network_type = 'IQN'
    elif args.use_recovery or args.use_lr or args.use_sqrl or args.use_rspo or args.use_rcpo or args.adaptive_risk == 'qrisk':
        args.use_safety_critic = True
        args.safety_critic_network_type = 'DQN'
    if args.use_safety_critic:
        args.safety_critic_network_num = 'Single' if args.not_use_double_q else 'Double'
        args.updates_per_step_safety_critic = args.updates_per_step_recovery if args.use_recovery else args.updates_per_step

    if args.sample_data_only == True:
        args.sample_data = True
        args.not_train = True
        args.not_pretrain = True
        args.not_test = True
        args.not_use_summary_writer = True
        args.experiment_name = 'sample_data'
        args.batch_size_run = 20
        args.batch_size_test = 1
        if args.data_name is None:
            print('No Data Name, exit ...')
            exit()

    if args.test_only == True:
        args.experiment_name = 'test'
        args.not_train = True
        args.not_pretrain = True
        args.not_use_summary_writer = True
        args.batch_size_run = 1
        args.batch_size_test = 20

    print(args.experiment_name)

    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)
    if not os.path.exists(args.save_pretrain_model_path):
        os.makedirs(args.save_pretrain_model_path)
    if not os.path.exists(args.save_data_path):
        os.makedirs(args.save_data_path)
    if not os.path.exists(args.save_expert_data_path):
        os.makedirs(args.save_expert_data_path)

    args.save_model_path = os.path.join(args.save_model_path, args.env_name)
    args.save_pretrain_model_path = os.path.join(
        args.save_pretrain_model_path, args.env_name)
    args.save_data_path = os.path.join(args.save_data_path, args.env_name)
    args.save_expert_data_path = os.path.join(
        args.save_expert_data_path, args.env_name)

    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)
    if not os.path.exists(args.save_pretrain_model_path):
        os.makedirs(args.save_pretrain_model_path)
    if not os.path.exists(args.save_data_path):
        os.makedirs(args.save_data_path)
    if not os.path.exists(args.save_expert_data_path):
        os.makedirs(args.save_expert_data_path)

    experiment = Experimemt(args)
    experiment.runAll()
