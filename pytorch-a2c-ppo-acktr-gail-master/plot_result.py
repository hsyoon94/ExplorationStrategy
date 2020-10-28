import os
import argparse

from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas
import numpy

from a2c_ppo_acktr import utils
from a2c_ppo_acktr.arguments import get_args


#python plot.py --exps HalfCheetah_q3_action128 HalfCheetah_q3_action4096 \
#HalfCheetah_q3_action16384 --save HalfCheetah_q3_actions

parser = argparse.ArgumentParser()
parser.add_argument('--exps',  nargs='+', type=str)
parser.add_argument('--save', type=str, default=None)
parser.add_argument('--env-name', default='MountainOldCarContinuous-v1',
                    help='environment to train on (default: PongNoFrameskip-v4)')

args = parser.parse_args()


def millions(x, pos):
    'The two args are the value and tick position'
    return '%1.2fM' % (x*1e-6)


def thousands(x, pos):
    'The two args are the value and tick position'
    return '%1.1fK' % (x*1e-3)


# formatter = FuncFormatter(thousands)
formatter = FuncFormatter(millions)

f, ax = plt.subplots(1, 1)
ax.xaxis.set_major_formatter(formatter)

ax.patch.set_facecolor('lavender')
ax.set_facecolor((234/256, 234/256, 243/256))
# ax.patch.set_alpha(0.5)

for i, exp in enumerate(args.exps):
    log_fname = os.path.join('graphs', args.env_name, 'data', exp + '.csv')
    csv = pandas.read_csv(log_fname)

    # color = cm.viridis(i / float(len(args.exps)))
    # colors = ['crimson', 'orchid', 'orange','teal']
    colors = ['steelblue', 'peru', 'olivedrab', 'firebrick', 'darkolivegreen', 'darkblue', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'orange',
              'darkgreen', 'tan', 'salmon', 'gold', 'darkred', 'turquoise',
              'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',]

    ax.plot(csv['Timesteps'], csv['ReturnAvg'], color=colors[i], label=exp, linewidth='2.0')
    ax.fill_between(csv['Timesteps'], csv['ReturnAvg'] - csv['ReturnStd'], csv['ReturnAvg'] + csv['ReturnStd'],
                    color=colors[i], alpha=0.2)

ax.legend()
ax.set_xlabel('Number of training steps')
ax.set_ylabel('Average Return')
ax.margins(0.0, 0.05)
ax.set_axisbelow(True)

ax.grid(linestyle='-', linewidth='1.0', color='white')

if args.save:
    os.makedirs('plots', exist_ok=True)
    # f.savefig(os.path.join('plots', args.save + '.jpg'))
    f.savefig(os.path.join('graphs', args.env_name, 'data', exp + '.pdf'))
else:
    plt.show()
