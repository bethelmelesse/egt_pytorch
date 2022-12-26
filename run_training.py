import sys
from lib.training.execute import get_configs_from_args, execute

if __name__ == '__main__':
    # args = sys.argv
    # args.append('configs/molhiv/egt_110m.yaml')
    config = get_configs_from_args(sys.argv)
    execute('train', config)
