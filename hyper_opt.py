# brute-force random search for hyper-parameter optimization
import os
import numpy as np

from config import args as FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

def args2str(args):
    ret = ''
    key_val = sorted([(key, val) for key, val in args.items()])
    for key, val in key_val:
        ret += '--%s=%s ' % (key, str(val))
    return ret.strip()

def _run_model(args_str):
    cmd = 'nohup python3 -u main.py --epoch=15 --gpu=%s %s > %s' % (FLAGS.gpu, args_str, _get_log_file(args_str))
    print('Command: %s' % cmd)
    os.system(cmd)

key2vals = {}
tried_args = set()

def _prepare_args():
    global key2vals
    key2vals['hidden_size'] = [64, 96, 128]
    key2vals['dropout_emb'] = [0.4, 0.5]
    key2vals['dropout_rnn_output'] = [0.2, 0.4, 0.5]
    key2vals['finetune_topk'] = [10, 50, 100]

_prepare_args()

def _get_random_args():
    global key2vals
    ret = {}
    for key in key2vals:
        vals = key2vals[key]
        idx = np.random.randint(0, len(vals))
        random_val = vals[idx]
        ret[key] = random_val
    return ret

def _get_log_file(args_str):
    suffix = args_str.replace(' ', '_').replace('-', '')
    return ('./hyper/log-%s.txt' % suffix)

def random_search():
    global tried_args
    while len(tried_args) < 15:
        args = _get_random_args()
        args_str = args2str(args)
        if args_str in tried_args or os.path.exists(_get_log_file(args_str)):
            continue
        tried_args.add(args_str)
        print('Iteration %d...' % len(tried_args))
        _run_model(args_str)

def check_hyper_logs(path='./hyper/'):
    for f in os.listdir(path):
        print(f)
        os.system('cat %s/%s | grep Dev | tail -n 5' % (path, f))

if __name__ == '__main__':
    # check_hyper_logs()
    os.makedirs('./hyper', exist_ok=True)
    random_search()
