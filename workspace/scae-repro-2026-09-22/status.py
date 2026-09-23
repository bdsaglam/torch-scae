"""Prints the latest step and metrics of every run under runs/."""
import glob, os, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')
keys = ['val_accuracy', 'val_loss', 'train/rec_ll_loss', 'train/log_prob_loss']
for run in sorted(glob.glob(os.path.join(root, '*/'))):
    evs = glob.glob(os.path.join(run, 'lightning_logs/*/events.*'))
    if not evs:
        print(os.path.basename(run[:-1]), 'no events yet'); continue
    ea = EventAccumulator(os.path.dirname(sorted(evs)[-1]), size_guidance={'scalars': 0}).Reload()
    tags = ea.Tags()['scalars']
    out = []
    for k in keys:
        if k in tags:
            s = ea.Scalars(k)[-1]
            out.append(f'{k}={s.value:.4g}@{s.step}')
    print(os.path.basename(run[:-1]), ' '.join(out))
