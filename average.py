import csv
import os

import numpy as np


def walklevel(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]


root = os.path.expanduser('~/Desktop/openai')
prefix = 'openai-2018-03-18-00-58'
dirs = walklevel(root, level=2)
best = []
for root1, dir1, file1 in dirs:
    for root2, dir2, file2 in dirs:
        if os.path.basename(root2).startswith(prefix):
            print(root2)
            rewards = []
            length = []
            with open(os.path.join(root2, 'log.txt'), 'r') as f:
                print(f.readlines()[1])
            data = csv.reader(open(os.path.join(root2, 'progress.csv')))
            try:
                fields = data.next()
                for row in data:
                    items = zip(fields, row)
                    item = {}
                    for (name, value) in items:
                        item[name] = value.strip()
                    rewards.append(float(item['EpRewMean']))
                    length.append(float(item['TimestepsSoFar']))
                best.append(max(rewards))
                print(max(length))
                assert max(length) > 3000000
            except:
                pass

print(best)
print(np.mean(best))
