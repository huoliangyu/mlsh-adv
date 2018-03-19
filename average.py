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
best = []
#
# prefixes = ['openai-2018-03-18-22-37'] # mlp 64
# prefixes = ['openai-2018-03-18-22-42-5'] # mlp 4
# prefixes = ['openai-2018-03-18-23-20-5'] # v9 max 2 LSTM
# prefixes = ['openai-2018-03-19-11-22-2'] # v9 max 3 LSTM
# prefixes = ['openai-2018-03-19-11-21-4'] # v9 max 4 LSTM
# prefixes = ['openai-2018-03-19-11-22-4', 'openai-2018-03-19-11-22-3'] # v9
# max 5 LSTM
# prefixes = ['openai-2018-03-19-11-23-3'] # v9 max 2 GRU
# prefixes = ['openai-2018-03-19-11-26-1'] # v9 max 4 GRU
# prefixes = ['openai-2018-03-19-11-56-4'] # v9 max 2 LSTM - 2
# prefixes = ['openai-2018-03-19-12-59-2'] # v9 max 5 LSTM - 2
# prefixes = ['openai-2018-03-19-12-59-5'] # v9 max 6 LSTM
# prefixes = ['openai-2018-03-19-17-21-1'] # v13 max 2 LSTM
# prefixes = ['openai-2018-03-19-17-22-2'] # v13 max 4 LSTM
prefixes = ['openai-2018-03-19-21-23-5', 'openai-2018-03-19-21-24-0'] # v13
# max 4 LSTM


cutoff = 6000000
# cutoff = 3000000
# cutoff = 1500000

for prefix in prefixes:
    dirs = walklevel(root, level=2)
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
                        if float(item['TimestepsSoFar']) < cutoff:
                            rewards.append(float(item['EpRewMean']))
                            length.append(float(item['TimestepsSoFar']))
                    if max(rewards) not in best:
                        best.append(max(rewards))
                    print(max(length))
                    assert max(length) > 3000000
                except Exception as e:
                    print(e)

print(best)
print(round(np.mean(best), 1))
