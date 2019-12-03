#!/usr/bin/env python3
# Evaluate tokenised results based on F1 score (character-based)

import numpy as np
import pandas as pd
import sys

# Calculate TP, FP and FN
def getMeasures(df):
    df['tp'] = np.where((df['system'] == 'T') & (df['gold'] == 'T'), 1, 0)
    df['fp'] = np.where((df['system'] == 'T') & (df['gold'] == 'O'), 1, 0)
    df['fn'] = np.where((df['system'] == 'O') & (df['gold'] == 'T'), 1, 0)
    tp = df.tp.sum()
    fp = df.fp.sum()
    fn = df.fn.sum()
    return tp, fp, fn

# Calculate F1 score
def getF1(tp, fp, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2*((precision*recall)/(precision+recall))
    return f1

# Mark start of tokens
def markT(string):
    marked = []
    skip = False
    for idx, char in enumerate(string):
        if (idx == 0):
            marked.append((char,"T"))
        elif (char == " "):
            skip = True
        elif (skip):
            marked.append((string[idx],"T"))
            skip = False
        else:
            marked.append((char,"O"))
    return marked

def main():
    str_sys = sys.stdin.read()

    with open('file.tok.gold') as f:
        str_gold = f.read()

    str_sys = str_sys.replace("\n", "")
    str_gold = str_gold.replace("\n", "")
    str_gold = str_gold.replace("-LRB-","(").replace("-RRB-",")").replace("-LSB-","[").replace("-RSB-","]").replace("-LCB-","{").replace("-RCB-","}")

    marked_sys = markT(str_sys)
    marked_gold = markT(str_gold)

    if (len(marked_sys) == len(marked_gold)):
        pass
    else:
        raise ValueError("The numbers of tokens don't match.")

    df = pd.DataFrame({"tokens": [i[0] for i in marked_sys],
                       "system": [i[1] for i in marked_sys],
                       "gold": [i[1] for i in marked_gold]})

    tp, fp, fn = getMeasures(df)
    f1 = getF1(tp, fp, fn)
    print(f1)

if __name__ == "__main__":
    main()