#!/usr/bin/env python3
# Create df for results of spd and calculates f1 score

import numpy as np
import pandas as pd
import re
import difflib
import sys

# Concatenate spd results into one string without extra spaces
def normaliseStr(sbd_lines):
    return " ".join([line.strip() for line in sbd_lines])

# Find position of S-SENT in terms of tokens delimited by spaces
def findSSENT(sbd):
    sbd_list = re.split("(\n)", sbd.strip())
    pos = []
    for idx, item in enumerate(sbd_list):
        if(idx%2==0):
            if(idx!=0):
                pos.append(len(item.split())+pos[int((idx-2)/2)])
            else:
                pos.append(len(item.split()))
    
    # Return list of positions but not including the end of the last sentence
    return pos[:-1]

# Mark S-SENT corresponding to idx of token delimited by spaces
def markSSENT(pos, num_tokens):
    np_array = np.full(num_tokens, "O", dtype="U6")
    for i in pos:
        np_array[i] = "S-SENT"
    np_array[0] = "S-SENT"
    return np_array

# Calculate TP, FP and FN
def getMeasures(df):
    df['tp'] = np.where((df['system'] == 'S-SENT') & (df['gold'] == 'S-SENT'), 1, 0)
    df['fp'] = np.where((df['system'] == 'S-SENT') & (df['gold'] == 'O'), 1, 0)
    df['fn'] = np.where((df['system'] == 'O') & (df['gold'] == 'S-SENT'), 1, 0)
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

def main():
    # Read my spd results from cat command
    my_sbd_lines = sys.stdin.readlines()
    my_sbd = "".join(my_sbd_lines)

    # Read gold results from file
    with open('file.sbd.gold') as f:
        gold_sbd_lines = f.readlines()
    gold_sbd = "".join(gold_sbd_lines)

    # Check if number of tokens match to form correct table later
    str_my = normaliseStr(my_sbd_lines)
    str_gold = normaliseStr(gold_sbd_lines)
    if (str_my == str_gold):
        pass
    else:
        raise ValueError("Number of tokens in the two input files don't match.")

    num_tokens = len(str_my.split())
    system = markSSENT(findSSENT(my_sbd), num_tokens)
    gold = markSSENT(findSSENT(gold_sbd), num_tokens)

    df = pd.DataFrame({"tokens": str_my.split(),
                       "system": system,
                       "gold": gold})

    tp, fp, fn = getMeasures(df)
    f1 = getF1(tp, fp, fn)
    print(f1)

if __name__== "__main__":
  main()