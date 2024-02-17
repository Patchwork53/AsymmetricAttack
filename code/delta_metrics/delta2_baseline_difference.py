import torch
import os
from transformers import CLIPTokenizer, CLIPTextModel
import argparse
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--csv_file", type=str)
parser.add_argument("--model_name", type=str, default="stabilityai/stable-diffusion-2-1-base")

args = parser.parse_args()

df = pd.read_csv(args.csv_file)


device = 'cuda'

clip_tokenizer = CLIPTokenizer.from_pretrained('stabilityai/stable-diffusion-2-1-base', subfolder='tokenizer')
clip_model = CLIPTextModel.from_pretrained('stabilityai/stable-diffusion-2-1-base', subfolder="text_encoder")
clip_tokenizer.pad_token = clip_tokenizer.eos_token
clip_model.to(device)
cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)



def longest_common_subsequence(X, Y):
  
    m = len(X)
    n = len(Y)
    L = [[None] * (n + 1) for i in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])


    index = L[m][n]

    lcs = [""] * (index + 1)
    lcs[index] = "" 


    i = m
    j = n
    while i > 0 and j > 0:
        if X[i - 1] == Y[j - 1]:
            lcs[index - 1] = X[i - 1]
            i -= 1
            j -= 1
            index -= 1

        elif L[i - 1][j] > L[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return lcs[:-1]


def find_difference3(str1, str2):
    words1 = str1.lower().split()
    words2 = str2.lower().split()
    words1 = ["a" if w in ["an","a"] else w for w in words1]
    words2 = ["a" if w in ["an","a"] else w for w in words2]

    if len(words1) > len(words2):
        words1, words2 = words2, words1

    lcs = longest_common_subsequence(words1, words2)
    iter_lcs = iter(lcs)
    iter_words1 = iter(words1)
    iter_words2 = iter(words2)

    current_lcs_word = next(iter_lcs, None)

    result = []
    for word1 in iter_words1:
        word2 = next(iter_words2, None)
        if word1 == current_lcs_word:
            result.append(word1)
            current_lcs_word = next(iter_lcs, None)
        elif word2 is None or word1 != word2:
            result.append("!")
        else:
            result.append(word2)

    for word2 in iter_words2:
        if word2 == current_lcs_word:
            result.append(word2)
            current_lcs_word = next(iter_lcs, None)
        else:
            result.append("!")

    return ' '.join(result)
    


def clip_cos_sim(sent1, sent2):
    MAX_LENGTH = 77
    t1 = clip_tokenizer(sent1,return_tensors="pt", padding="max_length",max_length=MAX_LENGTH, truncation=True)["input_ids"].to(device)
    t2 = clip_tokenizer(sent2,return_tensors="pt", padding="max_length",max_length=MAX_LENGTH, truncation=True)["input_ids"].to(device)
    e1 = clip_model(t1)[0][0].view(-1)
    e2 = clip_model(t2)[0][0].view(-1)
    
    return cos(e1,e2).item()


for i, row in df.iterrows():
    baseline_sen = find_difference3(row["input_text"], row["target_text"])
    df.loc[i, "baseline_sen"] = baseline_sen
    df.loc[i, "baseline_diff"] = clip_cos_sim(row["target_text"], baseline_sen) - clip_cos_sim(row["input_text"], baseline_sen)


df.to_csv(args.csv_file, index=False)
