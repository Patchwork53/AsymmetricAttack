import os
import shutil
import json
import torch
from diffusers import StableDiffusionPipeline
from typing import Any, Optional, Tuple, Union
import numpy as np
import torch
import random
import torch.nn.functional as F
from tqdm import tqdm
import math
import random
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline
from torch import autocast
from PIL import Image


def get_embeddings(model, input_ids):
  return model.text_model.embeddings(input_ids)

def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):

    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

def get_grad(model, adv_input_tokens, target, avoid, suff_start, suff_end, object_key_mask=None):
  embedding_weight = model.get_input_embeddings().weight

  one_hot = torch.zeros(
    adv_input_tokens[suff_start:suff_end].shape[0],
    embedding_weight.shape[0],
    device=model.device,
    dtype=embedding_weight.dtype
  )

  one_hot.scatter_(
    1,
    adv_input_tokens[suff_start:suff_end].unsqueeze(1),
    torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embedding_weight.dtype)
  )
  one_hot.requires_grad_()

  suffix_embeds = (one_hot @ embedding_weight).unsqueeze(0)
  embeds = get_embeddings(model, adv_input_tokens.unsqueeze(0)).detach()
  new_embeds = torch.cat(
      [
          embeds[:,:suff_start,:],
          suffix_embeds,
          embeds[:,suff_end:,:]
      ],
      dim=1)

  hidden_states = new_embeds
  attention_mask = None

  input_ids = adv_input_tokens.unsqueeze(0)
  input_shape = input_ids.shape

  causal_attention_mask = _make_causal_mask(input_shape, hidden_states.dtype, device=hidden_states.device)

  if attention_mask is not None:
      attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

  encoder_outputs = model.text_model.encoder(
      inputs_embeds=hidden_states,
      attention_mask=attention_mask,
      causal_attention_mask=causal_attention_mask,
      # output_attentions=output_attentions,
      # output_hidden_states=output_hidden_states,
      return_dict=True,
  )


  last_hidden_state = encoder_outputs[0]
  last_hidden_state = model.text_model.final_layer_norm(last_hidden_state)
 
  cosine_sim1 = F.cosine_similarity(last_hidden_state.view(-1), target, dim=0)
  cosine_sim2 = F.cosine_similarity(last_hidden_state.view(-1), avoid, dim=0)

  loss1 = 1 - cosine_sim1
  loss2 = 1 - cosine_sim2

  loss = loss1 - loss2
  loss.backward()

  return one_hot.grad.clone()

def get_allowed_characters():
    allowed_characters=['·','~','!','@','#','$','%','^','&','*','(',')','=','-','*','+','.','<','>','?',',','\'',';',':','|','\\','/']
    for i in range(ord('A'),ord('Z')+1):
        allowed_characters.append(chr(i))
    for i in range(0,10):
        allowed_characters.append(str(i))
    return allowed_characters

def find_mismatches(list1, list2):
    mismatches = []
    for i, (elem1, elem2) in enumerate(zip(list1, list2)):
        if elem1 != elem2:
            mismatches.append((i, elem1, elem2))
    return mismatches

def check_encode_decode(tokenizer, tokens, MAX_LENGTH):
    text = tokenizer.decode(tokens,skip_special_tokens=True)
    new_tokens = tokenizer(text,return_tensors="pt", padding="max_length",max_length=MAX_LENGTH, truncation=True)["input_ids"][0]
    if tokens.tolist()==new_tokens.tolist():
        return True
    return False


def gradient_greedy_search(T,k,B,model, input_tokens, target, avoid, 
                           suff_start, suff_end, allowed_char_idxes, 
                           tried_adv_tokens, START_THRESH, 
                           END_THRESH, SKIP_TRIED_SUFFIXES, tokenizer,
                           MAX_LENGTH):
    max_sim = -np.infty #MAX
    adv_input_tokens = input_tokens.clone()

    cos_dim1 = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    for t in range(T):
      g = get_grad(model, adv_input_tokens, target, avoid, suff_start, suff_end)
      
      if allowed_char_idxes is not None:
          mask = torch.ones_like(g, dtype=torch.bool)
          mask[:, allowed_char_idxes] = 0
          g[mask] = np.infty

      indices = (-g).topk(k).indices
 
      new_adv_tokens = []
      b = 0
      while b<B:
        adv_input_tokens_b = adv_input_tokens.clone()
        for i in range(suff_start, suff_end):
            if random.random() < max(END_THRESH, START_THRESH - t/T):
                adv_input_tokens_b[i] = indices[i-suff_start][torch.randint(k, (1,))]
        
        if SKIP_TRIED_SUFFIXES and adv_input_tokens_b in tried_adv_tokens:
            continue

        tried_adv_tokens.add(adv_input_tokens_b)
        new_adv_tokens.append(adv_input_tokens_b)
        b+=1

      new_adv_tokens = torch.stack(new_adv_tokens)

      output_embeds = None
      with torch.no_grad():
        output_embeds = model(new_adv_tokens)[0].view(B,-1)

      target_expanded = target.unsqueeze(0).repeat(B,1)
      avoid_expanded = avoid.unsqueeze(0).repeat(B,1)

 
      cos_sim1 = cos_dim1(target_expanded, output_embeds)
      cos_sim2 = cos_dim1(avoid_expanded, output_embeds)
            
      cos_sim = cos_sim1 - cos_sim2
      max_idx = torch.argmax(cos_sim,dim=0)

      if max_sim < cos_sim[max_idx] and check_encode_decode(tokenizer, new_adv_tokens[max_idx], MAX_LENGTH):
        # print("t:",t, "cos:", cos_sim[max_idx].item(), "adv_prompt:", tokenizer.decode(new_adv_tokens[max_idx], clean_up_tokenization_spaces =True,skip_special_tokens=True))
        print("t:",t, "cos:", "{:.3f}".format(cos_sim[max_idx].item()), "adv_prompt:", tokenizer.decode(new_adv_tokens[max_idx], clean_up_tokenization_spaces =True,skip_special_tokens=True))
        max_sim = cos_sim[max_idx]
        adv_input_tokens = new_adv_tokens[max_idx].clone()
    
    print("Done: ", tokenizer.decode(adv_input_tokens, clean_up_tokenization_spaces =True,skip_special_tokens=True))
    return adv_input_tokens


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

