#!/usr/bin/env python
import pickle
import argparse
import numpy as np
import warnings
from functools import partial
import matplotlib
import matplotlib.pyplot as plt
import math
from copy import deepcopy

def colorize(words, color_array, colors='RdBu'):
    cmap = plt.cm.get_cmap(colors)
    template = '<span class="barcode"; style="color: black; \
                background-color: {}">{}</span>'
    colored_string = ''
    for word, color in zip(words, color_array):
        color = matplotlib.colors.rgb2hex(cmap(color)[:3])
        if word == '<unk>':
            word = '&ltunk&gt'
        colored_string += template.format(color, '&nbsp' + word + '&nbsp')
    return colored_string

with open('visualization_blog.html', 'a') as f:
    f.write('<table style="width:100%"> <tr> <th>Normalization Method</th> <th>Color</th> <th>Text</th> </tr>')

# Methods to try 
# Rescale by standard deviation (see section 4.1.3 of https://arxiv.org/pdf/1801.05453.pdf)
# Multiply gradient by word embedding
# Divide by total std dev across all words
# probably others (i had some more ideas on my phone)

all_normalized_scores = pickle.load(open('cached/grad_cached_scores.pkl','rb'))
normalization_methods = ['binary_across_words', 'across_words', 'divide_by_total_max', 'divide_by_total_binary_max', 'divide_binary_max', 'divide_max']
colors = ['RdBu', 'seismic', 'bwr','coolwarm'] #'cool','BuPu','plasma','inferno']
clip_methods = ['clip_to_one','clip_to_99th_percentile','none']

all_scores = []
for words, normalized_scores in all_normalized_scores:     
    for score in normalized_scores:   
        all_scores.append(score)
all_scores.sort()
max_99th_score_cap = all_scores[int(.99*len(all_scores))]
min_99th_score_cap = all_scores[int(.01*len(all_scores))]
max_score_overall = max(all_scores[-1], math.fabs(all_scores[0]))

for words, normalized_scores in all_normalized_scores:        
    for color in colors:        
        for normalization_method in normalization_methods:        
            for clip_method in clip_methods:                
                
                scores = deepcopy(normalized_scores)
                if color != 'RdBu':
                    scores = [-1 * s for s in scores] # flip colors for non RdBu colors

                if clip_method == "clip_to_one": # clip to 1 or -1                   
                    for idx, s in enumerate(scores):                                    
                        if scores[idx] > 1:
                            scores[idx] = 1
                        if scores[idx] < -1:
                            scores[idx] = -1       

                if clip_method == "clip_to_99th_percentile": # clip to 1 or -1                   
                    for idx, s in enumerate(scores):                                    
                        if scores[idx] > max_99th_score_cap:
                            scores[idx] = max_99th_score_cap
                        if scores[idx] < min_99th_score_cap:
                            scores[idx] = min_99th_score_cap       
                
                if normalization_method == 'binary_across_words':
                    total_score_pos = 1e-6    # 1e-6 for case where all positive/neg scores are 0
                    total_score_neg = 1e-6
                    for idx, s in enumerate(scores):
                        if s < 0:
                            total_score_neg = total_score_neg + math.fabs(s)
                        else:
                            total_score_pos = total_score_pos + s
                    for idx, s in enumerate(scores):
                        if s < 0:
                            scores[idx] = (s / total_score_neg) / 2
                        else:
                            scores[idx] = (s / total_score_pos) / 2

                elif normalization_method == 'across_words':            
                    total_score = 0
                    for idx, s in enumerate(scores):
                        total_score = total_score + math.fabs(s)
                    for idx, s in enumerate(scores):
                        scores[idx] = (s / total_score) / 2
        
                elif normalization_method == 'divide_by_total_max':
                    for idx, s in enumerate(scores):                                                        
                        scores[idx] = scores[idx] / max_score_overall

                elif normalization_method == 'divide_by_total_binary_max':
                    for idx, s in enumerate(scores):                                                        
                        if s < 0:                        
                            scores[idx] = scores[idx] / math.fabs(all_scores[0])
                        if s > 0:
                            scores[idx] = scores[idx] / math.fabs(all_scores[-1])

                elif normalization_method == 'divide_binary_max':
                    max_score_pos = 0
                    max_score_neg = 0
                    for idx, s in enumerate(scores):
                        if s < 0:
                            max_score_neg = min(max_score_neg, s)
                        else:
                            max_score_pos = max(max_score_pos, s)

                    for idx, s in enumerate(scores):
                        if s < 0:
                            scores[idx] = (s / math.fabs(max_score_neg)) / 2  # abs to prevent sign flipping when divide by negative
                        else:
                            scores[idx] = (s / max_score_pos) / 2

                elif normalization_method == 'divide_max':
                    max_score = 0            
                    for idx, s in enumerate(scores):                
                        max_score = max(max_score, math.fabs(s))

                    for idx, s in enumerate(scores):                
                        scores[idx] = (s / max_score) / 2           


                scores = [0.5 + n for n in scores]  # center scores                    
                visual = colorize(words, scores, colors=color)    
                with open('visualization_blog.html', 'a') as f:
                    f.write('<tr>')
                    
                    f.write('<td>')                        
                    f.write(color)            
                    f.write('</td>')

                    f.write('<td>')                        
                    f.write(normalization_method)            
                    f.write('</td>')

                    f.write('<td>')                        
                    f.write(clip_method)            
                    f.write('</td>')

                    f.write('<td>') 
                    f.write(visual)
                    f.write('</td>')    
                    f.write('</tr>')