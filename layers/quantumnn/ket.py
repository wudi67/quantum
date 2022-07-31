# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 19:45:10 2020

@author: Qiuchi Li
"""

import torch
import torch.nn

'''
returns a list of [real, image] arrays
the length of list is the time stamps
each array is of shape (batch_size, embed_dim, embed_dim)
'''
class QKet(torch.nn.Module):
    def __init__(self):
        super(QKet, self).__init__()
        
    def forward(self, x):

        if not isinstance(x, list):
            raise ValueError('xr should be called '
                             'on a list of 2 inputs.')

        if len(x) != 2:
            raise ValueError('x should be called '
                            'on a list of 2 inputs.'
                            'Got ' + str(len(x)) + ' inputs.')
        
        #x[0], x[1] has shape:
        #(batch_size, time_stamps, embedding_dim) (32,33,50)
        real = x[0].transpose(0,1)
        imag = x[1].transpose(0,1)  #(33,32,50)
        output = []
        

        for r, i in zip(real, imag):    #(32,50)
                 
            output.append([r,i])

        return output

#
        


    
    
    
    
