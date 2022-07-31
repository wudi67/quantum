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
class QOuter(torch.nn.Module):
    def __init__(self):
        super(QOuter, self).__init__()
        
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
        # real = x[0].transpose(0,1)
        # imag = x[1].transpose(0,1)  #(33,32,50)

        real = torch.unsqueeze(x[0], dim = 0)
        imag = torch.unsqueeze(x[1], dim = 0)
        output = []
        for r, i in zip(real, imag):    #(32,50)
            output_rr = []
            output_ii = []
            for rr, ii in zip(r,i):     #(50)
                unsqueezed_rr = torch.unsqueeze(rr, dim = -1)
                unsqueezed_ii = torch.unsqueeze(ii, dim = -1)         #(50,1)
                _r = torch.mm(unsqueezed_rr,unsqueezed_rr.t())+ torch.mm(unsqueezed_ii, unsqueezed_ii.t())
                _i = -torch.mm(unsqueezed_rr,unsqueezed_ii.t())+ torch.mm(unsqueezed_ii, unsqueezed_rr.t())
                
                output_rr.append(_r)    #(50,50)
                output_ii.append(_i)
            
            output_rr = torch.stack(output_rr, dim = 0)
            output_ii = torch.stack(output_ii, dim = 0)     #(32,50,50)
            output.append([output_rr,output_ii])

        return output[0]

#
        


    
    
    
    
