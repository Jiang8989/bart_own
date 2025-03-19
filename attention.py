from torch import nn
import torch
import math
import numpy as np
import torch.nn.functional as F

from typing import Literal,Optional,Tuple, Union

class MultiHeadAttention(nn.Module):
  '''
  param hidden_size:int 隐含层神经元个数
  param num_attention_heads:int 多头注意力的多头数
  param attention_probs_dropout_prob:float softmax后的dropout rate
  param dropout_rate: float pos_dropout对应的dropout rate  
  param attention_scale: bool 是否对attention_scores进行缩放，默认为True
  param output_attention: bool 是否返回attention_scores，默认为False
  param bias: bool, qkvo的weight是否包含bias，默认为True
  param rope_scaling:dict,rope的position encoding的参数，默认为None
  param _attn_implementation:Literal枚举值，计算attention score的方式，支持'sdpa'，'xformers',
'flash_attn_2',"eager"等，默认为None
  param use_logn_attn:bool,是否使用use_logn_attn,默认为None
  param layer_idx: int, transformer block的层序号
'''
  def __init__(self,
            hidden_size:int,
            num_attention_heads:int,
            attention_probs_dropout_prob: float,
            dropout_rate: float=0.1,
            attention_scale:bool = True,
            output_attention: bool = False,
            bias:bool = True,
            rope_scaling: dict = None,
            _attn_implementation:Literal['sdpa','xformers','flash_attn_2','eager']='eager',
            use_logn_attn:bool=None,
            layer_idx:int=None,
            num_key_value_heads:int=None,
            **kwargs):
    super(MultiHeadAttention,self).__init__()
    self.hidden_size= hidden_size
    self.num_attention_heads = num_attention_heads  
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.dropout_rate = dropout_rate
    self.is_decoder = kwargs.get('is_decoder',False)
    self.attention_scale = attention_scale
    self.output_attentions = output_attentions
    self.bias = bias
    self.rope_scaling = rope_scaling or dict()
    self.layer_idx = layeer_dix
    self.sliding_window = kwargs.get('sliding_window')
    self.max_window_layers = kwargs.get('max_window_layeers')
    self._attn_implementation = _attn_implementation # attention的实现
    self.use_logn_attn = use_logn_attn #使用logn_attn
    self.max_position = kwargs.get('max_position')
    # t5_pegasus_small中hidden_size/num_attention_heads!= 0

    self.attention_head_size = kwargs.get('attention_head_size',int(hidden_size/num_attention_heads))
    self.attention_key_size = kwargs.get('attention_key_size',self.attention_head_size)
    self.scaling = self.attention_head_size**(-0.5)
    q_inner_dim = self.attention_key_size*num_attention_heads
    k_inner_dim = q_inner_dim
    v_inner_dim = self.attention_head_size*num_attention_heads

    # multi query attention: chatglm中叫num_key_value_heads
    if num_key_value_heads is not None:
        self.num_key_value_heads = num_key_value_heads
        k_inner_dim_tmp = self.attention_head_size*self.num_key_value_heads
        v_inner_dim_tmp = k_inner_dim_tmp
    #longlora
    if kwargs.get('longlora_group_size') is not None:
        self.longlora_group_size = kwargs.get('longlora_group_size')

    self.q = nn.Linear(hidden_size,q_inner_dim,bias = bias)
    self.k = nn.Linear(hidden_size,k_inner_dim_tmp if hasattr(self,'num_key_value_heads')else k_inner_dim,bias=bias)
    self.v = nn.Linear(hidden_size,v_inner_dim_tmp if hasattr(self,'num_key_value_heads')else v_inner_dim,bias=bias)
    self.o = nn.Linear(v_inner_dim,hidden_size,bias=bias)
    self.dropout = nn.Dropout(attention_probs_dropout_prob) if attention_probs_dropout_prob >0 else lambda x:x
    self.init_position_encoding(**kwargs)

    def _get_qkv_states(self,hidden_states,attention_mask,encoder_hidden_states,encoder_attention_mask,past_key_value,position_ids):
        '''获取qkv states,主要是未来下游继承'''
        pass

    def forward(self,
                hidden_states:Optional[torch.Tensor]=None,
                attention_mask:Optional[torch.FloatTensor]=None,
                encoder_hidden_states:Optional[torch.FloatTensor]=None,
                encoder_attention_mask:Optional[torch.FloatTensor]=None,
                past_key_value:Optional[Tuplep[Tuple[torch.FloatTensor]]]=None,
                position_ids=None,
                **model_kwargs
        ):
        '''
        :param hidden_states:[batch_size,seq_q,hidden_size]
        :param attention_mask:[batch_size,1,1,seq_q]或者[batch_size,1,seq_q,seq_q]
        :param encoder_hidden_states:[batch_size,seq_k,hidden_size]
        :param encoder_attention_mask:[batch_size,]
