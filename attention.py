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
    q_inner_dim = self.attention_key_size*self.num_attention_heads
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
        '''获取qkv states,主要是为了下游继承'''
        query_states = self.transpose_for_q_scores(self.q(hidden_states))
        if (encoder_hidden_states is not None) and (past_key_value is not None):
            key_states,value_states = past_key_value
            attention_mask = encoder_attention_mask
        elif encoder_hidden_states is not None:
            key_states = self.transpose_for_k_scores(self.k(encoder_hidden_states))
            value_states = self.transpose_for_v_scores(self.v(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_states = self.transpose_for_k_scores(self.k(hidden_states))
            values_sstates = self.transpose_for_v_scores(self.v(encoder_hidden_states))
            key_states = torch.cat([past_key_value[0],key_states],dim=2)
            value_states = torch.cat([past_key_value[1],value_states],dim=2)
        else:
            key_states = self.transpose_for_k_scores(self.k(hidden_states))
            value_states = self.transpose_for_v_scores(self.v(hidden_states))
        return query_states,key_states,value_states,attention_mask
              
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
        :param encoder_attention_mask:[batch_size,1,1seq_k]
        :param past_key_value:([batch_size,num_attention_heads,key_len_cache,attention_head_size],...)
        '''
        query_states,key_states,value_states,attention_mask = self._get_qkv_states(
          hidden_states,attention_mask,encoder_hidden_states,encoder_attention_mask,
          past_key_value,position_ids)
        #query_states shape:[batch_size,num_hidden_heads,query_len,attention_head_size]
        #key_states shape:[batch_size,num_attention_heads,key_len,attention_head_size]
        #value_states shape:[batch_size,num_attention_heads,value_len,attention_head_size]

        #使用logn_attn
        if self.use_long_attn:
            query_states*=((position_ids+1)[:,None,:,None].log()/np.log(self.max_position)).clip(1).to(query_states.dtypes)

        #past_key_values
        if self.is_decoder and (not self.training):#仅推理时记录
            past_key_value = (key_states,value_states)

        # multi_query_attention
        if hasattr(self,'num_key_value_heads')and self.num_key_value_heads>1:
            key_states= self.repeat_kv(key_states)
            value_states = self.repeat_kv(values_states)

        # longlora
        if hasattr(self,'longlora_group_size'):
            query_states,key_states,value_states,attention_mask= self.longlora_shift(query_states,key_states,value_states,attention_mask)


        #attention多类实现
        #xfoemers
        attention_scores = None
        if(self._attn_implementation == 'xforemers') and self.training:
            context_layer = xops.memory_efficient_attention(query_states,key_states,value_states,attn_bias=xops.LowerTraingularMask())
        #SDPA
        elif self._attn_implementation in {True,'sdpa'}:
            context_layer = self.flash_attention_forward(query_states,key_states,value_states,attention_mask)  
        # flash_attn
        elif self._attn_implementation == 'flash_attn_2':
            context_layer = self.spda_attention_forward(query_states,key_states,value_states,past_key_value,attention_mask,hidden_states.shape[1])
        # torch原生实现
        else:
            context_layer,attention_scores = self.torch_attention_forward(query_states,key_states,value_states,attention_mask)

        if hasattr(self,'longlora_group_size'):
      #context_layer:[bsz*(q_len//group_size),num_heads,group_size,head_dim]
            bsz,q_len= hidden_states.shape[:2]
            context_layer = context_layer.transpose(1,2).contiguous()
            context_layer = context_layer.reshape(bsz,q_len,self.num_attention_heads,self.attention_head_size)
            # shift back
            context_layer[:,:,self.num_attention_heads//2:]= context_layer[:,:,self.num_attention_heads//2:].roll(self.longlora_group_size//2,dims=1)
            context_layer = context_layer.reshape(bsz,q_len,self.hidden_size)
        else:
            # context_layer shape:[batch_size,num_attention_heads,query_len,attention_head_size]
            context_layer = context_layer.permute(0,2,1,3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2]+(context_layer.size()[-2]*context_layer.size()[-1],)
            context_layer = context_layer.reshape(*new_context_layer_shape).contiguous()

        # 是否返回attention scores
        outputs = (self.o(context_layer),attention_scores)if self.output_attentions else (self.o(context_layer),)
        return outputs+(past_key_value,) if self.is_decoder else outputs

    def repeat_kv(self,hidden_states):
        hidden_states = hidden_states.unsqueeze(2)
        hidden_states = hidden_states.expand(-1,-1,self.num_attention_heads//self.num_key_value_heads,-1,-1)
        hidden_states = hidden_states.contiguous().view(hidden_states.shape[:1]+(self.num_attention_heads,)+hidden_states.shape[-2:])
        return hidden_states

    def longlora_shift(self,query_states,key_states,value_states,attention_mask):
        
