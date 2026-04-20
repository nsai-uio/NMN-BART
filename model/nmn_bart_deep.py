import sys, os
import torch
import math
import json
import random
sys.path.append('..')
import utils.layers as layers
from transformers.models.bart import modeling_bart, BartConfig
import torch.nn as nn
import composite_modules as modules
# import controller
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqModelOutput,
    Seq2SeqLMOutput,
)
from transformers.models.bart.modeling_bart import BartEncoder
# from . import composite_modules as modules
from controller_nmn import Controller
import numpy as np

# def m_fusion(lm_feats, graph_feats, ie_layers, last_hidden):
#     fuse = torch.cat([graph_feats, lm_feats], dim=1)
#     fuse_feats = ie_layers(fuse)
#     lm_feats, graph_feats = torch.split(fuse_feats, [lm_feats.size(1), graph_feats.size(1)], dim=1)

#     update_last_hidden = torch.clone(last_hidden)
#     update_last_hidden[:, 0, :] = lm_feats
#     hidden_states = update_last_hidden
#     return hidden_states

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

class BartForNLE(modeling_bart.BartForConditionalGeneration):
    def __init__(self, config: BartConfig, module_kwargs):
        super().__init__(config)
        self.model = BARTNmnModel(config, module_kwargs)

    def forward(
        self,
        vision_features: Optional[torch.Tensor] = None,
        relation_masks: Optional[torch.Tensor] = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None, # should be None for no custom embeddings
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            # if use_cache:
            #     logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            vision_features,
            relation_masks,
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = self.lm_head(outputs[0])
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        masked_lm_loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

class BARTNmnModel(modeling_bart.BartModel):
    def __init__(self, config: BartConfig, module_kwargs):
        super().__init__(config)
        self.encoder = BARTNmnEncoder(config, module_kwargs, self.shared)

    def forward(
        self,
        vision_features: Optional[torch.Tensor] = None,
        relation_masks: Optional[torch.Tensor] = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqModelOutput]:
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                vision_features = vision_features,
                relation_masks = relation_masks,
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        return super().forward(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, head_mask, decoder_head_mask, cross_attn_head_mask, encoder_outputs, past_key_values, inputs_embeds, decoder_inputs_embeds, use_cache, output_attentions, output_hidden_states, return_dict)

class BARTNmnEncoder(modeling_bart.BartEncoder):
    def __init__(self, config: BartConfig, module_kwargs, embed_tokens: Optional[nn.Embedding] = None, ie_dim=200, ie_layer_num=1, p_fc=0.2, ):
        super().__init__(config, embed_tokens)
        # self.encoder = BartEncoder(config, self.shared)
        self.module_kwargs = module_kwargs
        self.h_dim = config.hidden_size
        self.dim_ie = module_kwargs["glimpses"] * module_kwargs["dim_vision"]
        self.ie_dim = ie_dim
        self.ie_layers = layers.MLP(self.h_dim + self.dim_ie, self.ie_dim, self.h_dim + self.dim_ie, ie_layer_num, p_fc)
        self.nmn_model = nmn(self.module_kwargs)

    def forward(
        self,
        vision_features: Optional[torch.Tensor] = None,
        relation_masks: Optional[torch.Tensor] = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:

        batch_size = len(input_ids)
        max_len = input_ids.size()[1]
        _questions_len = [max_len] * batch_size 
        questions_len = torch.LongTensor(np.asarray(_questions_len))

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_ids = input_ids.view(-1, input_ids.shape[-1])
        elif inputs_embeds is not None:
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input)
        embed_pos = embed_pos.to(inputs_embeds.device)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]
            # get intermediate outputs from the initial layer
            if idx == 0:
                # get question_output, hidden_states, embedding and length for controller
                _question_outputs = hidden_states
                # change shape to [seq_max_len, batch_size, d]
                question_outputs = _question_outputs.permute(1, 0, 2)

                question_hidden = encoder_states
                last_hidden = question_hidden[-1] # select the last layer
                pooled_hidden = last_hidden[:, 0] # select the first token from each sentence in the batch

                _question_embedding = question_hidden[0]
                question_embedding = _question_embedding.permute(1, 0, 2)
                
                # module output list for intermediate results
                module_output = self.nmn_model(question_outputs, pooled_hidden, question_embedding, questions_len, vision_features, relation_masks)
            
            if idx < 3:
                last_hidden = encoder_states[-1] # select the last layer
                pooled_hidden = last_hidden[:, 0] # select the first token from each sentence in the batch

                _lm_feats = pooled_hidden  
                _graph_feats = module_output[idx-3] #[batch_size, dim]

                fuse = torch.cat([_graph_feats, _lm_feats], dim=1)
                fuse_feats = self.ie_layers(fuse)
                lm_feats, graph_feats = torch.split(fuse_feats, [_lm_feats.size(1), _graph_feats.size(1)], dim=1)

                update_last_hidden = torch.clone(last_hidden)
                update_last_hidden[:, 0, :] = lm_feats
                hidden_states = update_last_hidden

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
                
        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )




class nmn(nn.Module):
    """
    retrive module for each question
    module_kwargs = {
        'dim_v': 512,
        'dim_hidden': 1024,
        'dim_edge': 256,
        'dim_vision': 2053,
        'dropout_prob': 0.5,
        'T_ctrl': 3,
        'glimpses': 2,
        # 'device': device,
        'stack_len': 4,
        'use_gumbel': 1,
    }
    """
    def __init__(self, module_kwargs):
        super().__init__()

        for k, v in module_kwargs.items():
            setattr(self, k, v)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.map_vision_to_v = nn.Sequential(
                    nn.Dropout(self.dropout_prob),
                    nn.Linear(self.dim_vision, self.dim_v, bias=False),
                    )
        self.map_two_v_to_edge = nn.Sequential(
                    nn.Dropout(self.dropout_prob),
                    nn.Linear(self.dim_v * 2, self.dim_edge, bias=False),
                    )

        # modules
        self.module_names = modules.MODULE_INPUT_NUM.keys()
        self.num_module = len(self.module_names)
        self.module_funcs = [getattr(modules, m[1:]+'Module')(**module_kwargs) for m in self.module_names]
        self.module_validity_mat_tmp = modules._build_module_validity_mat(self.stack_len, self.module_names)
        self.module_validity_mat = torch.Tensor(self.module_validity_mat_tmp).to(self.device)
        for name, func in zip(self.module_names, self.module_funcs):
            self.add_module(name, func)

        # controller
        controller_kwargs = {
            'num_module': len(self.module_names),
            'dim_model': self.dim_hidden,
            'T_ctrl': self.T_ctrl,
            'use_gumbel': self.use_gumbel,
        }
        self.controller = Controller(**controller_kwargs)


    def forward(self, question_outputs, question_hidden, question_embedding, questions_len, vision_feat, relation_mask):
        """
        Args:
        questions [Tensor] (batch_size, seq_len)
        question_hidden: pooled_hidden after first token pooling
        questions_len [Tensor] (batch_size)
        vision_feat (batch_size, dim_vision, num_feat)
        relation_mask (batch_size, num_feat, num_feat)
        """
        batch_size = len(questions_len)
        module_logits, module_probs, c_list, cv_list = self.controller(
            question_outputs, question_hidden, question_embedding, questions_len)
        
        ## feature processing
        vision_feat_n = vision_feat / (vision_feat.norm(p=2, dim=1, keepdim=True) + 1e-12)
        feat_inputs = vision_feat_n.permute(0,2,1)
        if self.dim_v != self.dim_vision:
            feat_inputs_up = self.map_vision_to_v(feat_inputs) # (batch_size, num_feat, dim_v)
        num_feat = feat_inputs_up.size(1)
        feat_inputs_expand_0 = feat_inputs_up.unsqueeze(1).expand(batch_size, num_feat, num_feat, self.dim_v)
        feat_inputs_expand_1 = feat_inputs_up.unsqueeze(2).expand(batch_size, num_feat, num_feat, self.dim_v)
        _feat_edge = torch.cat([feat_inputs_expand_0, feat_inputs_expand_1], dim=3) # (bs, num_feat, num_feat, 2*dim_v)
        feat_edge = self.map_two_v_to_edge(_feat_edge)

        ## stack initialization
        # memroy stack to store image attention maps
        att_stack = torch.zeros(batch_size, num_feat, self.glimpses, self.stack_len).to(self.device)
        stack_ptr = torch.zeros(batch_size, self.stack_len).to(self.device)
        stack_ptr[:, 0] = 1
        mem = torch.zeros(batch_size, self.glimpses * self.dim_vision).to(self.device)
        intermediate_mem = []
        for t in range(self.T_ctrl):
            c_i = c_list[t] #(batch_size, dim_hidden)
            module_logit = module_logits[t] # .to(torch.bool) # (batch_size, num_module)
            if self.use_validity:
                if t < self.T_ctrl-1:
                    module_validity = torch.matmul(stack_ptr, self.module_validity_mat)
                    module_validity[:, 5] = 0  # (batch_size, num_module)
                else: # last step must describe
                    module_validity = torch.zeros(batch_size, self.num_module).to(self.device)
                    module_validity[:, 5] = 1
                module_invalidity = (1 - torch.round(module_validity)).byte() # hard validate
                # module_invalidity = (1 - torch.round(module_validity)).bool()
                module_logit.masked_fill_(module_invalidity, -float('inf'))
                module_prob = F.gumbel_softmax(module_logit, hard=self.use_gumbel)
            else:
                module_prob = module_probs[t]
            module_prob = module_prob.permute(1,0) # (num_module, batch_size)

            # run all modules
            res = [f(vision_feat.permute(0,2,1), feat_inputs_up, feat_edge, c_i, relation_mask, att_stack, stack_ptr, mem) for f in self.module_funcs]
            att_stack_avg = torch.sum(module_prob.view(self.num_module,batch_size,1,1,1) * torch.stack([r[0] for r in res]), dim=0)
            _stack_ptr_avg = torch.sum(module_prob.view(self.num_module,batch_size,1) * torch.stack([r[1] for r in res]), dim=0)
            stack_ptr_avg = modules._sharpen_ptr(_stack_ptr_avg, hard=False)
            mem_avg = torch.sum(module_prob.view(self.num_module,batch_size,1) * torch.stack([r[2] for r in res]), dim=0)
            att_stack, stack_ptr, mem = att_stack_avg, stack_ptr_avg, mem_avg
            intermediate_mem.append(mem)

        return intermediate_mem





        

    








    