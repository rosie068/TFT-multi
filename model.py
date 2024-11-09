import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import math
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
import warnings

from typing import List, Dict, Tuple, Optional
import copy
import math
from omegaconf import OmegaConf,DictConfig

from functools import partial
from tqdm import tqdm
import torch.nn.init as init

class TimeDistributed(nn.Module):
    def __init__(self, module: nn.Module, batch_first: bool = True, return_reshaped: bool = True):
        super(TimeDistributed, self).__init__()
        self.module: nn.Module = module
        self.batch_first: bool = batch_first
        self.return_reshaped: bool = return_reshaped

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * time-steps, input_size)
        y = self.module(x_reshape)

        if self.return_reshaped:
            if self.batch_first:
                y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, time-steps, output_size)
            else:
                y = y.view(-1, x.size(1), y.size(-1))  # (time-steps, samples, output_size)

        return y


class NullTransform(nn.Module):
    def __init__(self):
        super(NullTransform, self).__init__()

    @staticmethod
    def forward(empty_input: torch.tensor):
        return []


class GatedLinearUnit(nn.Module):
    def __init__(self, input_dim: int):
        super(GatedLinearUnit, self).__init__()

        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        sig = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return torch.mul(sig, x)


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout: Optional[float] = 0.05,
                 context_dim: Optional[int] = None,
                 batch_first: Optional[bool] = True):
        super(GatedResidualNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # =================================================
        # Input conditioning components (Eq.4 in the original paper)
        # =================================`================
        self.project_residual: bool = self.input_dim != self.output_dim
        if self.project_residual:
            self.skip_layer = TimeDistributed(nn.Linear(self.input_dim, self.output_dim))

        self.fc1 = TimeDistributed(nn.Linear(self.input_dim, self.hidden_dim), batch_first=batch_first)

        if self.context_dim is not None:
            self.context_projection = TimeDistributed(nn.Linear(self.context_dim, self.hidden_dim, bias=False),
                                                      batch_first=batch_first)
        self.elu1 = nn.ELU()

        # ============================================================
        # Further projection components (Eq.3 in the original paper)
        # ============================================================
        self.fc2 = TimeDistributed(nn.Linear(self.hidden_dim, self.output_dim), batch_first=batch_first)

        # ============================================================
        # Output gating components (Eq.2 in the original paper)
        # ============================================================
        self.dropout = nn.Dropout(self.dropout)
        self.gate = TimeDistributed(GatedLinearUnit(self.output_dim), batch_first=batch_first)
        self.layernorm = TimeDistributed(nn.LayerNorm(self.output_dim), batch_first=batch_first)

    def forward(self, x, context=None):
        if self.project_residual:
            residual = self.skip_layer(x)
        else:
            residual = x
        # ===========================
        # Compute Eq.4
        # ===========================
        x = self.fc1(x)
        if context is not None:
            context = self.context_projection(context)
            x = x + context

        x = self.elu1(x)

        # ===========================
        # Compute Eq.3
        # ===========================
        x = self.fc2(x)

        # ===========================
        # Compute Eq.2
        # ===========================
        x = self.dropout(x)
        x = self.gate(x)
        x = x + residual
        x = self.layernorm(x)

        return x

    
class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_dim: int, num_inputs: int, hidden_dim: int, dropout: float,
                 context_dim: Optional[int] = None,
                 batch_first: Optional[bool] = True):
        super(VariableSelectionNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_inputs = num_inputs
        self.dropout = dropout
        self.context_dim = context_dim

        self.flattened_grn = GatedResidualNetwork(input_dim=self.num_inputs * self.input_dim,
                                                  hidden_dim=self.hidden_dim,
                                                  output_dim=self.num_inputs,
                                                  dropout=self.dropout,
                                                  context_dim=self.context_dim,
                                                  batch_first=batch_first)
        self.softmax = nn.Softmax(dim=1)

        self.single_variable_grns = nn.ModuleList()
        for _ in range(self.num_inputs):
            self.single_variable_grns.append(
                GatedResidualNetwork(input_dim=self.input_dim,
                                     hidden_dim=self.hidden_dim,
                                     output_dim=self.hidden_dim,
                                     dropout=self.dropout,
                                     batch_first=batch_first))

    def forward(self, flattened_embedding, context=None):
        sparse_weights = self.flattened_grn(flattened_embedding, context)
        sparse_weights = self.softmax(sparse_weights).unsqueeze(2)
        # [(num_samples * num_temporal_steps) x num_inputs x 1]

        processed_inputs = []
        for i in range(self.num_inputs):
            processed_inputs.append(
                self.single_variable_grns[i](flattened_embedding[..., (i*self.input_dim): (i+1)*self.input_dim]))
        # [(num_samples * num_temporal_steps) x state_size]

        processed_inputs = torch.stack(processed_inputs, dim=-1)
        # processed_inputs: [(num_samples * num_temporal_steps) x state_size x num_inputs]

        outputs = processed_inputs * sparse_weights.transpose(1, 2)
        # outputs: [(num_samples * num_temporal_steps) x state_size x num_inputs]

        outputs = outputs.sum(axis=-1)
        # outputs: [(num_samples * num_temporal_steps) x state_size]

        return outputs, sparse_weights
    
    
class InputChannelEmbedding(nn.Module):
    def __init__(self, state_size: int, num_numeric: int, num_categorical: int, 
                 categorical_cardinalities: List[int], time_distribute: Optional[bool] = False):
        super(InputChannelEmbedding, self).__init__()

        self.state_size = state_size
        self.num_numeric = num_numeric
        self.num_categorical = num_categorical
        self.categorical_cardinalities = categorical_cardinalities
        self.time_distribute = time_distribute

        if (num_numeric + num_categorical) < 1:
            raise ValueError(f"""At least a single input variable (either numeric or categorical) should included
            as part of the input channel.
            According to the provided configuration:
            num_numeric + num_categorical = {num_numeric} + {num_categorical} = {num_numeric + num_categorical} < 1
            """)

        if self.time_distribute:
            self.numeric_transform = TimeDistributed(
                NumericInputTransformation(num_inputs=num_numeric, state_size=state_size), return_reshaped=False)
            self.categorical_transform = TimeDistributed(
                CategoricalInputTransformation(num_inputs=num_categorical, state_size=state_size,
                                               cardinalities=categorical_cardinalities), return_reshaped=False)
        else:
            self.numeric_transform = NumericInputTransformation(num_inputs=num_numeric, state_size=state_size)
            self.categorical_transform = CategoricalInputTransformation(num_inputs=num_categorical,
                                                                        state_size=state_size,
                                                                        cardinalities=categorical_cardinalities)

        if num_numeric == 0:
            self.numeric_transform = NullTransform()
        if num_categorical == 0:
            self.categorical_transform = NullTransform()

    def forward(self, x_numeric, x_categorical) -> torch.tensor:
        batch_shape = x_numeric.shape if x_numeric.nelement() > 0 else x_categorical.shape

        processed_numeric = self.numeric_transform(x_numeric)
        
        processed_categorical = self.categorical_transform(x_categorical)
        merged_transformations = torch.cat(processed_numeric + processed_categorical, dim=1)
        # [(num_samples * num_temporal_steps) x (state_size * total_input_variables)]

        if self.time_distribute:
            merged_transformations = merged_transformations.view(batch_shape[0], batch_shape[1], -1)
            # [num_samples x num_temporal_steps x (state_size * total_input_variables)]

        return merged_transformations

    
class NumericInputTransformation(nn.Module):
    def __init__(self, num_inputs: int, state_size: int):
        super(NumericInputTransformation, self).__init__()
        self.num_inputs = num_inputs
        self.state_size = state_size

        self.numeric_projection_layers = nn.ModuleList()
        for _ in range(self.num_inputs):
            self.numeric_projection_layers.append(nn.Linear(1, self.state_size))

    def forward(self, x: torch.tensor) -> List[torch.tensor]:
        projections = []
        for i in range(self.num_inputs):
            projections.append(self.numeric_projection_layers[i](x[:, [i]]))

        return projections
    
class CategoricalInputTransformation(nn.Module):
    def __init__(self, num_inputs: int, state_size: int, cardinalities: List[int]):
        super(CategoricalInputTransformation, self).__init__()
        self.num_inputs = num_inputs
        self.state_size = state_size
        self.cardinalities = cardinalities

        self.categorical_embedding_layers = nn.ModuleList()
        for idx, cardinality in enumerate(self.cardinalities):
            self.categorical_embedding_layers.append(nn.Embedding(cardinality, self.state_size))

    def forward(self, x: torch.tensor) -> List[torch.tensor]:
        embeddings = []
        for i in range(self.num_inputs):
            embeddings.append(self.categorical_embedding_layers[i](x[:, i]))

        return embeddings
    
class GateAddNorm(nn.Module):
    def __init__(self, input_dim: int, dropout: Optional[float] = None):
        super(GateAddNorm, self).__init__()
        self.dropout_rate = dropout
        if dropout:
            self.dropout_layer = nn.Dropout(self.dropout_rate)
        self.gate = TimeDistributed(GatedLinearUnit(input_dim), batch_first=True)
        self.layernorm = TimeDistributed(nn.LayerNorm(input_dim), batch_first=True)

    def forward(self, x, residual=None):
        if self.dropout_rate:
            x = self.dropout_layer(x)
        x = self.gate(x)
        if residual is not None:
            x = x + residual
        x = self.layernorm(x)

        return x

class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super(InterpretableMultiHeadAttention, self).__init__()

        self.d_model = embed_dim  
        self.num_heads = num_heads 
        self.all_heads_dim = embed_dim * num_heads 

        self.w_q = nn.Linear(embed_dim, self.all_heads_dim) 
        self.w_k = nn.Linear(embed_dim, self.all_heads_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim) 
        self.out = nn.Linear(self.d_model, self.d_model)

    def forward(self, q, k, v, mask=None):
        num_samples = q.size(0)

        # Dimensions:
        # queries tensor - q: [num_samples x num_future_steps x state_size]
        # keys tensor - k: [num_samples x (num_total_steps) x state_size]
        # values tensor - v: [num_samples x (num_total_steps) x state_size]
        q_proj = self.w_q(q).view(num_samples, -1, self.num_heads, self.d_model)
        k_proj = self.w_k(k).view(num_samples, -1, self.num_heads, self.d_model)
        v_proj = self.w_v(v).repeat(1, 1, self.num_heads).view(num_samples, -1, self.num_heads, self.d_model)

        q_proj = q_proj.transpose(1, 2)  # (num_samples x num_future_steps x num_heads x state_size)
        k_proj = k_proj.transpose(1, 2)  # (num_samples x num_total_steps x num_heads x state_size)
        v_proj = v_proj.transpose(1, 2)  # (num_samples x num_total_steps x num_heads x state_size)

        attn_outputs_all_heads, attn_scores_all_heads = self.attention(q_proj, k_proj, v_proj, mask)
        # attn_scores_all_heads: [num_samples x num_heads x num_future_steps x num_total_steps]
        # attn_outputs_all_heads: [num_samples x num_heads x num_future_steps x state_size]

        attention_scores = attn_scores_all_heads.mean(dim=1)
        attention_outputs = attn_outputs_all_heads.mean(dim=1)
        # attention_scores: [num_samples x num_future_steps x num_total_steps]
        # attention_outputs: [num_samples x num_future_steps x state_size]

        output = self.out(attention_outputs)
        # output: [num_samples x num_future_steps x state_size]

        return output, attention_outputs, attention_scores

    def attention(self, q, k, v, mask=None):
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)
        # attention_scores: [num_samples x num_heads x num_future_steps x num_total_steps]

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask, -1e9)

        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_outputs = torch.matmul(attention_scores, v)
        # attention_outputs: [num_samples x num_heads x num_future_steps x state_size]

        return attention_outputs, attention_scores
    
class TemporalFusionTransformer(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()

        self.config = config

        data_props = config['data_props']
        self.num_historical_numeric = data_props.get('num_historical_numeric', 0)
        self.num_historical_categorical = data_props.get('num_historical_categorical', 0)
        self.historical_categorical_cardinalities = data_props.get('historical_categorical_cardinalities', [])

        self.num_static_numeric = data_props.get('num_static_numeric', 0)
        self.num_static_categorical = data_props.get('num_static_categorical', 0)
        self.static_categorical_cardinalities = data_props.get('static_categorical_cardinalities', [])

        self.num_future_numeric = data_props.get('num_future_numeric', 0)
        self.num_future_categorical = data_props.get('num_future_categorical', 0)
        self.future_categorical_cardinalities = data_props.get('future_categorical_cardinalities', [])

        self.historical_ts_representative_key = 'historical_ts_numeric' if self.num_historical_numeric > 0 \
            else 'historical_ts_categorical'
        self.future_ts_representative_key = 'future_ts_numeric' if self.num_future_numeric > 0 \
            else 'future_ts_categorical'
        
        self.num_features_to_predict = data_props.get('num_feature_predicted', 0)
       
        self.task_type = config.task_type
        self.attention_heads = config.model.attention_heads
        self.dropout = config.model.dropout
        self.lstm_layers = config.model.lstm_layers
        self.target_window_start_idx = (config.target_window_start-1) if config.target_window_start is not None else 0
        if self.task_type == 'regression':
            self.output_quantiles = config.model.output_quantiles
            self.num_outputs = len(self.output_quantiles)
        elif self.task_type == 'classification':
            self.output_quantiles = None
            self.num_outputs = 1
        else:
            raise ValueError(f"unsupported task type: {self.task_type}")
        self.state_size = config.model.state_size

        self.static_transform = InputChannelEmbedding(state_size=self.state_size,
                                                      num_numeric=self.num_static_numeric,
                                                      num_categorical=self.num_static_categorical,
                                                      categorical_cardinalities=self.static_categorical_cardinalities,
                                                      time_distribute=False)

        self.historical_ts_transform = InputChannelEmbedding(
            state_size=self.state_size,
            num_numeric=self.num_historical_numeric,
            num_categorical=self.num_historical_categorical,
            categorical_cardinalities=self.historical_categorical_cardinalities,
            time_distribute=True)

        self.future_ts_transform = InputChannelEmbedding(
            state_size=self.state_size,
            num_numeric=self.num_future_numeric,
            num_categorical=self.num_future_categorical,
            categorical_cardinalities=self.future_categorical_cardinalities,
            time_distribute=True)

        self.static_selection = VariableSelectionNetwork(
            input_dim=self.state_size,
            num_inputs=self.num_static_numeric + self.num_static_categorical,
            hidden_dim=self.state_size, dropout=self.dropout)

        self.historical_ts_selection = VariableSelectionNetwork(
            input_dim=self.state_size,
            num_inputs=self.num_historical_numeric + self.num_historical_categorical,
            hidden_dim=self.state_size,
            dropout=self.dropout,
            context_dim=self.state_size)

        self.future_ts_selection = VariableSelectionNetwork(
            input_dim=self.state_size,
            num_inputs=self.num_future_numeric + self.num_future_categorical,
            hidden_dim=self.state_size,
            dropout=self.dropout,
            context_dim=self.state_size)

        static_covariate_encoder = GatedResidualNetwork(input_dim=self.state_size,
                                                        hidden_dim=self.state_size,
                                                        output_dim=self.state_size,
                                                        dropout=self.dropout)
        self.static_encoder_selection = copy.deepcopy(static_covariate_encoder)
        self.static_encoder_enrichment = copy.deepcopy(static_covariate_encoder)
        self.static_encoder_sequential_cell_init = copy.deepcopy(static_covariate_encoder)
        self.static_encoder_sequential_state_init = copy.deepcopy(static_covariate_encoder)

        self.past_lstm = nn.LSTM(input_size=self.state_size,
                                 hidden_size=self.state_size,
                                 num_layers=self.lstm_layers,
                                 dropout=self.dropout,
                                 batch_first=True)

        self.future_lstm = nn.LSTM(input_size=self.state_size,
                                   hidden_size=self.state_size,
                                   num_layers=self.lstm_layers,
                                   dropout=self.dropout,
                                   batch_first=True)

        self.post_lstm_gating = GateAddNorm(input_dim=self.state_size, dropout=self.dropout)

        self.static_enrichment_grn = GatedResidualNetwork(input_dim=self.state_size,
                                                          hidden_dim=self.state_size,
                                                          output_dim=self.state_size,
                                                          context_dim=self.state_size,
                                                          dropout=self.dropout)

        self.multihead_attn = InterpretableMultiHeadAttention(embed_dim=self.state_size,num_heads=self.attention_heads)
        self.post_attention_gating = GateAddNorm(input_dim=self.state_size, dropout=self.dropout)

        self.pos_wise_ff_grn = GatedResidualNetwork(input_dim=self.state_size,
                                                    hidden_dim=self.state_size,
                                                    output_dim=self.state_size,
                                                    dropout=self.dropout)
        self.pos_wise_ff_gating = GateAddNorm(input_dim=self.state_size, dropout=None)
        self.output_layer = nn.Linear(self.state_size, self.num_outputs*self.num_features_to_predict)

    def apply_temporal_selection(self, temporal_representation: torch.tensor,
                                 static_selection_signal: torch.tensor,
                                 temporal_selection_module: VariableSelectionNetwork
                                 ) -> Tuple[torch.tensor, torch.tensor]:
        num_samples, num_temporal_steps, _ = temporal_representation.shape

        time_distributed_context = self.replicate_along_time(static_signal=static_selection_signal,
                                                             time_steps=num_temporal_steps)
        # time_distributed_context: [num_samples x num_temporal_steps x state_size]
        # temporal_representation: [num_samples x num_temporal_steps x (total_num_temporal_inputs * state_size)]

        temporal_flattened_embedding = self.stack_time_steps_along_batch(temporal_representation)
        time_distributed_context = self.stack_time_steps_along_batch(time_distributed_context)
        # temporal_flattened_embedding: [(num_samples * num_temporal_steps) x (total_num_temporal_inputs * state_size)]
        # time_distributed_context: [(num_samples * num_temporal_steps) x state_size]

        temporal_selection_output, temporal_selection_weights = temporal_selection_module(
            flattened_embedding=temporal_flattened_embedding, context=time_distributed_context)
        # temporal_selection_output: [(num_samples * num_temporal_steps) x state_size]
        # temporal_selection_weights: [(num_samples * num_temporal_steps) x (num_temporal_inputs) x 1]

        temporal_selection_output = temporal_selection_output.view(num_samples, num_temporal_steps, -1)
        temporal_selection_weights = temporal_selection_weights.squeeze(-1).view(num_samples,num_temporal_steps,-1)
        # temporal_selection_output: [num_samples x num_temporal_steps x state_size)]
        # temporal_selection_weights: [num_samples x num_temporal_steps x num_temporal_inputs)]

        return temporal_selection_output, temporal_selection_weights

    @staticmethod
    def replicate_along_time(static_signal: torch.tensor, time_steps: int) -> torch.tensor:
        time_distributed_signal = static_signal.unsqueeze(1).repeat(1, time_steps, 1)
        return time_distributed_signal

    @staticmethod
    def stack_time_steps_along_batch(temporal_signal: torch.tensor) -> torch.tensor:
        return temporal_signal.view(-1, temporal_signal.size(-1))

    def transform_inputs(self, batch: Dict[str, torch.tensor]) -> Tuple[torch.tensor, ...]:
        empty_tensor = torch.empty((0, 0))
        
        static_rep = self.static_transform(x_numeric=batch.get('static_feats_numeric', empty_tensor),
                                           x_categorical=batch.get('static_feats_categorical', empty_tensor))
        historical_ts_rep = self.historical_ts_transform(x_numeric=batch.get('historical_ts_numeric', empty_tensor),
                                                         x_categorical=batch.get('historical_ts_categorical',
                                                                                 empty_tensor))
        future_ts_rep = self.future_ts_transform(x_numeric=batch.get('future_ts_numeric', empty_tensor),
                                                 x_categorical=batch.get('future_ts_categorical', empty_tensor))
        return future_ts_rep, historical_ts_rep, static_rep

    def get_static_encoders(self, selected_static: torch.tensor) -> Tuple[torch.tensor, ...]:
        c_selection = self.static_encoder_selection(selected_static)
        c_enrichment = self.static_encoder_enrichment(selected_static)
        c_seq_hidden = self.static_encoder_sequential_state_init(selected_static)
        c_seq_cell = self.static_encoder_sequential_cell_init(selected_static)
        return c_enrichment, c_selection, c_seq_cell, c_seq_hidden

    def apply_sequential_processing(self, selected_historical: torch.tensor, selected_future: torch.tensor,
                                    c_seq_hidden: torch.tensor, c_seq_cell: torch.tensor) -> torch.tensor:
        lstm_input = torch.cat([selected_historical, selected_future], dim=1)
        past_lstm_output, hidden = self.past_lstm(selected_historical,
                                                  (c_seq_hidden.unsqueeze(0).repeat(self.lstm_layers, 1, 1),
                                                   c_seq_cell.unsqueeze(0).repeat(self.lstm_layers, 1, 1)))

        future_lstm_output, _ = self.future_lstm(selected_future, hidden)
        lstm_output = torch.cat([past_lstm_output, future_lstm_output], dim=1)

        gated_lstm_output = self.post_lstm_gating(lstm_output, residual=lstm_input)
        return gated_lstm_output

    def apply_static_enrichment(self, gated_lstm_output: torch.tensor,
                                static_enrichment_signal: torch.tensor) -> torch.tensor:
        num_samples, num_temporal_steps, _ = gated_lstm_output.shape

        time_distributed_context = self.replicate_along_time(static_signal=static_enrichment_signal,
                                                             time_steps=num_temporal_steps)
        # time_distributed_context: [num_samples x num_temporal_steps x state_size]

        flattened_gated_lstm_output = self.stack_time_steps_along_batch(gated_lstm_output)
        time_distributed_context = self.stack_time_steps_along_batch(time_distributed_context)
        # flattened_gated_lstm_output: [(num_samples * num_temporal_steps) x state_size]
        # time_distributed_context: [(num_samples * num_temporal_steps) x state_size]

        enriched_sequence = self.static_enrichment_grn(flattened_gated_lstm_output, context=time_distributed_context)
        # enriched_sequence: [(num_samples * num_temporal_steps) x state_size]

        enriched_sequence = enriched_sequence.view(num_samples, -1, self.state_size)
        # enriched_sequence: [num_samples x num_temporal_steps x state_size]

        return enriched_sequence

    def apply_self_attention(self, enriched_sequence: torch.tensor,
                             num_historical_steps: int,
                             num_future_steps: int):
        output_sequence_length = num_future_steps - self.target_window_start_idx
        mask = torch.cat([torch.zeros(output_sequence_length,
                                      num_historical_steps + self.target_window_start_idx,
                                      device=enriched_sequence.device),
                          torch.triu(torch.ones(output_sequence_length, output_sequence_length,
                                                device=enriched_sequence.device),
                                     diagonal=1)], dim=1)
        # mask: [output_sequence_length x (num_historical_steps + num_future_steps)]

        post_attention, attention_outputs, attention_scores = self.multihead_attn(
            q=enriched_sequence[:, (num_historical_steps + self.target_window_start_idx):, :],
            k=enriched_sequence,
            v=enriched_sequence,
            mask=mask.bool())
        # post_attention: [num_samples x num_future_steps x state_size]
        # attention_outputs: [num_samples x num_future_steps x state_size]
        # attention_scores: [num_samples x num_future_steps x num_total_steps]

        gated_post_attention = self.post_attention_gating(x=post_attention,
                                                          residual=enriched_sequence[:, (num_historical_steps + self.target_window_start_idx):, :])
        # gated_post_attention: [num_samples x num_future_steps x state_size]

        return gated_post_attention, attention_scores

    def forward(self, batch):
        num_samples, num_historical_steps, _ = batch[self.historical_ts_representative_key].shape
        num_future_steps = batch[self.future_ts_representative_key].shape[1]
        
        future_ts_rep, historical_ts_rep, static_rep = self.transform_inputs(batch)
        # static_rep: [num_samples x (total_num_static_inputs * state_size)]
        # historical_ts_rep: [num_samples x num_historical_steps x (total_num_historical_inputs * state_size)]
        # future_ts_rep: [num_samples x num_future_steps x (total_num_future_inputs * state_size)]
                
        selected_static, static_weights = self.static_selection(static_rep)
        # selected_static: [num_samples x state_size]
        # static_weights: [num_samples x num_static_inputs x 1]
                
        c_enrichment, c_selection, c_seq_cell, c_seq_hidden = self.get_static_encoders(selected_static)
        # [num_samples x state_size]

        selected_historical, historical_selection_weights = self.apply_temporal_selection(
            temporal_representation=historical_ts_rep,
            static_selection_signal=c_selection,
            temporal_selection_module=self.historical_ts_selection)
        # selected_historical: [num_samples x num_historical_steps x state_size]
        # historical_selection_weights: [num_samples x num_historical_steps x total_num_historical_inputs]
                
        selected_future, future_selection_weights = self.apply_temporal_selection(
            temporal_representation=future_ts_rep,
            static_selection_signal=c_selection,
            temporal_selection_module=self.future_ts_selection)
        # selected_future: [num_samples x num_future_steps x state_size]
        # future_selection_weights: [num_samples x num_future_steps x total_num_future_inputs]
                
        gated_lstm_output = self.apply_sequential_processing(selected_historical=selected_historical,
                                                             selected_future=selected_future,
                                                             c_seq_hidden=c_seq_hidden,
                                                             c_seq_cell=c_seq_cell)
        # gated_lstm_output : [num_samples x (num_historical_steps + num_future_steps) x state_size]

        enriched_sequence = self.apply_static_enrichment(gated_lstm_output=gated_lstm_output,
                                                         static_enrichment_signal=c_enrichment)
        # enriched_sequence: [num_samples x (num_historical_steps + num_future_steps) x state_size]
        gated_post_attention, attention_scores = self.apply_self_attention(enriched_sequence=enriched_sequence,
                                                                           num_historical_steps=num_historical_steps,
                                                                           num_future_steps=num_future_steps)

        post_poswise_ff_grn = self.pos_wise_ff_grn(gated_post_attention)
        gated_poswise_ff = self.pos_wise_ff_gating(
            post_poswise_ff_grn,
            residual=gated_lstm_output[:, (num_historical_steps + self.target_window_start_idx):, :])
        # gated_poswise_ff: [num_samples x output_sequence_length x state_size]
                
       predicted_quantiles = self.output_layer(gated_poswise_ff)
        # predicted_quantiles: [num_samples x num_future_steps x (num_quantilesxnum_features_predict)]

        return {
            'predicted_quantiles': predicted_quantiles,  # [num_samples x output_sequence_length x num_feature_pred x num_quantiles]
            'static_weights': static_weights.squeeze(-1),  # [num_samples x num_static_inputs]
            'historical_selection_weights': historical_selection_weights,
            # [num_samples x num_historical_steps x total_num_historical_inputs]
            'future_selection_weights': future_selection_weights,
            # [num_samples x num_future_steps x total_num_future_inputs]
            'attention_scores': attention_scores
            # [num_samples x output_sequence_length x (num_historical_steps + num_future_steps)]
        }