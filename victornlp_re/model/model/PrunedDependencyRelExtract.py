"""
@module PrunedDependencyRelExtract

Implements RE model that runs GCN on pruned dependency tree.
> Y. Zhang. et al.(2018) Graph Convolution over Pruned Dependency Trees Improves Relation Extraction
"""

import re

import torch
import torch.nn as nn

from . import register_model

from ...victornlp_utils.module import BilinearAttention, GCN

@register_model('pruned-dependency')
class PrunedDependencyRelExtract(nn.Module):
  """
  @class PrunedDependencyRelExtract

  GCN-over-dependency-tree model for Relation Extraction.
  """

  def __init__(self, embeddings, labels, config):
    """
    Constructor for Deep-Biaffine parser.
    
    @param self The object pointer.
    @param embeddings List of Embedding-like objects. Refer to 'embedding.py' for more details.
    @param labels Dictionary of lists.
    @param config Dictionary config file accessed with 'mtre-sentence' key.
    """
    super(PrunedDependencyRelExtract, self).__init__()

    # Embedding layer
    self.embeddings = nn.ModuleList(embeddings)
    input_size = 0
    for embedding in embeddings:
      input_size += embedding.embed_size
    self.input_size = input_size

    # Bi-LSTM encoder
    self.lstm_hidden_size = config['lstm_hidden_size']
    self.lstm_layers = config['lstm_layers']
    self.encoder = nn.LSTM(input_size=self.input_size, hidden_size=self.lstm_hidden_size, num_layers=self.lstm_layers, batch_first=True, bidirectional=True)

    # - GCN layer
    self.gcn_hidden_size = config['gcn_hidden_size']
    self.gcn_layers = config['gcn_layers']
    self.gcn = GCN(self.lstm_hidden_size * 2, self.gcn_hidden_size, self.gcn_layers)

    # - Prediction Layer
    self.relation_type_size = config['relation_type_size']
    self.re_labels = labels['relation_labels'] # Relation Type Labels
    self.re_labels_stoi = {}
    for i, label in enumerate(self.re_labels):
      self.re_labels_stoi[label] = i
    self.prediction = nn.Sequential(
      nn.Linear(self.gcn_hidden_size * 3, self.relation_type_size),
      nn.ReLU(),
      nn.Linear(self.relation_type_size, len(self.re_labels)),
      nn.LogSoftmax()
    )
  
  def run(self, inputs, **kwargs):
    """
    Runs the model and obtain softmax-ed distribution for each possible labels.
    
    @param self The object pointer.
    @param inputs List of dictionaries. Refer to 'dataset.py' for more details.
    @return relations List, inputs[i]['relations'] flattened. len(relations) == total_relations.
    @return relation_scores Tensor(total_relations, len(relation_labels)). Softmax-ed to obtain label probabilities.
    """
    batch_size = len(inputs)
    device = next(self.parameters()).device

    max_length = -1
    if 'lengths' not in kwargs:
      lengths = torch.zeros(batch_size, dtype=torch.long, device=device).detach()
      for i, input in enumerate(inputs):
        lengths[i] = input['word_count'] + 1 # DP requires [ROOT] node; index 0 is given
        if max_length < input['word_count'] + 1:
          max_length = input['word_count'] + 1
    else:
      lengths = kwargs['lengths']
      assert lengths.dim()==1 and lengths.size(0) == batch_size
      lengths = lengths.detach()
    if 'mask' not in kwargs:
      mask = torch.zeros(batch_size, 1, max_length, max_length, device=device).detach()
      for i, length in enumerate(lengths):
        for j in range(1, length):
          mask[i, 0, j, :length] = 1
    else:
      mask = kwargs['mask']
      assert mask.size(0) == batch_size and mask.size(2) == mask.size(3) == max_length
      mask = mask.detach()
    
    # Adjacent matrix and entity_embeddings_index
    adj_matrix = torch.zeros(batch_size, max_length, max_length, device=device).detach()
    entities = []
    # entity_embeddings_index: List[ List[begin, end] -> named_entities ] -> batch
    entity_embeddings_index = []

    for i, input in enumerate(inputs):
      assert 'dependency' in input and "Input must include golden dependency"
      for arc in input['dependency']:
        adj_matrix[i, arc['dep'], arc['head']] = 1

      assert 'named_entity' in input
      entity_embeddings_index.append([])
      for entity in input['named_entity']:
        entities.append(entity)
        entity_embeddings_index[i].append([])

        # Find all indices of spaces
        space_indices = [m.start() for m in re.finditer(' ', input['text'])] + [len(input['text'])]
        assert len(space_indices) == input['word_count']
        for m_i, m in enumerate(space_indices):
          if entity['begin'] >= m:
            entity_embeddings_index[i][-1].append(m_i + 1)
          if entity['end'] <= m:
            entity_embeddings_index[i][-1].append(m_i + 1)


    # Embedding
    embedded = []
    for embedding in self.embeddings:
      embedded.append(embedding(inputs))
    embedded = torch.cat(embedded, dim=2)
    
    # Run LSTM encoder first
    lstm_output, _ = self.encoder(embedded)

    # Relation Extraction
    relations = []
    subject_embeddings = []
    predicate_embeddings = []
    context_embeddings = []

    # GCN-contextualized embeddings
    gcn_output = self.gcn(adj_matrix, lstm_output)

    # Max-pooling to get sentence vector
    mask = mask[:, 0, :, 0].unsqueeze(2)
    context_vector, _ = torch.max(gcn_output, dim=1)
    # context_vector: Tensor(batch, gcn_hidden)

    # Max-pooling to get entity vector
    def max_pool_entity_vec(batch, begin_end):
      """
      Max-pooling for entities from GCN output.
      @param batch Integer that indicates batch ID
      @param begin_end List of two integers, indicating the begin and last word-phrase of the entity.
      @return entity_embedding Tensor[gcn_hidden_size]. Max-pooled entity embedding
      """
      entity = gcn_output[i, begin_end[0]:begin_end[1] + 1]
      entity_embedding, _ = torch.max(entity, dim=0)
      return entity_embedding

    # Prediction
    for i, input in enumerate(inputs):
      for relation in input['relation']:
        relations.append(relation)
        
        subject_embeddings.append(max_pool_entity_vec(i, entity_embeddings_index[i][relation['subject']]).unsqueeze(0))
        predicate_embeddings.append(max_pool_entity_vec(i, entity_embeddings_index[i][relation['predicate']]).unsqueeze(0))
        context_embeddings.append(context_vector[i].unsqueeze(0))

    subject_embeddings = torch.cat(subject_embeddings, dim=0)
    predicate_embeddings = torch.cat(predicate_embeddings, dim=0)
    context_embeddings = torch.cat(context_embeddings, dim=0)

    prediction_input = torch.cat([subject_embeddings, predicate_embeddings, context_embeddings], dim=1)
    
    relation_scores = self.prediction(prediction_input)

    return relations, relation_scores
