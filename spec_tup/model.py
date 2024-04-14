from typing import Dict, List, Optional, Any, Union
from overrides import overrides
import torch
from torch.nn.modules import Linear, Dropout, Embedding
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertModel
import numpy as np
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import Metric, CategoricalAccuracy, F1Measure
from allennlp.nn.util import batched_index_select
from transformers import RobertaModel, ElectraModel
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder


@Model.register("spec_bert_tup_binary_dep")
class SrlBert(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 bert_model: str,
                 tup_attention: Seq2SeqEncoder or TextFieldEmbedder,
                 embedding_dropout: float = 0.0,
                 tro: float = 1,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 label_smoothing: float = None,
                 tuple_metric: Metric = None) -> None:
        super(SrlBert, self).__init__(vocab, regularizer)

        self.plm_model = bert_model

        if 'roberta' in self.plm_model:
            self.roberta_model = RobertaModel.from_pretrained(self.plm_model)
            self.plm_features = self.roberta_model.config.hidden_size
            self.verb_embd = Embedding(2, self.plm_features, padding_idx=0)
        elif 'electra' in self.plm_model:
            options_name = "google/electra-base-discriminator"
            self.electra_model = ElectraModel.from_pretrained(options_name)
            self.plm_features = self.electra_model.config.hidden_size
            self.verb_embd = Embedding(2, self.plm_features, padding_idx=0)
        else:
            self.bert_model = BertModel.from_pretrained(self.plm_model)
            self.plm_features = self.bert_model.config.hidden_size


        self.embedding_dropout = Dropout(p=embedding_dropout)
        self._label_smoothing = label_smoothing
        self._spec_metric = tuple_metric
        self._composition_layer = torch.nn.Linear(self.plm_features, self.plm_features)
        self._sent_spec_layer = torch.nn.Linear(self.plm_features, 2)
        self._sent_spec_loss = torch.nn.CrossEntropyLoss()
        self._tup_spec_scores = torch.nn.Linear(self.plm_features, 2)
        self._tup_spec_loss = torch.nn.CrossEntropyLoss()
        self._tup_adjustment = self.compute_tup_adjustment(tro=tro)
        self._sent_adjustment = self.compute_sent_adjustment(tro=tro)
        self.tup_embedding = Embedding(18, 768, padding_idx=0)
        self.dep_embedding = Embedding(43, 768, padding_idx=0)
        self.tup_attention = tup_attention
        initializer(self)

    def compute_tup_adjustment(self, tro=1):
        labels = {'N/A': 90136, 'can': 4258, 'might': 2894, 'will': 1412, 'should': 1037, 'would': 880, 'had': 549}
        label_val = sorted(labels.values())
        label_freq_array = np.array(list(label_val))
        label_freq_array = label_freq_array / label_freq_array.sum()
        adjustments = np.log(label_freq_array ** tro + 1e-12)
        adjustments = torch.from_numpy(adjustments)
        adjustments = adjustments.to(0)
        return adjustments

    def compute_sent_adjustment(self, tro=1):
        labels = {'0': 90136, '1': 11030}
        label_val = sorted(labels.values())
        label_freq_array = np.array(list(label_val))
        label_freq_array = label_freq_array / label_freq_array.sum()
        adjustments = np.log(label_freq_array ** tro + 1e-12)
        adjustments = torch.from_numpy(adjustments)
        adjustments = adjustments.to(0)
        return adjustments

    def forward(self, # type: ignore
                tokens: Dict[str, torch.Tensor],
                verb_indicator: torch.Tensor,
                dep_nodes: Dict[str, torch.Tensor],
                pos_nodes: Dict[str, torch.Tensor],
                tup_seq: Dict[str, torch.Tensor],
                mask_ids: torch.LongTensor,
                metadata: List[Any],
                sent_spec_labels: torch.LongTensor = None,
                tup_spec_labels: torch.LongTensor = None,
                optimizer = None):

        mask = get_text_field_mask(tokens)

        if 'roberta' in self.plm_model:
            mask[:, 0] = 1
            verb_embed = self.verb_embd(verb_indicator)
            outputs = self.roberta_model(input_ids=tokens["tokens"], attention_mask=mask, encoder_hidden_states=False)
            plm_embeddings, pooled = outputs.last_hidden_state, outputs.pooler_output
            plm_embeddings = plm_embeddings + verb_embed
        elif 'electra' in self.plm_model:
            outputs = self.electra_model(input_ids=tokens["tokens"], token_type_ids=verb_indicator, attention_mask=mask, output_hidden_states=False)
            plm_embeddings = outputs.last_hidden_state
            pooled = plm_embeddings[:, 0, :]
        else:
            plm_embeddings, pooled = self.bert_model(input_ids=tokens["tokens"], token_type_ids=verb_indicator,
                                                     attention_mask=mask, output_all_encoded_layers=False)

        pooled = self.embedding_dropout(pooled)
        plm_embeddings = self.embedding_dropout(plm_embeddings)
        dep_label_embeddings = self.dep_embedding(dep_nodes['dep_tags'])
        batch_size, dep_seq_length, embed_dim = dep_label_embeddings.size()

        cuda_device = dep_label_embeddings.get_device()
        if cuda_device < 0:
            dep_embeddings = torch.zeros([batch_size, dep_seq_length, embed_dim], dtype=torch.float32)
        else:
            dep_embeddings = torch.zeros([batch_size, dep_seq_length, embed_dim], dtype=torch.float32, device=cuda_device)

        for i in range(0, batch_size):
            offset_start = metadata[i]['offsets']
            offset_end = metadata[i]['end_offsets']
            plm_embedding = plm_embeddings[i, :, :]
            dep_index_list = metadata[i]["dep_nodes_index"]

            for j, a in enumerate(dep_index_list):
                x, y = offset_start[a], offset_end[a]
                if x == y:
                    dep_embeddings[i, j, :] = plm_embedding[x, :]
                else:
                    dep_embeddings[i, j, :] = torch.mean(plm_embedding[x:y + 1, :], dim=0)

        tuple_logits = self._tup_spec_scores(pooled)
        tuple_probs = torch.nn.functional.softmax(tuple_logits, dim=-1)

        output_dict = {"tuple_logits": tuple_logits, "tuple_probs": tuple_probs}

        if sent_spec_labels is not None and tup_spec_labels is not None:
            output_dict["loss"] = self._tup_spec_loss(tuple_logits, tup_spec_labels.long().view(-1))

        if metadata[0]['validation']:
            output_dict = self.decode(output_dict)
            # think about how to get confidence score
            sent_list, sent_spec_labels, sent_spec_levels, tup_spec_labels, tup_spec_levels = [], [], [], [], []
            for x in metadata:
                sent = ' '.join(x['words'])
                sent_list.append(sent)
                sent_spec_labels.append(x['sent_spec'])
                tup_spec_labels.append(x['tup_spec'])

                # get difficulty level for sent and tuple
                if 'sent_spec_level' in x.keys():
                    sent_spec_levels.append(x['sent_spec_level'])
                else:
                    sent_spec_levels.append('N/A')
                if 'tup_spec_level' in x.keys():
                    tup_spec_levels.append(x['tup_spec_level'])
                else:
                    tup_spec_levels.append('N/A')

            tup_spec_preds = output_dict['tup_label']

            self._spec_metric(sent=sent_list,
                              sent_spec_gold=sent_spec_labels, sent_spec_pred='test',
                              sent_spec_level=sent_spec_levels,
                              tup_spec_level=tup_spec_levels,
                              tup_spec_gold=tup_spec_labels, tup_spec_pred=tup_spec_preds
                              )

        return output_dict


    @overrides
    def decode(self, output_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the probabilities, converts index to string label, and
        add ``"label"`` key to the dictionary with the result.
        """

        tuple_predictions = output_dict["tuple_probs"]
        if tuple_predictions.dim() == 2:
            tuple_predictions_list = [tuple_predictions[i] for i in range(tuple_predictions.shape[0])]
        else:
            tuple_predictions_list = [tuple_predictions]
        tuple_classes = []
        for tuple_prediction in tuple_predictions_list:
            tuple_label_idx = tuple_prediction.argmax(dim=-1).item()
            tuple_classes.append(tuple_label_idx)
        output_dict["tup_label"] = tuple_classes

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if not self.training:
            if self._spec_metric is not None:
                all_metrics.update(self._spec_metric.get_metric(reset=reset))  # type: ignore
        return all_metrics


    def get_viterbi_pairwise_potentials(self):
        """
        Generate a matrix of pairwise transition potentials for the BIO labels.
        The only constraint implemented here is that I-XXX labels must be preceded by either an identical I-XXX tag or
        a B-XXX tag. In order to achieve this constraint, pairs of labels which do not satisfy this constraint have a
        pairwise potential of -inf.
        Returns
        -------
        transition_matrix : torch.Tensor
            A (num_labels, num_labels) matrix of pairwise potentials.
        """
        all_labels = self.vocab.get_index_to_token_vocabulary("labels")
        num_labels = len(all_labels)
        transition_matrix = torch.zeros([num_labels, num_labels])

        for i, previous_label in all_labels.items():
            for j, label in all_labels.items():
                # I labels can only be preceded by themselves or their corresponding B tag.
                if i != j and label[0] == 'I' and not previous_label == 'B' + label[1:]:
                    transition_matrix[i, j] = float("-inf")
        return transition_matrix


    def get_start_transitions(self):
        """
        In the BIO sequence, we cannot start the sequence with an I-XXX tag.
        This transition sequence is passed to viterbi_decode to specify this constraint.
        Returns
        -------
        start_transitions : torch.Tensor
            The pairwise potentials between a START token and
            the first token of the sequence.
        """
        all_labels = self.vocab.get_index_to_token_vocabulary("labels")
        num_labels = len(all_labels)

        start_transitions = torch.zeros(num_labels)

        for i, label in all_labels.items():
            if label[0] == "I":
                start_transitions[i] = float("-inf")

        return start_transitions
