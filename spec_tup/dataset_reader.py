import logging
from typing import Dict, List, Iterable, Tuple, Any

from overrides import overrides
from pytorch_pretrained_bert.tokenization import BertTokenizer

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField, LabelField, IndexField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from spec_tup.dataset_helper import convert_verb_indices_to_wordpiece_indices, \
    convert_tags_to_wordpiece_tags

from transformers import RobertaTokenizer, ElectraTokenizer

import json

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("spec_bert_tup_binary_dep")
class SrlReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 domain_identifier: str = None,
                 lazy: bool = False,
                 validation: bool = False,
                 data_type: str = 'oo',
                 verbal_indicator: bool = True,
                 tup_labels_file: str = "config/vocab/tuple_labels.txt",
                 dep_labels_file: str = "config/vocab/dep_labels.txt",
                 pos_labels_file: str = "config/vocab/pos_labels.txt",
                 bert_model_name: str = None) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._tup_tag_indexers = {"tup_tags": SingleIdTokenIndexer(namespace="tuple_labels")}
        self._dep_tag_indexers = {"dep_tags": SingleIdTokenIndexer(namespace="dep_labels")}
        self._pos_tag_indexers = {"pos_tags": SingleIdTokenIndexer(namespace="pos_labels")}
        self._tup_tag_vocab = {line.strip(): i for i, line in enumerate(open(tup_labels_file, 'r'))}
        self._dep_tag_vocab = {line.strip(): i for i, line in enumerate(open(dep_labels_file, 'r'))}
        self._pos_tag_vocab = {line.strip(): i for i, line in enumerate(open(pos_labels_file, 'r'))}
        self._domain_identifier = domain_identifier
        self._validation = validation
        self._verbal_indicator = verbal_indicator
        self._data_type = data_type
        self._special_token = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]", '[ARG0]', '[ARG1]', '[ARG2]', '[ARG3]', '[ARG4]', '[ARG5]', '[REL]']

        self._model_name = bert_model_name


        if 'roberta' in self._model_name:
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base", add_prefix_space=True)
        elif 'electra' in self._model_name:
            options_name = "google/electra-base-discriminator"
            self.tokenizer = ElectraTokenizer.from_pretrained(options_name, do_basic_tokenize=True, never_split=["[MASK]"])
            self.lowercase_input = True
        else:
            self.tokenizer = BertTokenizer.from_pretrained(bert_model_name, never_split=(self._special_token))
            self.lowercase_input = "uncased" in bert_model_name

    def _wordpiece_tokenize_input(self, tokens: List[str]) -> Tuple[List[str], List[int], List[int]]:
        word_piece_tokens: List[str] = []
        end_offsets = []
        start_offsets = []
        cumulative = 0
        for token in tokens:
            if self.lowercase_input and token not in self._special_token:
                token = token.lower()
            word_pieces = self.tokenizer.wordpiece_tokenizer.tokenize(token)
            start_offsets.append(cumulative + 1)
            cumulative += len(word_pieces)
            end_offsets.append(cumulative)
            word_piece_tokens.extend(word_pieces)
        wordpieces = ["[CLS]"] + word_piece_tokens + ["[SEP]"]
        return wordpieces, end_offsets, start_offsets

    def _wordpiece_tokenize_input_roberta(self, tokens: List[str]) -> Tuple[List[str], List[int], List[int]]:
        word_piece_tokens: List[str] = []
        end_offsets = []
        start_offsets = []
        cumulative = 0
        for token in tokens:
            word_pieces = self.tokenizer.tokenize(token)
            start_offsets.append(cumulative + 1)
            cumulative += len(word_pieces)
            end_offsets.append(cumulative)
            word_piece_tokens.extend(word_pieces)
        wordpieces = ["<s>"] + word_piece_tokens + ["</s>"]
        return wordpieces, end_offsets, start_offsets


    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        logger.info("Reading instances from dataset files at: %s", file_path)
        if self._domain_identifier is not None:
            logger.info("Filtering to only include file paths containing the %s domain", self._domain_identifier)

        file_in = open(file_path, 'r', encoding='utf-8')
        json_sent = json.load(file_in)
        file_in.close()
        n = 0
        for sent, sent_val in json_sent.items():
            tokens = [Token(x) for x in sent_val['tokens']]
            tuples = sent_val['tuples']
            syntax_info = {'spacy_pos': sent_val['spacy_pos'], 'spacy_tag': sent_val['spacy_tag'],
                           'dep_graph_edges': sent_val['dep_graph_edges']}

            n += 1
            # if n > 1000:
            #     continue
            if self._validation:
                sent_spec = sent_val['sent_spec']
                if 'sent_spec_level' in sent_val.keys():
                    sent_spec_level = sent_val['sent_spec_level']
                    for tup in tuples:
                        yield self.text_to_instance(tokens, tup, sent_spec=sent_spec, sent_spec_level=sent_spec_level, dep=syntax_info)
                else:
                    for tup in tuples: yield self.text_to_instance(tokens, tup, sent_spec=sent_spec, dep=syntax_info)
            else:
                if len(tokens) > 120: continue
                sent_spec = sent_val['sent_spec']
                for tup in tuples:
                    yield self.text_to_instance(tokens, tup, sent_spec=sent_spec, dep=syntax_info)



    def text_to_instance(self, tokens: List[Token], tup: Dict[str, Any], sent_spec: int = None,
                         sent_spec_level: str = None, oie_tags: List[str] = None, dep: Dict[str, Any] = None) -> Instance:

        metadata_dict: Dict[str, Any] = {}
        sent_tokens = [t.text for t in tokens]
        verb_label = tup['verb_label']
        verb_index = verb_label.index(1)
        verb = tokens[verb_index].text
        tup_tags = tup['tags']

        spacy_pos = dep['spacy_pos']
        spacy_tag = dep['spacy_tag']
        dep_graph_edges = dep['dep_graph_edges']

        kept_edges, connected_nodes = [], []
        kept_edges2, connected_nodes2 = [], []
        for (edge_i, edge_j), dep_tag in dep_graph_edges:
            # the siblings of the verb node
            if edge_i == verb_index and edge_j < verb_index:
                kept_edges.append(((edge_i, edge_j), dep_tag))
                connected_nodes.append((edge_j, dep_tag, spacy_pos[edge_j], spacy_tag[edge_j]))

            # the parent of the verb node
            if edge_j == verb_index and edge_i < verb_index:
                kept_edges.append(((edge_i, edge_j), dep_tag))
                connected_nodes.append((edge_i, dep_tag, spacy_pos[edge_i], spacy_tag[edge_i]))

                # get the siblings of the parent node.
                if spacy_pos[edge_i] == 'VERB':
                    for (i, j), dep_tag_2 in dep_graph_edges:
                        if i == edge_i and j < verb_index:
                            kept_edges2.append(((i, j), dep_tag_2))
                            connected_nodes2.append((j, dep_tag_2, spacy_pos[j], spacy_tag[j]))


        dep_seq = [x for _, x, _, _ in connected_nodes]
        dep_field = TextField([Token(t, text_id=self._dep_tag_vocab[t]) for t in dep_seq], token_indexers=self._dep_tag_indexers)
        pos_seq = [x for _, _, x, _ in connected_nodes]
        pos_field = TextField([Token(t, text_id=self._pos_tag_vocab[t]) for t in pos_seq], token_indexers=self._pos_tag_indexers)

        if 'roberta' in self._model_name:
            wordpieces, offsets, start_offsets = self._wordpiece_tokenize_input_roberta(sent_tokens)
        else:
            wordpieces, offsets, start_offsets = self._wordpiece_tokenize_input(sent_tokens)



        new_verbs = convert_verb_indices_to_wordpiece_indices(verb_label, offsets)
        metadata_dict["offsets"] = start_offsets
        metadata_dict["end_offsets"] = offsets

        dep_node_seq = [x for x, _, _, _ in connected_nodes]
        metadata_dict["dep_nodes"] = connected_nodes
        metadata_dict["dep_nodes_index"] = dep_node_seq


        if 'roberta' in self._model_name:
            text_field = TextField([Token(t, text_id=self.tokenizer._convert_token_to_id(t)) for t in wordpieces], token_indexers=self._token_indexers)
        else:
            text_field = TextField([Token(t, text_id=self.tokenizer.vocab[t]) for t in wordpieces], token_indexers=self._token_indexers)

        verb_indicator = SequenceLabelField(new_verbs, text_field)
        spec_mask_index = start_offsets[verb_index]
        spec_mask_field = LabelField(spec_mask_index, skip_indexing=True)
        fields: Dict[str, Field] = {'tokens': text_field, 'verb_indicator': verb_indicator, 'mask_ids': spec_mask_field,
                                    'dep_nodes': dep_field, 'pos_nodes': pos_field}

        tup_seq, tup_offset_start, tup_offset_end = [], [], []
        for i, (tup_tag, offset_start, offset_end) in enumerate(zip(tup_tags, start_offsets, offsets)):
            if tup_tag == 'O':
                continue
            tup_seq.append(tup_tag)
            tup_offset_start.append(offset_start)
            tup_offset_end.append(offset_end)


        tup_seq_field = TextField([Token(t, text_id=self._tup_tag_vocab[t]) for t in tup_seq], token_indexers=self._tup_tag_indexers)
        # spec_mask2_field = LabelField(mask_id2, skip_indexing=True)
        fields['tup_seq'] = tup_seq_field
        # fields['tup_mask_ids'] = spec_mask2_field

        metadata_dict["words"] = [x.text for x in tokens]
        metadata_dict["verb"] = verb
        metadata_dict["verb_index"] = verb_index
        metadata_dict["validation"] = self._validation
        metadata_dict["sent_spec"] = sent_spec

        metadata_dict["tup_offset_start"] = tup_offset_start
        metadata_dict["tup_offset_end"] = tup_offset_end

        if sent_spec is not None:
            if self._validation is False:
                fields['sent_spec_labels'] = LabelField(sent_spec, skip_indexing=True)
            else:
                if sent_spec_level is not None:
                    metadata_dict["sent_spec_level"] = sent_spec_level

        # tuple_spec = tup['pred_str'].split()
        # if len(tuple_spec) > 1:
        #     metadata_dict["tup_spec"] = tuple_spec[0]
        # else:
        #     metadata_dict["tup_spec"] = 'N/A'

        metadata_dict["tup_spec"] = tup["tup_spec"]

        if self._validation is False:
            fields['tup_spec_labels'] = LabelField(metadata_dict["tup_spec"], skip_indexing=True)
        else:
            if 'spec_level' in tup.keys():
                metadata_dict["tup_spec_level"] = tup['spec_level']
                metadata_dict["tup_spec_type"] = tup['spec_type']


        fields["metadata"] = MetadataField(metadata_dict)
        return Instance(fields)
