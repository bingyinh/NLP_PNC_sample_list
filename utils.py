from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
# modified from https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Custom_Named_Entity_Recognition_with_BERT_only_first_wordpiece.ipynb#scrollTo=jrkdZBLYHVcB
class dataset(Dataset):
    def __init__(self, tokens, tags, tokenizer, max_len):
        self.len = len(tokens)
        self.tokens = tokens
        self.tags = tags
        self.tokenizer = tokenizer
        self.max_len = max_len
        # create encodings for tokens and labels
        self.unique_tags = set(tag for doc in tags for tag in doc)
        self.tag2id = {tag: _id for _id, tag in enumerate(self.unique_tags)}
        self.id2tag = {_id: tag for tag, _id in self.tag2id.items()}

    def __getitem__(self, index):
        # step 1: get the sentence and word labels (skip, we already have it)

        # step 2: use tokenizer to encode sentence (includes padding/truncation up to max length)
        # BertTokenizerFast provides a handy "return_offsets_mapping" functionality for individual tokens
        encoding = self.tokenizer(self.tokens[index],
                                  is_split_into_words=True,
                                  return_offsets_mapping=True,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_len)
        
        # step 3: create token labels only for first word pieces of each tokenized word
        tags = [self.tag2id[tag] for tag in self.tags[index]]
        # code based on https://huggingface.co/transformers/custom_datasets.html#tok-ner
        # create an empty array of -100 of length max_length
        encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
        
        # the original label aligning approach does not work due to a bug in offset_mapping
        # generation by the DeBERTa and RoBERTa fast tokenizer.
#         # set only labels whose first offset position is 0 and the second is not 0
#         i = 0
#         for idx, mapping in enumerate(encoding["offset_mapping"]):
#             if mapping[0] == 0 and mapping[1] != 0:
#                 # overwrite label
#                 encoded_labels[idx] = tags[i]
#                 i += 1

        # use instead tokenize function for each pre-tokenized token to find out how long
        # it becomes after tokenized by the tokenizer and align label accordingly
        i = 1 # 0 is for the prefix space (0,0), start from 1
        for idx, token in enumerate(self.tokens[index]):
            # only update the label of the first subtoken
            encoded_labels[i] = tags[idx]
            # check the number of subtokens of current token
            # and move i to the first subtoken of next token
            i += len(self.tokenizer.tokenize(token))

        # step 4: turn everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(encoded_labels)
        
        return item

    def __len__(self):
        return self.len

def load_conll(file):
    texts, tags = [],[]
    with open(file,'r') as f:
        text, tag = [],[]
        for l in f:
            if l == '\n':
                texts.append(text)
                tags.append(tag)
                text, tag = [], []
            else:
                tx,tg = l[:-1].split('\t') # ignore the tailing '\n'
                text.append(tx)
                tag.append(tg)
    return texts, tags

from transformers import Seq2SeqTrainer

class Seq2SeqTrainerGenKwargs(Seq2SeqTrainer):
    def __init__(self, bad_words_ids=None, force_words_ids=None, *args, **kwargs):
        self.bad_words_ids = bad_words_ids if bad_words_ids else None
        self.force_words_ids = force_words_ids if force_words_ids else None
        super().__init__(*args, **kwargs)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval'):
        gen_kwargs = {}
        if self.bad_words_ids:
            gen_kwargs['bad_words_ids'] = self.bad_words_ids
        if self.force_words_ids:
            gen_kwargs['force_words_ids'] = self.force_words_ids
        return super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix, **gen_kwargs)

# Use constructor to store my gen_kwargs
# update evaluate() and predict() to use those gen_kwargs without being passed through the input args