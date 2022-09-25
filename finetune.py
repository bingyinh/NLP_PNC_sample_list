from typing import Any, Dict, List, Optional, Tuple, Union
import os
from glob import glob
import csv
import pandas as pd
import re
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
from transformers import (T5TokenizerFast, T5ForConditionalGeneration, Seq2SeqTrainingArguments,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer, LogitsProcessor,
                          LogitsProcessorList)
import evaluate

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

# modified from https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Custom_Named_Entity_Recognition_with_BERT_only_first_wordpiece.ipynb#scrollTo=jrkdZBLYHVcB
class T5Dataset(Dataset):
    def __init__(self, tokens, tags, tokenizer, max_len, task_prefix):
        self.len = len(tokens)
        self.tokens = tokens
        self.tags = tags
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.task_prefix = task_prefix
        # create encodings for tokens and labels
        self.unique_tags = set(tag for doc in tags for tag in doc)
        self.tag2id = {tag: _id for _id, tag in enumerate(self.unique_tags)}
        self.id2tag = {_id: tag for tag, _id in self.tag2id.items()}

    def __getitem__(self, index):
        # step 1: get the sentence and word labels (skip, we already have it)

        # step 2: use tokenizer to encode sentence (includes padding/truncation up to max length)
        # BertTokenizerFast provides a handy "return_offsets_mapping" functionality for individual tokens
        encoding = self.tokenizer(self.task_prefix.split() + self.tokens[index],
                                  is_split_into_words=True,
#                                   return_offsets_mapping=True,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_len)
        # step 3: use tokenizer to encode labels as a sentence
        target_encoding = self.tokenizer(self.tags[index],
                                         is_split_into_words=True,
                                         padding='max_length',
                                         truncation=True,
                                         max_length=self.max_len)
        encoded_labels = torch.as_tensor(target_encoding.input_ids)
        # step 4: turn everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        # replace padding token id's of the labels by -100 so it's ignored by the loss
        encoded_labels[encoded_labels == self.tokenizer.pad_token_id] = -100
        item['labels'] = encoded_labels
        return item

    def __len__(self):
        return self.len

# logits_processor sems to take in input_ids, next_token_scores and return next_token_scores
class TaggingLogitsProcessor(LogitsProcessor): # adapted from EncoderNoRepeatNGramLogitsProcessor
    def __init__(self, trulens, labelids, tokenizer):
        self.batch_size = trulens.size(0)
        self.tokenizer = tokenizer
        self.trulens = trulens # number of expected label tokens; only sort of cheating
        self.labelids = labelids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        num_hypos = scores.shape[0] # batch_size * beam_sz
        beam_sz = num_hypos // self.batch_size
        cur_len = input_ids.shape[-1] # note this includes bos/pad, so we want everything 1 longer
        for b in range(self.batch_size):
            if cur_len == self.trulens[b].item(): # only allow eos
                scores[b*beam_sz:(b+1)*beam_sz].fill_(-float("inf"))
                scores[b*beam_sz:(b+1)*beam_sz, self.tokenizer.eos_token_id] = 1
            elif cur_len > self.trulens[b].item(): # only allow pad
                scores[b*beam_sz:(b+1)*beam_sz].fill_(-float("inf"))
                scores[b*beam_sz:(b+1)*beam_sz, self.tokenizer.pad_token_id] = 1
            else: # only allow labels
                labelids = self.labelids.unsqueeze(0).expand(beam_sz, -1)
                labelscores = scores[b*beam_sz:(b+1)*beam_sz].gather(1, labelids)
                scores[b*beam_sz:(b+1)*beam_sz].fill_(-float("inf"))
                scores[b*beam_sz:(b+1)*beam_sz].scatter_(1, labelids, labelscores)
        return scores



class MyTrainer(Seq2SeqTrainer):
    # just hacking
    def set_things(self, labelids, tokenizer, beam_sz):
        self.labelids = labelids
        self.tokenizer = tokenizer
        self.beam_sz = beam_sz

    # adapted from https://github.com/huggingface/transformers/blob/v4.22.1/src/transformers/trainer_seq2seq.py
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.
        Subclass and override to inject custom behavior.
        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        assert has_labels
        trulens = (inputs['labels'] != -100).sum(1)

        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        #gen_kwargs = self._gen_kwargs.copy() # not sure why we don't inherit this
        gen_kwargs = {"max_length": trulens.max().item()+1, "num_beams": self.beam_sz}

        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.model.config.max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.model.config.num_beams
        )
        default_synced_gpus = False #True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)
        if "global_attention_mask" in inputs:
            gen_kwargs["global_attention_mask"] = inputs.get("global_attention_mask", None)

        # prepare generation inputs
        # some encoder-decoder models can have varying encoder's and thus
        # varying model input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        if self.labelids.device != inputs['labels'].device:
            self.labelids = self.labelids.to(inputs['labels'].device)
        gen_kwargs['logits_processor'] = LogitsProcessorList(
            [TaggingLogitsProcessor(trulens, self.labelids, self.tokenizer)])
        generated_tokens = self.model.generate(
            generation_inputs,
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if gen_kwargs.get("max_length") is not None and generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])
        elif gen_kwargs.get("max_new_tokens") is not None and generated_tokens.shape[-1] < (
            gen_kwargs["max_new_tokens"] + 1
        ):
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_new_tokens"] + 1)

        with torch.no_grad():
            with self.compute_loss_context_manager():
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if gen_kwargs.get("max_length") is not None and labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
            elif gen_kwargs.get("max_new_tokens") is not None and labels.shape[-1] < (
                gen_kwargs["max_new_tokens"] + 1
            ):
                labels = self._pad_tensors_to_max_len(labels, (gen_kwargs["max_new_tokens"] + 1))
        else:
            labels = None

        return (loss, generated_tokens, labels)


def main():
    texts, tags = load_conll('08272022.conll')
    train_texts, val_texts, train_tags, val_tags = train_test_split(
        texts, tags, test_size=.2, random_state=7)
    metric = evaluate.load("seqeval")

    """
    predictions = [['O','B-P','L-P'],['N','O','O','O']]
    references = [['O','B-P','L-P'],['M','O','O','O']]
    metric.compute(predictions=predictions,references=references)
    """

    tokenizer = T5TokenizerFast.from_pretrained("t5-base",add_prefix_space=True)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # In case the model returns more than the prediction logits
        if isinstance(preds, tuple):
            preds = preds[0]

        # somehow preds are also getting padded with -100s...
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100s in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds = [pred.strip().split() for pred in decoded_preds]
        decoded_labels = [label.strip().split() for label in decoded_labels]

        # More post-processing to make sure the predicted seq has the same len as labels
        for i in range(len(decoded_labels)):
            label_len = len(decoded_labels[i])
            if len(decoded_preds[i]) > label_len:
                decoded_preds[i] = decoded_preds[i][:label_len]
            elif len(decoded_preds[i]) < label_len:
                decoded_preds[i] += ['O']*(label_len-len(decoded_preds[i]))

        all_metrics = metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {
            "precision": all_metrics["overall_precision"],
            "recall": all_metrics["overall_recall"],
            "f1": all_metrics["overall_f1"],
            "accuracy": all_metrics["overall_accuracy"],
        }


    training_set = T5Dataset(train_texts, train_tags, tokenizer, max_len=200, task_prefix='')
    val_set = T5Dataset(val_texts, val_tags, tokenizer, max_len=200, task_prefix='')

    # get possible ids of label tokens
    labelids = set()
    for thing in training_set:
        labelids.update(thing['labels'].numpy())
    labelids.remove(tokenizer.eos_token_id)
    labelids.remove(-100)
    labelids = torch.LongTensor(list(labelids))

    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    args = Seq2SeqTrainingArguments(
        "T5-pretrained-labelseq-ner", evaluation_strategy="epoch", save_strategy="epoch",
        learning_rate=1e-4, num_train_epochs=10, weight_decay=0.01, predict_with_generate=True,
        per_device_train_batch_size=8, seed=7, #     no_cuda=True,
        )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = MyTrainer(
        model=model, args=args, train_dataset=training_set, eval_dataset=val_set,
        data_collator=data_collator, compute_metrics=compute_metrics, tokenizer=tokenizer,
    )
    beamsz = 4
    trainer.set_things(labelids, tokenizer, beamsz)
    trainer.train()

if __name__ == "__main__":
    main()
