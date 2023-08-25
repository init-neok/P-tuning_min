import torch
from torch.nn.utils.rnn import pad_sequence
from os.path import join

import re

from transformers import AutoTokenizer

from p_tuning.models import get_embedding_layer, create_model
from data_utils.vocab import get_vocab_by_strategy, token_wrapper
from data_utils.dataset import load_file
from p_tuning.prompt_encoder import PromptEncoder


class PTuneForLAMA(torch.nn.Module):

    def __init__(self, args, device, template):
        super().__init__()
        self.args = args
        self.device = device

        # load relation templates
        self.relation_templates = dict(
            (d['relation'], d['template']) for d in load_file(join(self.args.data_dir, 'relations.jsonl')))

        # load tokenizer
        tokenizer_src = 'roberta-large' if 'megatron' in self.args.model_name else self.args.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, use_fast=False)

        # load pre-trained model
        if 'megatron' in self.args.model_name and self.args.use_lm_finetune:
            raise RuntimeError("Can not apply args.use_lm_finetune=True on MegatronLM 11B.")
        self.model = create_model(self.args)
        self.model = self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = self.args.use_lm_finetune
        self.embeddings = get_embedding_layer(self.args, self.model)

        # set allowed vocab set
        self.vocab = self.tokenizer.get_vocab()
        self.allowed_vocab_ids = set(self.vocab[k] for k in get_vocab_by_strategy(self.args, self.tokenizer))

        if 'gpt' in self.args.model_name or 'megatron' in self.args.model_name:
            template = (template[0], template[1], 0)
        self.template = template

        # load prompt encoder
        self.hidden_size = self.embeddings.embedding_dim
        self.tokenizer.add_special_tokens({'additional_special_tokens': [self.args.pseudo_token]})
        self.pseudo_token_id = self.tokenizer.get_vocab()[self.args.pseudo_token]
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.unk_token_id

        self.spell_length = sum(self.template)
        self.prompt_encoder = PromptEncoder(self.template, self.hidden_size, self.tokenizer, self.device, args)#返回了一个trainable的 9*768的矩阵
        self.prompt_encoder = self.prompt_encoder.to(self.device)

    def embed_input(self, queries):
        bz = queries.shape[0]
        tokenizer1=AutoTokenizer.from_pretrained('bert-base-cased')
        tokenizer1.add_special_tokens({'additional_special_tokens':['[PROMPT]']})
        
        print('原生的query[0]的样子 ',tokenizer1.convert_ids_to_tokens(list(queries[0])))
        queries_for_embedding = queries.clone()
        queries_for_embedding[(queries == self.pseudo_token_id)] = self.tokenizer.unk_token_id
        raw_embeds = self.embeddings(queries_for_embedding)
        print(' raw_embeds  在这变成了embedding！：', raw_embeds.shape)
        # For using handcraft prompts
        if self.args.use_original_template: 
            return raw_embeds
        print('queries.shape:     ',queries.shape)
        blocked_indices = (queries == self.pseudo_token_id).nonzero().reshape((bz, self.spell_length, 2))[:, :, 1]  # bz
        # print('blocked_indices:     ',blocked_indices)
        replace_embeds = self.prompt_encoder()
        for bidx in range(bz):
            # print('self.prompt_encoder.spell_length: ',self.prompt_encoder.spell_length)
            for i in range(self.prompt_encoder.spell_length):
                raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]
        print('使用完Promptencoder后 的是 raw_embeds[0]的shape:',raw_embeds[0].shape)
        print('raw_embeds:',raw_embeds.shape)
        return raw_embeds#####返回的内容是每个batch的raw_embeds 目前不太理解，形状为【batchsize8   seq_len（不一定，20左右，长度都不一样）768】

    def get_query(self, x_h, prompt_tokens, x_t=None):
        # For using handcraft prompts
        if self.args.use_original_template:
            if 'gpt' in self.args.model_name or 'megatron' in self.args.model_name:
                query = re.sub(r'\[Y\].*', '', self.relation_templates[self.args.relation_id].replace('[X]', x_h))
                return self.tokenizer(' ' + query)['input_ids']
            else:
                query = self.relation_templates[self.args.relation_id].replace('[X]', x_h).replace('[Y]',
                                                                                                   self.tokenizer.mask_token)####qury是一个字符串
                return self.tokenizer(' ' + query)['input_ids']
        # For P-tuning
        if 'gpt' not in self.args.model_name and 'megatron' not in self.args.model_name:
            # BERT-style model
            return [[self.tokenizer.cls_token_id]  # [CLS]
                    + prompt_tokens * self.template[0]       # ['prompt']这个词在词表的索引乘3 (head entity)
                    + [self.tokenizer.mask_token_id]  # head entity [mask]
                    + prompt_tokens * self.template[1] ### ['prompt']这个词在词表的索引乘3 (tail entity)
                    + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' ' + x_h))  # [MASK] (tail entity)
                    + (prompt_tokens * self.template[2] if self.template[
                                                               2] > 0 else self.tokenizer.convert_tokens_to_ids(['.']))
                    + [self.tokenizer.sep_token_id]
                    ]
        elif 'gpt' in self.args.model_name or 'megatron' in self.args.model_name:
            # GPT-style models
            return [prompt_tokens * self.template[0]
                    + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' ' + x_h))  # head entity
                    + prompt_tokens * self.template[1]
                    + (self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(' ' + x_t)) if x_t is not None else [])
                    ]
        else:
            raise NotImplementedError("The query template for {} has not been defined.".format(self.args.model_name))

    def forward(self, x_hs, x_ts, return_candidates=False):
        bz = len(x_hs)#batchsize
        print('x_hs 是',x_hs)
        # construct query ids
        prompt_tokens = [self.pseudo_token_id]#新加在词表里的假的tokenid
        x_ts = [token_wrapper(self.args, x_t) for x_t in x_ts]
        queries = [torch.LongTensor(self.get_query(x_hs[i], prompt_tokens)).squeeze(0) for i in range(bz)]
        # print('queries[0]是：',queries[0].shape)
        # print('queries是[7]：',queries[7].shape)
        queries = pad_sequence(queries, True, padding_value=self.pad_token_id).long().to(self.device)
        print('经过pad_sequence queries是：',queries.shape)
        print('输入embedding之前的queries[0]是：',queries[0])
        # construct label ids
        print('x_ts是 就是label',x_ts)
        label_ids = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(x_ts)).reshape(
            (bz, -1)).to(self.device)
        print('label_ids.shape是     ',label_ids.shape)
        attention_mask = queries != self.pad_token_id

        # get embedded input
        inputs_embeds = self.embed_input(queries)

        def bert_out():
            label_mask = (queries == self.tokenizer.mask_token_id).nonzero().reshape(bz, -1)[:, 1].unsqueeze(
                1).to(self.device)  # bz * 1    这个地方nonzero后都取的4 再reshape时候，因为是
            #label mask=[[4],...[4]] 8*1 tensor
            # print('一个中间过程的label_mask tensor  ',(queries == self.tokenizer.mask_token_id).nonzero())
            # print('一个中间过程的label_mask tensor  ',(queries == self.tokenizer.mask_token_id).nonzero().reshape(bz, -1))


            print('label_mask是：',label_mask)
            labels = torch.empty_like(queries).fill_(-100).long().to(self.device)  # bz * seq_len
            labels = labels.scatter_(1, label_mask, label_ids)   #label_ids 是8*1的tensor 是每个答案词 
            #label_mask是每个mask token 在输入的index
           
            output = self.model(inputs_embeds=inputs_embeds.to(self.device),
                                attention_mask=attention_mask.to(self.device).bool(),
                                labels=labels.to(self.device))
            loss, logits = output.loss, output.logits
            print('logits.shape模型算出来的东西的shape  ',logits.shape)
            pred_ids = torch.argsort(logits, dim=2, descending=True)
            hit1 = 0
            top10 = []
            for i in range(bz):
                pred_seq = pred_ids[i, label_mask[i, 0]].tolist()#取到mask位置的预测
                for pred in pred_seq:
                    if pred in self.allowed_vocab_ids:
                        print('我break了，因为pred是： ',pred)
                        break
                if pred == label_ids[i, 0]:
                    print('输出正确的自然语言是:', self.tokenizer.convert_ids_to_tokens(torch.argmax(logits[i,:,:],dim=1)))
                    # print()
                    hit1 += 1#如果第一个就是  就是猜对  

            if return_candidates:
                return loss, hit1, top10
            print('本batch猜对了{}个'.format(hit1))
            return loss, hit1

        def gpt_out():
            labels = torch.empty_like(queries).fill_(-100).long().to(self.device)  # bz * seq_len
            label_mask = (attention_mask.long().sum(dim=1) - 1).unsqueeze(1).to(self.device)
            labels = labels.scatter_(1, label_mask, label_ids)

            output = self.model(inputs_embeds=inputs_embeds.to(self.device).half(),
                                attention_mask=attention_mask.to(self.device).half(),
                                labels=labels.to(self.device))
            loss, logits = output.loss, output.logits

            pred_ids = torch.argsort(logits, dim=2, descending=True)
            hit1 = 0
            top10 = []
            for i in range(bz):
                top10.append([])
                pred_seq = pred_ids[i, label_mask[i, 0]].tolist()
                for pred in pred_seq:
                    if pred in self.allowed_vocab_ids:
                        top10[-1].append(pred)
                        if len(top10[-1]) >= 10:
                            break
                pred = top10[-1][0]
                if pred == label_ids[i, 0]:
                    hit1 += 1

            if return_candidates:
                return loss, hit1, top10
            return loss, hit1

        def megatron_out():
            labels = torch.empty_like(queries).fill_(-100).long().to(self.device)  # bz * seq_len
            label_mask = (attention_mask.long().sum(dim=1) - 1).unsqueeze(1).to(self.device)
            labels = labels.scatter_(1, label_mask, label_ids)
            if not self.args.use_lm_finetune:
                _attention_mask = attention_mask.float().half()
                _input_embeds = inputs_embeds.float().half()
            else:
                _attention_mask = attention_mask.float()
                _input_embeds = inputs_embeds.float()
            output = self.model.decoder.predict(prev_output_tokens=queries,
                                                inputs_embeds=_input_embeds.to(self.device),
                                                attention_mask=_attention_mask.to(self.device).bool(),
                                                labels=labels.to(self.device))
            logits, loss = output

            pred_ids = torch.argsort(logits, dim=2, descending=True)
            hit1 = 0
            top10 = []
            for i in range(bz):
                top10.append([])
                pred_seq = pred_ids[i, label_mask[i, 0]].tolist()
                for pred in pred_seq:
                    if pred in self.allowed_vocab_ids:
                        top10[-1].append(pred)
                        if len(top10[-1]) >= 10:
                            break
                pred = top10[-1][0]
                if pred == label_ids[i, 0]:
                    hit1 += 1
            if return_candidates:
                return loss, hit1, top10
            return loss, hit1

        if 'bert' in self.args.model_name:
            return bert_out()
        elif 'gpt' in self.args.model_name:
            return gpt_out()
        elif 'megatron' in self.args.model_name:
            return megatron_out()
        else:
            raise NotImplementedError()
