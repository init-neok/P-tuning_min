import argparse
import torch
from torch.nn.utils.rnn import pad_sequence
from os.path import join
from transformers import GPT2LMHeadModel, AutoTokenizer, AutoModelForMaskedLM
import re

from transformers import AutoTokenizer

# from p_tuning.models import get_embedding_layer, create_model
# from data_utils.vocab import get_vocab_by_strategy, token_wrapper
# from data_utils.dataset import load_file
# from p_tuning.prompt_encoder import PromptEncoder

class PromptEncoder(torch.nn.Module):
    def __init__(self, template, hidden_size, tokenizer, device):
        super().__init__()
        self.device = device
        self.spell_length = sum(template)#9
        self.hidden_size = hidden_size#768
        self.tokenizer = tokenizer
        # self.args = args
        # ent embedding
        self.cloze_length = template###tuple() (3, 3, 3)
        self.cloze_mask = [
            [1] * self.cloze_length[0]  # first cloze
            + [1] * self.cloze_length[1]  # second cloze
            + [1] * self.cloze_length[2]  # third cloze
        ]#[[1, 1, 1, 1, 1, 1, 1, 1, 1]]
        self.cloze_mask = torch.LongTensor(self.cloze_mask).bool().to(self.device)

        self.seq_indices = torch.LongTensor(list(range(len(self.cloze_mask[0])))).to(self.device)#tensor=size 9   0-8
        
        # embedding
        self.embedding = torch.nn.Embedding(len(self.cloze_mask[0]), self.hidden_size).to(self.device)
        # LSTM
        self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size,
                                       hidden_size=self.hidden_size // 2,
                                       num_layers=2,
                                       dropout=0.0,
                                       bidirectional=True,
                                       batch_first=True)
        self.mlp_head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.hidden_size))
        print("init prompt encoder...")

    def forward(self):
        input_embeds = self.embedding(self.seq_indices).unsqueeze(0)
        output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0]).squeeze()
        return output_embeds



def get_embedding_layer(model):
    
    embeddings = model.bert.get_input_embeddings()
   
    return embeddings

class PTuneForLAMA(torch.nn.Module):

    def __init__(self, device, template):
        super().__init__()
        # self.args = args
        self.device = device

        # load relation templates
        self.relation_templates = dict(
            (d['relation'], d['template']) for d in load_file(join('/root/code/P_tuing_Liu/P-tuning/data/LAMA', 'relations.jsonl')))

        # load tokenizer
        tokenizer_src = 'bert-base-cased'
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, use_fast=False)

        # load pre-trained model
       
        self.model = AutoModelForMaskedLM.from_pretrained('bert-base-cased')
        self.model = self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False#把预训练模型全部冻结
        self.embeddings = get_embedding_layer(self.model)

        # set allowed vocab set
        self.vocab = self.tokenizer.get_vocab()#返回的是一个字典
        self.allowed_vocab_ids = set(self.vocab[k] for k in get_vocab_by_strategy( self.tokenizer))
        self.template = template

        # load prompt encoder
        self.hidden_size = self.embeddings.embedding_dim
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['[PROMPT]']})
        self.pseudo_token_id = self.tokenizer.get_vocab()['[PROMPT]']#字典的索引方式
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.unk_token_id

        self.spell_length = sum(self.template)
        self.prompt_encoder = PromptEncoder(self.template, self.hidden_size, self.tokenizer, self.device)#返回了一个trainable的 9*768的矩阵
        self.prompt_encoder = self.prompt_encoder.to(self.device)

    def embed_input(self, queries):
        '''整个函数就是在输入模型之前把那9个token插入到模型里'''
        bz = queries.shape[0]        
        # print('原生的query[0]的样子 ',self.tokenizer.convert_ids_to_tokens(list(queries[0])))
        queries_for_embedding = queries.clone()
        queries_for_embedding[(queries == self.pseudo_token_id)] = self.tokenizer.unk_token_id
        raw_embeds = self.embeddings(queries_for_embedding)
        # print(' raw_embeds  在这变成了embedding！：', raw_embeds.shape)
        # For using handcraft prompts

        print('queries.shape:     ',queries.shape)
        blocked_indices = (queries == self.pseudo_token_id).nonzero().reshape((bz, self.spell_length, 2))[:, :, 1]  # bz
        # print('blocked_indices:     ',blocked_indices)
        replace_embeds = self.prompt_encoder()####重点看这句！！！！！！！
        for bidx in range(bz):
            # print('self.prompt_encoder.spell_length: ',self.prompt_encoder.spell_length)
            for i in range(self.prompt_encoder.spell_length):
                raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]
        print('使用完Promptencoder后 的是 raw_embeds[0]的shape:',raw_embeds[0].shape)
        print('raw_embeds:',raw_embeds.shape)
        return raw_embeds#####返回的内容是每个batch的raw_embeds 目前不太理解，形状为【batchsize8   seq_len（不一定，20左右，长度都不一样）768】

    def get_query(self, x_h, prompt_tokens, x_t=None):
    #这一步是得到初始的嵌入，就只是一个模版的拼接 下一步要进入embed_input
        # For P-tuning
       
            # BERT-style model
        return [[self.tokenizer.cls_token_id]  # [CLS]
                + prompt_tokens * self.template[0]       # ['prompt']这个词在词表的索引乘3 (head entity)
                + [self.tokenizer.mask_token_id]  # head entity [mask]
                + prompt_tokens * self.template[1] ### ['prompt']这个词在词表的索引乘3 (tail entity)
                + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' ' + x_h))  # x_h是head entity
                + (prompt_tokens * self.template[2] if self.template[
                                                            2] > 0 else self.tokenizer.convert_tokens_to_ids(['.']))
                + [self.tokenizer.sep_token_id]
                ]
        
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
                        # print('我break了，因为pred是： ',pred)
                        break
                if pred == label_ids[i, 0]:
                    print('输出正确的自然语言是:', self.tokenizer.convert_ids_to_tokens(torch.argmax(logits[i,:,:],dim=1)))
                    # print()
                    hit1 += 1#如果第一个就是  就是猜对  

            if return_candidates:
                return loss, hit1, top10
            print('本batch猜对了{}个'.format(hit1))
            return loss, hit1

        

            
        
        return bert_out()
        




def get_vocab(model_name, strategy):
    if strategy == 'shared':
        if 'gpt' in model_name:
            return shared_vocab['gpt2-xl']
        elif 'roberta' in model_name or 'megatron' in model_name:
            return shared_vocab['roberta-large']
        else:
            assert model_name in shared_vocab
            return shared_vocab[model_name]
    elif strategy == 'lama':
        if 'gpt' in model_name:
            return lama_vocab['gpt2-xl']
        elif 'roberta' in model_name or 'megatron' in model_name:
            return lama_vocab['roberta-large']
        else:
            assert model_name in lama_vocab
            return lama_vocab[model_name]

def token_wrapper( token):
    return token






def get_vocab_by_strategy(tokenizer):
    
    return get_vocab('bert-base-cased', 'shared')

class LAMADataset(Dataset):
    def __init__(self, dataset_type, data, tokenizer):
        super().__init__()
        # self.args = args
        self.data = list()
        self.dataset_type = dataset_type
        self.x_hs, self.x_ts = [], []

        vocab = get_vocab_by_strategy( tokenizer)
        for d in data:
            if token_wrapper(d['obj_label']) not in vocab:
                continue
            self.x_ts.append(d['obj_label'])
            self.x_hs.append(d['sub_label'])
            self.data.append(d)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]['sub_label'], self.data[i]['obj_label']

def load_file(filename):
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data#一行是一个字典，list嵌套dict
def get_TREx_parameters():
    #下面就是P1001json的内容
    '''{"relation": "P1001", "template": "[X] is a legal term in [Y] .", "label": "applies to jurisdiction",
        "description": "the item (an institution, law, public office ...) or statement belongs to or has power over or applies 
        to the value (a territorial jurisdiction: a country, state, municipality, ...)", "type": "N-M"}'''
    relation = load_file(join('/root/code/P_tuing_Liu/P-tuning/LAMA', "single_relations/{}.jsonl".format("P1001"))[0]
    data_path_pre = "fact-retrieval/original/{}/".format("P1001")
    data_path_post = ".jsonl"
    return relation, data_path_pre, data_path_post
    #返回字典，包含relation，template，label，description，type
    #返回俩字符串

def init_vocab():
    global shared_vocab, lama_vocab
    shared_vocab = json.load(open(join('/root/code/P_tuing_Liu/P-tuning/data/LAMA/', '29k-vocab.json')))
    lama_vocab = json.load(open(join('/root/code/P_tuing_Liu/P-tuning/data/LAMA/', '34k-vocab.json')))

def set_seed():
    np.random.seed(34)
    torch.manual_seed(34)
    if torch.cuda.is_available :
        torch.cuda.manual_seed_all(34)
class Trainer(object):
    def __init__(self):
        # self.args = args
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


        # load tokenizer
        tokenizer_src = 'bert-base-uncased' 
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, use_fast=False)
        init_vocab()

        # load datasets and dataloaders
        self.relation, self.data_path_pre, self.data_path_post = self.get_TREx_parameters()

        self.train_data = load_file(join('/root/code/P_tuing_Liu/P-tuning/data/LAMA/fact-retrieval/original/P1001' + 'train' + self.data_path_post))
        self.dev_data = load_file(join('/root/code/P_tuing_Liu/P-tuning/data/LAMA/fact-retrieval/original/P1001'+ 'dev' + self.data_path_post))
        self.test_data = load_file(join('/root/code/P_tuing_Liu/P-tuning/data/LAMA/fact-retrieval/original/P1001' + 'test' + self.data_path_post))

        self.test_set = LAMADataset('test', self.test_data, self.tokenizer)
        self.train_set = LAMADataset('train', self.train_data, self.tokenizer)
        self.dev_set = LAMADataset('dev', self.dev_data, self.tokenizer)
        os.makedirs(self.get_save_path(), exist_ok=True)

        self.train_loader = DataLoader(self.train_set, batch_size=8, shuffle=True, drop_last=True)
        self.dev_loader = DataLoader(self.dev_set, batch_size=8)
        self.test_loader = DataLoader(self.test_set, batch_size=8)

        self.model = PTuneForLAMA( self.device,(3,3,3))

    def get_TREx_parameters(self):
        relation = load_file(join(self.args.data_dir, "single_relations/{}.jsonl".format("P1001")))[0]
        data_path_pre = "fact-retrieval/original/{}/".format("P1001")
        data_path_post = ".jsonl"
        return relation, data_path_pre, data_path_post

    def evaluate(self, epoch_idx, evaluate_type):
        self.model.eval()
        if evaluate_type == 'Test':
            loader = self.test_loader
            dataset = self.test_set
        else:
            loader = self.dev_loader
            dataset = self.dev_set
        with torch.no_grad():
            self.model.eval()
            hit1, loss = 0, 0
            for x_hs, x_ts in loader:
                if False and self.args.extend_data:
                    _loss, _hit1 = self.model.test_extend_data(x_hs, x_ts)
                elif evaluate_type == 'Test':
                    _loss, _hit1, top10 = self.model(x_hs, x_ts, return_candidates=True)
                else:
                    _loss, _hit1 = self.model(x_hs, x_ts)
                hit1 += _hit1
                loss += _loss.item()
            hit1 /= len(dataset)
            print("{} {} Epoch {} Loss: {} Hit@1:".format("P1001", evaluate_type, epoch_idx,
                                                          loss / len(dataset)), hit1)
        return loss, hit1

    def get_task_name(self):
        if self.args.only_evaluate:
            return "_".join([self.args.model_name + ('_' + self.args.vocab_strategy), 'only_evaluate'])
        names = [self.args.model_name + ('_' + self.args.vocab_strategy),
                 "template_{}".format(self.args.template if not self.args.use_original_template else 'original'),
                 "fixed" if not self.args.use_lm_finetune else "fine-tuned",
                 "seed_{}".format(self.args.seed)]
        return "_".join(names)

    def get_save_path(self):
        return join(self.args.out_dir, 'prompt_model', self.args.model_name, 'search', self.get_task_name(),
                    self.args.relation_id)

    def get_checkpoint(self, epoch_idx, dev_hit1, test_hit1):
        ckpt_name = "epoch_{}_dev_{}_test_{}.ckpt".format(epoch_idx, round(dev_hit1 * 100, 4),
                                                          round(test_hit1 * 100, 4))
        return {'embedding': self.model.prompt_encoder.state_dict(),
                'dev_hit@1': dev_hit1,
                'test_hit@1': test_hit1,
                'test_size': len(self.test_set),
                'ckpt_name': ckpt_name,
                'time': datetime.now(),
                'args': self.args}

    def save(self, best_ckpt):
        ckpt_name = best_ckpt['ckpt_name']
        path = self.get_save_path()
        os.makedirs(path, exist_ok=True)
        torch.save(best_ckpt, join(path, ckpt_name))
        # print("# Prompt:", self.model.prompt)
        print("# {} Checkpoint {} saved.".format(self.args.relation_id, ckpt_name))

    def train(self):
        best_dev, early_stop, has_adjusted = 0, 0, True
        best_ckpt = None
        params = [{'params': self.model.prompt_encoder.parameters()}]
        if self.args.use_lm_finetune:
            params.append({'params': self.model.model.parameters(), 'lr': 5e-6})
        optimizer = torch.optim.Adam(params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.args.decay_rate)

        for epoch_idx in range(5):
            # check early stopping
            if epoch_idx > -1:
                dev_loss, dev_hit1 = self.evaluate(epoch_idx, 'Dev')
                if epoch_idx == 0:
                    test_loss, test_hit1 = self.evaluate(epoch_idx, 'Test')
                if epoch_idx > 0 and (dev_hit1 >= best_dev) or self.args.only_evaluate:
                    test_loss, test_hit1 = self.evaluate(epoch_idx, 'Test')
                    best_ckpt = self.get_checkpoint(epoch_idx, dev_hit1, test_hit1)
                    early_stop = 0
                    best_dev = dev_hit1
                else:
                    early_stop += 1
                    if early_stop >= self.args.early_stop:
                        self.save(best_ckpt)
                        print("{} Early stopping at epoch {}.".format(self.args.relation_id, epoch_idx))
                        return best_ckpt
            if self.args.only_evaluate:
                break

            # run training
            hit1, num_of_samples = 0, 0
            tot_loss = 0
            for batch_idx, batch in tqdm(enumerate(self.train_loader)):
                self.model.train()
                loss, batch_hit1 = self.model(batch[0], batch[1])
                hit1 += batch_hit1
                tot_loss += loss.item()
                num_of_samples += len(batch[0])

                loss.backward()
                torch.cuda.empty_cache()
                optimizer.step()
                torch.cuda.empty_cache()
                optimizer.zero_grad()
            my_lr_scheduler.step()
        self.save(best_ckpt)

        return best_ckpt


def main(relation_id=None):
    args = construct_generation_args()
    if relation_id:
        args.relation_id = relation_id
    if type(args.template) is not tuple:
        args.template = eval(args.template)
    assert type(args.template) is tuple
    print(args.relation_id)
    print(args.model_name)
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # pre-parsing args
    parser.add_argument("--relation_id", type=str, default="P1001")
    parser.add_argument("--model_name", type=str, default='bert-base-cased', choices=SUPPORT_MODELS)
    parser.add_argument("--pseudo_token", type=str, default='[PROMPT]')

    parser.add_argument("--t5_shard", type=int, default=0)
    parser.add_argument("--mid", type=int, default=0)
    parser.add_argument("--template", type=str, default="(3, 3, 3)")
    parser.add_argument("--early_stop", type=int, default=20)

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=34, help="random seed for initialization")
    parser.add_argument("--decay_rate", type=float, default=0.98)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    # lama configuration
    parser.add_argument("--only_evaluate", type=bool, default=False)
    parser.add_argument("--use_original_template", type=bool, default=False)
    parser.add_argument("--use_lm_finetune", type=bool, default=False)

    parser.add_argument("--vocab_strategy", type=str, default="shared", choices=['original', 'shared', 'lama'])
    parser.add_argument("--lstm_dropout", type=float, default=0.0)

    # directories
    parser.add_argument("--data_dir", type=str, default=join(abspath(dirname(__file__)), '../data/LAMA'))
    parser.add_argument("--out_dir", type=str, default=join(abspath(dirname(__file__)), '../out/LAMA'))
    # MegatronLM 11B
    parser.add_argument("--checkpoint_dir", type=str, default=join(abspath(dirname(__file__)), '../checkpoints'))

    args = parser.parse_args()