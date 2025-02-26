import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast


class MyTokenizer:
    """
    Load tokenizer from bert config or defined BPE vocab dict
    """

    ################################################
    ### You can custome your own tokenizer here. ###
    ################################################
    def __init__(self, args, is_eval=False):
        if args.vocab == 'bert':
            tokenizer = AutoTokenizer.from_pretrained(args.config_name)
            self.tokenizer = tokenizer
            self.sep_token_id = tokenizer.sep_token_id
            self.pad_token_id = tokenizer.pad_token_id
            # save
            if not is_eval:
                tokenizer.save_pretrained(args.checkpoint_path)
        else:
            # load vocab from the path
            print('#' * 30, 'load vocab from', args.vocab)
            vocab_dict = {'[START]': 0, '[END]': 1, '[UNK]': 2, '[PAD]': 3}
            with open(args.vocab, 'r', encoding='utf-8') as f:
                for row in f:
                    vocab_dict[row.strip().split(' ')[0]] = len(vocab_dict)
            self.tokenizer = vocab_dict
            self.rev_tokenizer = {v: k for k, v in vocab_dict.items()}
            self.sep_token_id = vocab_dict['[END]']
            self.pad_token_id = vocab_dict['[PAD]']
            # save
            if int(os.environ['LOCAL_RANK']) == 0 and not is_eval:
                path_save_vocab = f'{args.checkpoint_path}/vocab.json'
                with open(path_save_vocab, 'w') as f:
                    json.dump(vocab_dict, f)

        self.vocab_size = len(self.tokenizer)
        args.vocab_size = self.vocab_size  # update vocab size in args

    def encode_token(self, sentences):
        if isinstance(self.tokenizer, dict):
            input_ids = [[0] + [self.tokenizer.get(x, self.tokenizer['[UNK]']) for x in seq.split()] + [1] for seq in
                         sentences]
        elif isinstance(self.tokenizer, PreTrainedTokenizerFast):
            input_ids = self.tokenizer(sentences, add_special_tokens=True)['input_ids']
        else:
            assert False, "invalid type of vocab_dict"
        return input_ids

    def decode_token(self, seq):
        if isinstance(self.tokenizer, dict):
            seq = seq.squeeze(-1).tolist()
            while len(seq) > 0 and seq[-1] == self.pad_token_id:
                seq.pop()
            tokens = " ".join([self.rev_tokenizer[x] for x in seq]).replace('__ ', '').replace('@@ ', '')
        elif isinstance(self.tokenizer, PreTrainedTokenizerFast):
            seq = seq.squeeze(-1).tolist()
            while len(seq) > 0 and seq[-1] == self.pad_token_id:
                seq.pop()
            tokens = self.tokenizer.decode(seq)
        else:
            assert False, "invalid type of vocab_dict"
        return tokens

    def decode_token_stop_at_sep(self, seq, return_length=None):
        length = 0
        if isinstance(self.tokenizer, dict):
            seq = seq.squeeze(-1).cpu().detach().numpy()
            idx = (seq == self.sep_token_id).argmax()
            length = idx + 1
            if seq[idx] == self.sep_token_id:
                seq = seq[:idx + 1]
            else:
                return self.decode_token(torch.tensor(seq).unsqueeze(-1))

            tokens = " ".join([self.rev_tokenizer[x] for x in seq]).replace('__ ', '').replace('@@ ', '')
        elif isinstance(self.tokenizer, PreTrainedTokenizerFast):
            seq = seq.squeeze(-1).cpu().detach().numpy()
            idx = (seq == self.sep_token_id).argmax()
            length = idx + 1
            if seq[idx] == self.sep_token_id:
                seq = seq[:idx + 1]
            else:
                return self.decode_token(torch.tensor(seq).unsqueeze(-1))

            tokens = self.tokenizer.decode(seq)
        else:
            assert False, "invalid type of vocab_dict"
        if return_length:
            return length, tokens
        else:
            return tokens
