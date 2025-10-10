# Since I plan to do this with a bulgarian dataset too, I need a custom tokenizer.
import itertools
import json
import os

class SimpleTokenizer:
    def __init__(self) -> None:
        self.id_to_token = self._create_vocab()
        self.token_to_id = {token: id for id, token in self.id_to_token.items()}
        self.vocab_size = len(list(self.id_to_token.keys()))

    def _create_vocab(self):
        """Encode all ascii 2 digit combinations"""
        chr_vocabulary = list(chr(c) for c in range(128))
        chr_combinations = itertools.product(chr_vocabulary, repeat=2)
        special_tokens = ['<BOS>', '<EOS>', '<PAD>', '<UNK>']
        id_to_token = {idx: special_token for idx, special_token in enumerate(special_tokens)}
        for idx, chr_combo in enumerate(chr_combinations):
            id_to_token[idx + len(special_tokens)] = ''.join(chr_combo)
        return id_to_token
    
    def encode(self, sequence):
        if sequence is None or len(sequence) == 0:
            raise Exception("Must provide a sequence.")
        if len(sequence) % 2 == 1:
            sequence += " "
        tokenized = []
        unk_id = self.token_to_id["<UNK>"]
        for idx in range(len(sequence) // 2):
            pair = sequence[2 * idx: 2*idx+2]
            id = self.token_to_id.get(''.join(pair), unk_id)
            tokenized.append(id)
        return tokenized
                
    def decode(self, ids):
        tokens = [self.id_to_token.get(i, "<UNK>") for i in ids]
        return ''.join(t for t in tokens if t not in ['<PAD>', '<BOS>', '<EOS>'])
    
    

class BulgarianEnglishSimpleTokenizer(SimpleTokenizer):
    def __init__(self):
        self.id_to_token = self._create_vocab()
        self.token_to_id = {token: id for id, token in self.id_to_token.items()}
        self.vocab_size = len(list(self.id_to_token.keys()))

    def _create_vocab(self):
        english_chars = [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]
        shared_symbols = [chr(i) for i in range(32, 127) if chr(i) not in english_chars]
        bulgarian_chars = [chr(i) for i in range(1040, 1104)]

        english_char_combinations = list(itertools.product(english_chars + shared_symbols, repeat=2))
        bulgarian_char_combinations = list(itertools.product(bulgarian_chars + shared_symbols, repeat=2))

        special_tokens = ['<BOS>', '<EOS>', '<PAD>', '<UNK>']
        id_to_token = {idx: special_token for idx, special_token in enumerate(special_tokens)}
        for idx, chr_combo in enumerate(english_char_combinations + bulgarian_char_combinations):
            id_to_token[idx + len(special_tokens)] = ''.join(chr_combo)
        return id_to_token

"""    
def test():
    tokenizer = SimpleTokenizer()
    print(tokenizer.vocab_size)
    tokenizer = BulgarianEnglishSimpleTokenizer()
    print(tokenizer.vocab_size)
    encoding = tokenizer.encode("This is a test sentence. It should work.")
    print(encoding)
    print(tokenizer.decode(encoding))

if __name__ == "__main__":
    test()
"""
        
        
