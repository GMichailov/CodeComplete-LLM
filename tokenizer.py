# Since I plan to do this with a bulgarian dataset too, I need a custom tokenizer.
import re

class BytePairEncoder:
    def __init__(self, corpus_sample) -> None:
        self.text = corpus_sample
        split_text = self._split_text()
        self.value_to_char_dict = self._bytes_to_unicode_ascii()
        self.char_to_value_dict = {v:k for k,v in self.value_to_char_dict.items()}

    def _split_text(self):
        pattern = r"'s|n't|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^ \p{L}\p{N}]+|\s+(?!\S)|\s+"
        return re.split(pattern, self.text)
    
    def _bytes_to_unicode_ascii(self):
        values = list(range(256))
        chars = [chr(v) for v in values]
        return dict(zip(values, chars))
    
    def train(self, iterations: int, max_vocab_size):
        pass

class BulgarianEnglishBytePairEncoder(BytePairEncoder):
    def __init__(self, corpus_sample) -> None:
        super().__init__(corpus_sample)
    

        
        
