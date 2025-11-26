class tokenizer():
    def __init__(self, max_relator_length: int):
        self._max_relator_length = max_relator_length
        self._pad_token = 0

        self._substitution_dict = self.substitution_dict()
        self._substitution_dict_inv = {v: k for k, v in self._substitution_dict.items()}

    def tokenize(self, s: str) -> list:
        pass

    def padding(self,s: tuple, max_relator_length: int) -> tuple:
        return s + (self._pad_token,)*(max_relator_length - len(s)) if len(s) < max_relator_length else s[:max_relator_length]
    
    def substitution_dict(self) -> dict:
        '''Create a substitution dictionary mapping action indices to action strings.'''
        substitution_dict = {}
        ind = 0
        for relator in ["r1", "r2"]:
            for i in range(self._max_relator_length):
                for j in range(self._max_relator_length):
                    for inverse in [True, False]:
                        substitution_dict[ind] = f"{relator},{i},{j},{inverse}"
                        ind += 1
        return substitution_dict

class FFN_tokenizer(tokenizer):
    def tokenize(self,s: str) -> list:
        '''
        Covert one relator by the rule:
            X to -1, Y to -2, x to 1, y to 2, AND pad with pad_tokens to max_relator_length

        Args:
            s: relator string
        '''
        res = []
        for c in s:
            if c == 'X':
                res.append(-1)
            elif c == 'Y':
                res.append(-2)
            elif c == 'x':
                res.append(1)
            elif c == 'y':
                res.append(2)
            else:
                raise ValueError(f"Unexpected character {c} in string {s}")
        while len(res) < self._max_relator_length:
            res.append(self._pad_token)
        return res
    
class DualRingActor_tokenizer(tokenizer):
    def tokenize(self,s: str) -> list:
        '''
        Covert one relator by the rule:
            X to 0, Y to 1, x to 2, y to 3, AND pad with pad_tokens to max_relator_length

        Args:
            s: relator string
        '''
        res = []
        for c in s:
            if c == 'X':
                res.append(0)
            elif c == 'Y':
                res.append(1)
            elif c == 'x':
                res.append(2)
            elif c == 'y':
                res.append(3)
            else:
                raise ValueError(f"Unexpected character {c} in string {s}")
        while len(res) < self._max_relator_length:
            res.append(self._pad_token)
        return res