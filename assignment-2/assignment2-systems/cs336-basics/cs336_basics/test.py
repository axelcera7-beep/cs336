import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
text = "ceci est un exqmple: 'ràçu'rtàçr / / /erfer"

matches = re.finditer(PAT, text)
for match in matches:
    print(match.group())
