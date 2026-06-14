import regex as re
import os
from typing import BinaryIO
import io
from multiprocessing import Pool
from collections import Counter, defaultdict
import json
import cProfile, pstats
from collections.abc import Iterable, Iterator
from pathlib import Path
import numpy as np
import time

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
    mini_chunk_size: int
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"
    
    #for testing
    #data = file.read(mini_chunk_size*20)
    #file = io.BytesIO(data)

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size


    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def chunk_pretokenization(args):
    start, end, filepath, special_tokens = args
    length_eot = len(b'<|endoftext|>')
    with open(filepath, "rb") as f:
        counter = Counter()
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        if start == 0:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
        else:
            f.seek(start+length_eot)
            chunk = f.read(end - start - length_eot).decode("utf-8", errors="ignore")
                
        chunks = re.split(special_tokens, chunk)

        for chunk in chunks:
            matches = re.finditer(PAT, chunk)
            for match in matches:
                token = match.group()
                token_bytes = token.encode('utf-8')
                token_tuple = tuple(token_bytes[i:i+1] for i in range(len(token_bytes)))
                counter[token_tuple] += 1
    return counter

def apply_pretokenization(filepath: str, special_tokens: list[str]) -> dict:
    t0 = time.time()
    done = 0
    special_tokens = "|".join([re.escape(tok) for tok in special_tokens])
    with open(filepath, "rb") as f:
        num_processes = os.cpu_count()
        boundaries = find_chunk_boundaries(f, num_processes*20, b"<|endoftext|>", mini_chunk_size=4096)

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        arg_list = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            arg_list.append((start, end, filepath, special_tokens))
        final_counter = Counter()
        with Pool(num_processes) as p:
            for counter in p.imap_unordered(chunk_pretokenization, arg_list):
                final_counter += counter
                done += 1
                elapsed = time.time() - t0
                print(f"[pretok] {done}/{len(arg_list)} chunks | "
                  f"{elapsed:.0f}s elapsed", flush=True)

    return final_counter

def apply_bpe(pretokenization_dict: dict, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    n = len(special_tokens)
    t0 = time.time()
    vocabulary = {i+n: bytes([i]) for i in range(256)}
    for i  in range(n):
        vocabulary[i] = special_tokens[i].encode('utf-8')
    merges = []

    #initialisation de pairs_counter et inverse_index
    pairs_counter = Counter()
    inverse_index = defaultdict(set) # pairs -> pretokens that have this pair
    for pretoken, pretoken_value in pretokenization_dict.items():
            for token1, token2 in zip(pretoken[:-1], pretoken[1:]):
                pairs_counter[(token1, token2)] += pretoken_value
                inverse_index[(token1, token2)].add(pretoken) 
    total_merges = vocab_size - 256 - n
    for itera in range(total_merges):
        if itera % 100 == 0:
            elapsed = time.time() - t0
            rate = (itera + 1) / elapsed if elapsed > 0 else 0
            print(f"[bpe] merge {itera}/{total_merges} | "
                  f"vocab size {len(vocabulary)} | "
                  f"pairs in counter {len(pairs_counter):,} | "
                  f"pretokens {len(pretokenization_dict):,} | "
                  f"{elapsed:.0f}s", flush=True)
            
         # now we create the new token
        if not pairs_counter:
            print("bpe stopped at", itera)
            break

        token_with_max, value_max = max(pairs_counter.items(), key=lambda x: (x[1], x[0]))
        new_token1, new_token2 = token_with_max
        new_token = new_token1 + new_token2

        merges.append((new_token1, new_token2))
        vocabulary[256 + n + itera] = new_token 

        #mise a jour de pairs_counter
        for pretoken in inverse_index[(new_token1, new_token2)]:
            index_to_merge = []
            pretoken_value = pretokenization_dict[pretoken]
            new_pretoken = []
            i = 0
            length = len(pretoken)
            while i < length:
                if i < length - 1 and pretoken[i] == new_token1 and pretoken[i+1] == new_token2:
                    new_pretoken.append(new_token)
                    index_to_merge.append(i)
                    i += 2
                else:
                    new_pretoken.append(pretoken[i])
                    i += 1
                    
            pretokenization_dict[tuple(new_pretoken)] = pretoken_value
            del pretokenization_dict[pretoken]
            new_pretoken = tuple(new_pretoken)

            last = -3
            for index in index_to_merge:
                if index-last > 2 and index > 0:
                    pairs_counter[(pretoken[index-1], pretoken[index])] -= pretoken_value
                if index < length - 2:
                    pairs_counter[(pretoken[index+1], pretoken[index+2])] -= pretoken_value
                last = index

            for token1, token2 in zip(pretoken[:-1], pretoken[1:]):
                if not (token1, token2) == (new_token1, new_token2):
                    inverse_index[(token1, token2)].discard(pretoken)

            for token1, token2 in zip(new_pretoken[:-1], new_pretoken[1:]):
                if token1 == new_token:
                    pairs_counter[(new_token, token2)] += pretoken_value

                elif token2 == new_token:
                    pairs_counter[(token1, new_token)] += pretoken_value
                inverse_index[(token1, token2)].add(new_pretoken)
            
        del inverse_index[(new_token1, new_token2)]
        del pairs_counter[(new_token1,new_token2)]

    return vocabulary, merges

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    pretokenization_dict = apply_pretokenization(input_path, special_tokens)
    vocabulary, merges = apply_bpe(pretokenization_dict, vocab_size, special_tokens)
    return vocabulary, merges

class Tokenizer():
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None =None):
        self.vocab = vocab
        self.merges = merges
        if special_tokens:
            special_tokens = sorted(special_tokens, key= lambda x: -len(x))
        self.special_tokens = special_tokens
        self.vocab_inverse = {v: k for k, v in self.vocab.items()}
        self.merges_rank = {v:i for i, v in enumerate(self.merges)}
    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)
        vocab = {int(k): bytes([BYTE_DECODER[x] for x in v]) for k, v in raw.items()}

        merges = []
        with open(merges_filepath, "r") as f:
            for ligne in f:
                ligne = ligne.rstrip("\n")
                if not ligne or ligne.startswith('#'):
                    continue
                bit1, bit2 = ligne.split(" ",1)
                merges.append((bytes([BYTE_DECODER[x] for x in bit1]), bytes([BYTE_DECODER[x] for x in bit2])))
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        pretokenized_text =  self._pretokenization(text)
        encoded_text = []

        for pretoken in pretokenized_text:
            if self.special_tokens and pretoken in self.special_tokens:
                encoded_text.append(self.vocab_inverse[pretoken.encode('utf-8')])
                continue

            pretoken_bytes = [bytes([b]) for b in pretoken.encode('utf-8')]
            merge_candidates = {}

            for byte1, byte2 in zip(pretoken_bytes[:-1], pretoken_bytes[1:]):
                if (byte1, byte2) in self.merges_rank:
                    merge_candidates[(byte1, byte2)] = self.merges_rank[(byte1, byte2)]

            while merge_candidates:
                new_pretoken_bytes = []
                new_merge_candidates = {}
                min_merge = min(merge_candidates, key= merge_candidates.get)
                merge1, merge2 = min_merge
                new_merge = merge1 + merge2
                i = 0
                n = len(pretoken_bytes)
                while i < n:
                    if i < n - 1 and pretoken_bytes[i] == merge1 and pretoken_bytes[i+1] == merge2:
                        new_pretoken_bytes.append(new_merge)
                        i += 2
                    else:
                        new_pretoken_bytes.append(pretoken_bytes[i])
                        i += 1
                
                for byte1, byte2 in zip(new_pretoken_bytes[:-1], new_pretoken_bytes[1:]):
                    if (byte1, byte2) in self.merges_rank:
                        new_merge_candidates[(byte1, byte2)] = self.merges_rank[(byte1, byte2)]
                
                pretoken_bytes = new_pretoken_bytes
                merge_candidates = new_merge_candidates
            encoded_pretoken = [self.vocab_inverse[byte] for byte in pretoken_bytes]
            encoded_text += encoded_pretoken
            
        return encoded_text


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for string in iterable:
            for id in self.encode(string):
                yield id

    def decode(self, ids: list[int]) -> str:
        byte_text = b''.join(self.vocab[id] for id in ids)
        decoded_text = byte_text.decode('utf-8', errors='replace')
        return decoded_text

    def _pretokenization(self, text: str) -> list[str]:
        pretokenized_text = []
        special_pat = "|".join([re.escape(tok) for tok in self.special_tokens]) if self.special_tokens else ""
        chunks = re.split(f"({special_pat})", text) if special_pat else [text]

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for chunk in chunks:
            if self.special_tokens and chunk in self.special_tokens:
                pretokenized_text.append(chunk)
            else:
                matches = re.finditer(PAT, chunk)
                for match in matches:
                    token = match.group()
                    pretokenized_text.append(token)
                
        return pretokenized_text

def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))

BYTE_ENCODER = bytes_to_unicode()
BYTE_DECODER = {v: k for k, v in BYTE_ENCODER.items()}

def save(vocab, merges, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_filepath = output_dir / "vocab.json"
    merges_filepath = output_dir / "merges.txt"

    with open(vocab_filepath, "w", encoding="utf-8") as f:
        serialized = {id_: ''.join([BYTE_ENCODER[x] for x in b]) for id_, b in vocab.items()}
        json.dump(serialized, f, ensure_ascii=False)
    
    with open(merges_filepath, "w", encoding="utf-8") as f:
        for b1, b2 in merges:
            f.write(f"{''.join([BYTE_ENCODER[x] for x in b1])} {''.join([BYTE_ENCODER[x] for x in b2])}\n")

def tokenize_text(input_path, output_path, tokenizer, max_chunk_size, is_train):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    if is_train:
        output_path = output_path / "train.bin"
    else:
        output_path = output_path / "valid.bin"
    buffer = []
    total_tokens = 0
    with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "wb") as f_out:
        for token in tokenizer.encode_iterable(f_in):
            buffer.append(token)
            if len(buffer) >= max_chunk_size:
                np.array(buffer, dtype=np.uint16).tofile(f_out)
                total_tokens += len(buffer)
                buffer = []
        if buffer:
            total_tokens += len(buffer)
            np.array(buffer, dtype=np.uint16).tofile(f_out)

    print(f"Saved {total_tokens:,} tokens to {output_path}")

def worker_tokenize_text(args):
    input_path, tokenizer, start, end = args
    with open(input_path, "rb") as f_in:
        f_in.seek(start)
        chunk = f_in.read(end - start)
        chunk_str = chunk.decode("utf-8")
        tokens = tokenizer.encode(chunk_str)
    return np.array(tokens, dtype=np.uint16)

def tokenize_text_with_pool(input_path, output_path, tokenizer, max_chunk_size, is_train):
    t0 = time.time()
    done = 0
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    if is_train:
        output_path = output_path / "train.bin"
    else:
        output_path = output_path / "valid.bin"
    total_tokens = 0

    with open(input_path, "rb") as f_in:
        num_processes = os.cpu_count()
        boundaries = find_chunk_boundaries(f_in, num_processes*20, b"<|endoftext|>", mini_chunk_size=max_chunk_size)
           
    arg_list = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        arg_list.append((input_path, tokenizer, start, end))

    with open(output_path, "wb") as f_out:
        with Pool(num_processes) as p:
            for arr in p.imap(worker_tokenize_text, arg_list):
                arr.tofile(f_out)
                total_tokens += len(arr)
                done += 1
                elapsed = time.time() - t0
                rate = total_tokens / elapsed if elapsed else 0
                print(f"[tokenize] {done}/{len(arg_list)} chunks | "
                  f"{total_tokens:,} tokens | "
                  f"{rate/1e6:.1f}M tok/s | "
                  f"{elapsed:.0f}s", flush=True)
    
    print(f"Saved {total_tokens:,} tokens to {output_path}")


if __name__ == "__main__":
    t0 = time.time()
    special_tokens = ["<|endoftext|>"]
    input_path_train = "/Users/axel/Documents/stanford-cs336/assignment-1/data/owt_train.txt"
    input_path_valid = "/Users/axel/Documents/stanford-cs336/assignment-1/data/owt_valid.txt"
    vocab_size = 32000
    output_dir = "artifacts/owt"
    print("[main] starting BPE training...", flush=True)
    vocabulary, merges = train_bpe(input_path=input_path_train, vocab_size=vocab_size, special_tokens=["<|endoftext|>"])
    print(f"[main] BPE done in {time.time() - t0:.0f}s, vocab={len(vocabulary)}", flush=True)

    save(vocab=vocabulary, merges=merges, output_dir=output_dir)
    tokenizer = Tokenizer(vocab=vocabulary, merges=merges, special_tokens=special_tokens)
    print("ok pour le tokenizer")

    max_chunk_size = 1_000_000
    output_path= "artifacts/owt"
    print("[main] tokenizing valid...", flush=True)
    tokenize_text_with_pool(input_path_valid, output_path, tokenizer, max_chunk_size, is_train = False)
    print("[main] tokenizing train...", flush=True)
    tokenize_text_with_pool(input_path_train, output_path, tokenizer, max_chunk_size, is_train=True)
    print(f"[main] all done in {time.time() - t0:.0f}s", flush=True)
# For optimization of tokenization of text:
"""
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        2  283.613  141.806 3818.675 1909.338 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/cs336_basics/tokenization.py:296(tokenize_text)
1216060860   84.252    0.000 3361.205    0.000 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/cs336_basics/tokenization.py:255(encode_iterable)
 15757889 2039.176    0.000 3276.472    0.000 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/cs336_basics/tokenization.py:211(encode)
 15757889  264.855    0.000  697.264    0.000 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/cs336_basics/tokenization.py:265(_pretokenization)
5584095261  412.427    0.000  412.427    0.000 {method 'append' of 'list' objects}
 34261107   45.594    0.000  212.057    0.000 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/regex/_main.py:459(_compile)
1000339867  160.089    0.000  160.089    0.000 {built-in method builtins.min}
 68522376   45.658    0.000  138.176    0.000 /opt/homebrew/Caskroom/miniconda/base/lib/python3.13/enum.py:1608(__and__)
 18503218    8.512    0.000  123.384    0.000 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/regex/_main.py:350(finditer)
2216402059  122.565    0.000  122.565    0.000 {built-in method builtins.len}
 15757889    7.717    0.000  118.928    0.000 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/regex/_main.py:324(split)
 15757889   55.979    0.000   93.521    0.000 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/regex/_main.py:387(escape)
205567154   44.816    0.000   69.666    0.000 /opt/homebrew/Caskroom/miniconda/base/lib/python3.13/enum.py:1590(_get_value)
544318113   58.373    0.000   58.373    0.000 {method 'encode' of 'str' objects}
541572784   50.451    0.000   50.451    0.000 {method 'group' of '_regex.Match' objects}
408388921   30.249    0.000   30.249    0.000 {built-in method builtins.isinstance}
     1217   23.117    0.019   23.117    0.019 {built-in method numpy.array}
 68522386   14.237    0.000   22.853    0.000 /opt/homebrew/Caskroom/miniconda/base/lib/python3.13/enum.py:696(__call__)
 34261107   10.320    0.000   15.805    0.000 <frozen importlib._bootstrap>:1390(_handle_fromlist)
173336779   13.192    0.000   13.192    0.000 {method 'isspace' of 'str' objects}
 68522386    8.616    0.000    8.616    0.000 /opt/homebrew/Caskroom/miniconda/base/lib/python3.13/enum.py:1157(__new__)
 15757889    8.543    0.000    8.543    0.000 {method 'split' of '_regex.Pattern' objects}
 34261107    8.027    0.000    8.027    0.000 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/regex/_main.py:481(complain_unused_args)
 18503218    5.483    0.000    5.483    0.000 {method 'finditer' of '_regex.Pattern' objects}
 34261161    4.453    0.000    4.453    0.000 {method 'get' of 'dict' objects}
 31515792    4.095    0.000    4.095    0.000 {method 'join' of 'str' objects}
 34261109    3.314    0.000    3.314    0.000 {built-in method builtins.hasattr}
   274694    0.136    0.000    0.482    0.000 <frozen codecs>:322(decode)
     1217    0.433    0.000    0.436    0.000 {method 'tofile' of 'numpy.ndarray' objects}
   274694    0.345    0.000    0.345    0.000 {built-in method _codecs.utf_8_decode}
        4    0.033    0.008    0.033    0.008 {built-in method _io.open}
     1217    0.001    0.000    0.003    0.000 <frozen abc>:117(__instancecheck__)
     1217    0.002    0.000    0.002    0.000 {built-in method _abc._abc_instancecheck}
      5/2    0.000    0.000    0.001    0.000 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/regex/_regex_core.py:452(_parse_pattern)
     13/7    0.000    0.000    0.001    0.000 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/regex/_regex_core.py:462(parse_sequence)
        4    0.000    0.000    0.000    0.000 {method '__exit__' of '_io._IOBase' objects}
        3    0.000    0.000    0.000    0.000 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/regex/_regex_core.py:850(parse_paren)
        1    0.000    0.000    0.000    0.000 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/regex/_regex_core.py:1185(parse_flags_subpattern)
        2    0.000    0.000    0.000    0.000 /opt/homebrew/Caskroom/miniconda/base/lib/python3.13/pathlib/_local.py:717(mkdir)
       10    0.000    0.000    0.000    0.000 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/regex/_regex_core.py:1256(parse_escape)
        2    0.000    0.000    0.000    0.000 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/regex/_regex_core.py:1511(parse_set)
        6    0.000    0.000    0.000    0.000 /opt/homebrew/Caskroom/miniconda/base/lib/python3.13/pathlib/_local.py:166(__fspath__)
        1    0.000    0.000    0.000    0.000 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/regex/_regex_core.py:1166(parse_subpattern)
        8    0.000    0.000    0.000    0.000 /opt/homebrew/Caskroom/miniconda/base/lib/python3.13/pathlib/_local.py:227(__str__)
      2/1    0.000    0.000    0.000    0.000 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/regex/_regex_core.py:2143(optimise)
        2    0.000    0.000    0.000    0.000 {built-in method posix.mkdir}
     13/7    0.000    0.000    0.000    0.000 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/regex/_regex_core.py:3514(optimise)
        1    0.000    0.000    0.000    0.000 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/regex/_regex_core.py:370(_compile_firstset)
        2    0.000    0.000    0.000    0.000 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/regex/_regex_core.py:1577(parse_set_imp_union)
      2/1    0.000    0.000    0.000    0.000 /Users/axel/Documents/stanford-cs336/assignment-1/assignment1-basics/.venv/lib/python3.13/site-packages/regex/_regex_core.py:2237(_flatten_branches)"""