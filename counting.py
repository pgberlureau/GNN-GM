import json
import chess
from torch import tensor, argmax
from torch.nn.functional import one_hot
from numpy.random import rand
from tqdm import tqdm

def load(line):
    kept = json.loads(line)
    to_save = {'fen':kept['fen'], 'evals':kept['evals'][0]['pvs'][0]}
    if not 'cp' in to_save['evals']: #means the position is evaluated as a mate
        if to_save['evals']['mate'] > 0:
            to_save['evals']['cp'] = 20_000
        else:
            to_save['evals']['cp'] = -20_000

    return to_save

def line_to_count(line):
    board = chess.Board(line["fen"])
    cp = int(line['evals']['cp'])

    res = 0
    for _,v in board.piece_map().items():
        if v.color:
            sign = 1
        else:
            sign = -1
        
        if v.piece_type == 1:
            res += 1*sign
        elif v.piece_type == 2:
            res += 3*sign
        elif v.piece_type == 3:
            res += 3*sign
        elif v.piece_type == 4:
            res += 5*sign
        elif v.piece_type == 5:
            res += 9*sign
    
    return (cp >0 and res > 0) or (cp>0 and res<0) or (rand() > 0.5), cp >=0

res = 0
ones_cp = 0
zeros_cp = 0

ones_y = 0
zeros_y = 0

data_size = 1e6

with open("lichess_db_eval.jsonl") as f:
    for i, line in tqdm(enumerate(f)):
    
        line = load(line)
        out, cp = line_to_count(line)
        res += out
        
        if cp:
            ones_cp += 1
        else:
            zeros_cp += 1

        if i >= data_size:
            break
    
print(res/data_size)
print(ones_cp/data_size)
print(zeros_cp/data_size)
#print(ones_y/data_size)
#print(zeros_y/data_size)
