import argparse
import json
from pathlib import Path

import chess
from tqdm import tqdm


def load_board(line) -> (chess.Board, int):
    kept = json.loads(line)
    to_save = {'fen': kept['fen'], 'evals': kept['evals'][0]['pvs'][0]}
    if not 'cp' in to_save['evals']:  # means the position is evaluated as a mate
        if to_save['evals']['mate'] > 0:
            to_save['evals']['cp'] = 20_000
        else:
            to_save['evals']['cp'] = -20_000

    board = chess.Board(to_save["fen"])
    cp = int(to_save['evals']['cp'])

    return board, cp


def count_qualities(board):
    res = 0
    for _, v in board.piece_map().items():
        if v.color:
            sign = 1
        else:
            sign = -1

        if v.piece_type == 1:
            res += 1 * sign
        elif v.piece_type == 2:
            res += 3 * sign
        elif v.piece_type == 3:
            res += 3 * sign
        elif v.piece_type == 4:
            res += 5 * sign
        elif v.piece_type == 5:
            res += 9 * sign

    return res


def update_vars(cp, out, data):
    if (cp >= 0 and out >= 0) or (cp < 0 and out < 0):
        data['qualities_accuracy'] += 1

    if cp >= 0:
        data['nb_win'] += 1
    else:
        data['nb_loose'] += 1

    if out >= 0:
        data['nb_pred_win'] += 1
    else:
        data['nb_pred_loose'] += 1


def main():
    parser = argparse.ArgumentParser(description="Process lichess JSONL file.")

    parser.add_argument(
        "lichess_file",
        type=Path,
        help="Path to the lichess JSONL file.",
    )
    parser.add_argument(
        "--size",
        type=int,
        required=True,
        help="Number of entries to process.",
    )

    args = parser.parse_args()

    lichess_file = args.lichess_file
    size = args.size

    data = {'qualities_accuracy': 0, 'nb_win': 0, 'nb_pred_win': 0, 'nb_loose': 0, 'nb_pred_loose': 0, }

    with tqdm(total=size) as pbar:
        with lichess_file.open('r') as f:
            for i, line in enumerate(f):
                board, cp = load_board(line)
                out = count_qualities(board)
                update_vars(cp, out, data)

                board = board.mirror().transform(chess.flip_vertical)
                out = count_qualities(board)
                update_vars(cp, out, data)

                if 2 * i >= size:
                    break

                pbar.update(2)

    qa = data['qualities_accuracy'] / size
    wr = data['nb_win'] / size
    lr = data['nb_loose'] / size
    pwr = data['nb_pred_win'] / size
    plr = data['nb_pred_loose'] / size

    print(f"{'Qualities Accuracy':<20}: {qa:.4f}")
    print(f"{'Win Rate':<20}: {wr:.4f}")
    print(f"{'Loss Rate':<20}: {lr:.4f}")
    print(f"{'Predicted Win Rate':<20}: {pwr:.4f}")
    print(f"{'Predicted Loss Rate':<20}: {plr:.4f}")


if __name__ == "__main__":
    main()

# python counting.py  --size 1000000
#
# Qualities Accuracy  : 0.5970
# Win Rate            : 0.7182
# Loss Rate           : 0.2818
# Predicted Win Rate  : 0.7203
# Predicted Loss Rate : 0.2797
