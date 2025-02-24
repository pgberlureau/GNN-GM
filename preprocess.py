import json
import shutil
from multiprocessing import Pool

import chess
import numpy as np
import torch
from tqdm import tqdm
import argparse
from pathlib import Path

# Values: PAWN: 1, KNIGHT: 3, BISHOP: 3, ROOK: 5, QUEEN: 9, KING: 255
PIECE_VALUES = np.asarray([1, 3, 3, 5, 9, 255], dtype=np.uint8)

NODE_PATH = "nodes"
EDGES_PATH = "edges"

CHESS_NB_CASE = 64
CHESS_EMBEDDING_SIZE = 15


def process_file(lichess_dataset: Path, out_dataset: Path, dataset_size: int, nb_process: int):
    in_f = lichess_dataset.open('r')

    node_out = out_dataset / NODE_PATH
    edges_out = out_dataset / EDGES_PATH

    node_out.mkdir(parents=True)
    edges_out.mkdir(parents=True)

    with tqdm(total=dataset_size - 1) as pbar:
        with Pool(processes=nb_process) as pool:
            cp_buf = np.zeros(dataset_size, dtype=np.int16)

            it = pool.imap_unordered(process, in_f, chunksize=100)

            i = 0
            for data_list in it:
                for nodes, edges, cp in data_list:
                    sub_dir = f"{i % 255:02x}"

                    (node_out / sub_dir).mkdir(parents=True, exist_ok=True)
                    (edges_out / sub_dir).mkdir(parents=True, exist_ok=True)

                    np.save(node_out / sub_dir / f"nodes_{i}", nodes)
                    np.save(edges_out / sub_dir / f"edges_{i}", edges)

                    cp_buf[i] = cp

                    if i >= dataset_size - 1:
                        np.save(out_dataset / "cp", cp_buf)

                        in_f.close()

                        dataset_info = {"size": dataset_size, "edges_dir": EDGES_PATH, "nodes_dir": NODE_PATH}
                        with (out_dataset / "dataset.json").open('w') as f:
                            json.dump(dataset_info, f)

                        return

                    pbar.update()
                    i += 1


def process(line: str) -> [(np.ndarray, np.ndarray, int)]:
    data = json.loads(line)
    pvs = data['evals'][0]['pvs'][0]

    board = chess.Board(data['fen'])
    node1 = embedding(board)
    node2 = invert_color(node1)

    moves = map(lambda move: [move.from_square, move.to_square], board.legal_moves)
    edges_list = np.fromiter(moves, dtype=np.dtype((np.uint8, 2))).T

    if 'mate' in pvs:
        mate = pvs['mate']
        if mate > 0:
            cp = 20_000
        else:
            cp = -20_000
    else:
        cp = pvs['cp']

    return [(node1, edges_list, cp), (node2, edges_list, -cp)]


def embedding(board: chess.Board) -> np.ndarray:
    """
    |-------|-------------------------------------------------------------------|
    | Index | Description                                                       |
    |-------|-------------------------------------------------------------------|
    |   0   | 1 if piece is Pawn                                                |
    |   1   | 1 if piece is Knight                                              |
    |   2   | 1 if piece is Bishop                                              |
    |   3   | 1 if piece is Rook                                                |
    |   4   | 1 if piece is Queen                                               |
    |   5   | 1 if piece is King                                                |
    |   6   | Piece value                                                       |
    |   7   | 1 if piece is White                                               |
    |   8   | 1 if piece is Black                                               |
    |   9   | Turn (1 for White, 0 for Black)                                   |
    |   10  | White kingside castling rights (1 for presence, 0 for absence)    |
    |   11  | White queenside castling rights (1 for presence, 0 for absence)   |
    |   12  | Black kingside castling rights (1 for presence, 0 for absence)    |
    |   13  | Black queenside castling rights (1 for presence, 0 for absence)   |
    |   14  | En passant target square (-1 if none)                             |
    |-------|-------------------------------------------------------------------|
    """
    nodes = np.zeros((CHESS_NB_CASE, CHESS_EMBEDDING_SIZE), dtype=np.uint8)

    for case, piece in board.piece_map().items():
        piece_type = piece.piece_type - 1  # PAWN: 0, KNIGHT: 1, BISHOP: 2, ROOK: 3, QUEEN: 4, KING: 5
        piece_color = piece.color  # WHITE: True, BLACK: False
        piece_value = PIECE_VALUES[piece_type]

        color_index = 7 + int(not piece_color)

        nodes[case, piece_type] = 1
        nodes[case, 6] = piece_value
        nodes[case, color_index] = 1

    nodes[:, 9] = board.turn
    nodes[:, 10] = board.has_kingside_castling_rights(chess.WHITE)
    nodes[:, 11] = board.has_queenside_castling_rights(chess.WHITE)
    nodes[:, 12] = board.has_kingside_castling_rights(chess.BLACK)
    nodes[:, 13] = board.has_queenside_castling_rights(chess.BLACK)
    nodes[:, 14] = 0xff if board.ep_square is None else board.ep_square

    return nodes


def invert_color(node: np.ndarray) -> np.ndarray:
    res = np.copy(node)
    res[0, 7:9] = res[0, 8:6:-1]
    return res


def main():
    parser = argparse.ArgumentParser(description="Process lichess JSONL file.")

    parser.add_argument(
        "lichess_file",
        type=Path,
        help="Path to the lichess JSONL file.",
    )
    parser.add_argument(
        "output_directory",
        type=Path,
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--size",
        type=int,
        required=True,
        help="Final length of the Dataset.",
    )
    parser.add_argument(
        "--process",
        type=int,
        default=2,
        help="Number of reading process.",
    )

    args = parser.parse_args()

    lichess_file = args.lichess_file
    output_dir = args.output_directory
    size = args.size
    process = args.process

    if not lichess_file.exists():
        raise FileNotFoundError(f"Lichess file not found: {lichess_file}")

    if output_dir.exists():
        print("Cleaning previous dataset...")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Processing...")

    try:
        process_file(lichess_file, output_dir, size, process)
        print(f"Processing complete. Results saved to: {output_dir}")
    except Exception as e:
        print(f"An error occurred during processing: {e}")


if __name__ == "__main__":
    main()
