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


def process_file(lichess_dataset: Path, out_dataset: Path, dataset_size: int, chunk_size: int, nb_process: int):
    in_f = lichess_dataset.open('r')

    node_out = out_dataset / NODE_PATH
    edges_out = out_dataset / EDGES_PATH

    node_out.mkdir(parents=True)
    edges_out.mkdir(parents=True)

    with Pool(processes=nb_process) as pool:
        cp_buf = np.zeros(dataset_size, dtype=np.int16)

        nodes_buf, edges_buf = {}, {}
        current_chunk_id = 0

        it = pool.imap_unordered(process, in_f, chunksize=100)

        for i, (nodes, edges, cp) in tqdm(enumerate(it), total=dataset_size - 1):
            chunk_id = i // chunk_size
            id_in_chunk = i % chunk_size

            if chunk_id != current_chunk_id:
                print(f"\nFlushing chunk {current_chunk_id}")
                np.savez(node_out / f"nodes_{current_chunk_id}", **nodes_buf)
                np.savez(edges_out / f"edges_{current_chunk_id}", **edges_buf)

                nodes_buf = {}
                edges_buf = {}
                current_chunk_id += 1

            cp_buf[i] = cp

            nodes_buf[f"node_{id_in_chunk}"] = nodes
            edges_buf[f"edges_{id_in_chunk}"] = edges

            if i >= dataset_size - 1:
                break

    np.savez(node_out / f"nodes_{current_chunk_id}", **nodes_buf)
    np.savez(edges_out / f"edges_{current_chunk_id}", **edges_buf)

    np.save(out_dataset / "cp", cp_buf)

    in_f.close()

    dataset_info = {"size": dataset_size, "chunk_size": chunk_size, "edges_dir": EDGES_PATH, "nodes_dir": NODE_PATH}
    with (out_dataset / "dataset.json").open('w') as f:
        json.dump(dataset_info, f)


def process(line: str) -> (np.ndarray, np.ndarray, int):
    data = json.loads(line)
    pvs = data['evals'][0]['pvs'][0]

    board = chess.Board(data['fen'])
    nodes = embedding(board)

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

    return nodes, edges_list, cp


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
    nodes = np.zeros((64, 15), dtype=np.uint8)

    for case, piece in board.piece_map().items():
        piece_type = piece.piece_type - 1  # PAWN: 0, KNIGHT: 1, BISHOP: 2, ROOK: 3, QUEEN: 4, KING: 5
        piece_color = piece.color  # WHITE: True, BLACK: False
        piece_value = PIECE_VALUES[piece_type]

        nodes[case, piece_type] = 1
        nodes[case, 6] = piece_value
        nodes[case, 7 + (not piece_color)] = 1

    nodes[:, 9] = board.turn
    nodes[:, 10] = board.has_kingside_castling_rights(chess.WHITE)
    nodes[:, 11] = board.has_queenside_castling_rights(chess.WHITE)
    nodes[:, 12] = board.has_kingside_castling_rights(chess.BLACK)
    nodes[:, 13] = board.has_queenside_castling_rights(chess.BLACK)
    nodes[:, 14] = 0xff if board.ep_square is None else board.ep_square

    return nodes


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
        "--chunk-size",
        type=int,
        default=100_000,
        help="Number of lines to write in each chunk (default: 100 000).",
    )
    parser.add_argument(
        "--size",
        type=int,
        required=True,
        help="Maximum number of lines to process.",
    )
    parser.add_argument(
        "--process",
        type=int,
        default=4,
        help="Number of process.",
    )

    args = parser.parse_args()

    lichess_file = args.lichess_file
    output_dir = args.output_directory
    size = args.size
    chunk_size = args.chunk_size
    process = args.process

    if not lichess_file.exists():
        raise FileNotFoundError(f"Lichess file not found: {lichess_file}")

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        process_file(lichess_file, output_dir, size, chunk_size, process)
        print(f"Processing complete. Results saved to: {output_dir}")
    except Exception as e:
        print(f"An error occurred during processing: {e}")


if __name__ == "__main__":
    main()
