import json
from multiprocessing import Pool

import chess
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path

# Values: PAWN: 1, KNIGHT: 3, BISHOP: 3, ROOK: 5, QUEEN: 9, KING: 255
PIECE_VALUES = np.asarray([1, 3, 3, 5, 9, 255], dtype=np.uint8)


def process_file(lichess_dataset: Path, out_dataset: Path, dataset_size: int, chunk_size: int):
    in_f = lichess_dataset.open('r')

    with Pool(processes=1) as pool:
        nodes_buf, edges_buf, cp_buf = {}, {}, []

        last_flush = None
        chunk_id = 0

        it = pool.imap_unordered(process, in_f, chunksize=100)

        for i, (nodes, edges, cp) in tqdm(enumerate(it), total=dataset_size):
            nodes_buf[f"node_{i}"] = nodes
            edges_buf[f"edges_{i}"] = edges

            cp_buf.append(cp)

            if (last_flush is None and i >= chunk_size - 1) or (
                    last_flush is not None and i - last_flush >= chunk_size):
                print(f"\nFlushing chunk {chunk_id}")
                np.savez(out_dataset / f"nodes_{chunk_id}", **nodes_buf)
                np.savez(out_dataset / f"edges_{chunk_id}", **edges_buf)

                nodes_buf = {}
                edges_buf = {}
                last_flush = i
                chunk_id += 1

            if i >= dataset_size - 1:
                break

    np.savez(out_dataset / f"nodes_{chunk_id}", **nodes_buf)
    np.savez(out_dataset / f"edges_{chunk_id}", **edges_buf)

    np.save(out_dataset / "cp", cp_buf)

    in_f.close()
    with (out_dataset / "dataset.json").open('w') as f:
        json.dump({"nb_chunk": chunk_id, "size": dataset_size, "chunk_size": chunk_size}, f)


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

    args = parser.parse_args()

    lichess_file = args.lichess_file
    output_dir = args.output_directory
    size = args.size
    chunk_size = args.chunk_size

    if not lichess_file.exists():
        raise FileNotFoundError(f"Lichess file not found: {lichess_file}")

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        process_file(lichess_file, output_dir, size, chunk_size)
        print(f"Processing complete. Results saved to: {output_dir}")
    except Exception as e:
        print(f"An error occurred during processing: {e}")


if __name__ == "__main__":
    main()
