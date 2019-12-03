import chess.engine
import random
from models import AutoEncoder, BoardValuator
from utils import *
import chess

def main():
    random.seed(0)

    model_path = 'models/axia_15_626000.pt'
    player = Player(model_path)

    engine = chess.engine.SimpleEngine.popen_uci(
        'stockfish/Mac/stockfish-10-64')

    all_results = []
    for i in range(0, 21, 3):
        print('Level: ', i)
        engine.configure({'Skill Level': i})

        # white = random.randint(0, 1)
        white = 1
        board = chess.Board()
        results = []
        for _ in range(10):
            while not board.is_game_over():
                # print(board)
                if (board.turn + white) % 2 == 0:
                    move = player.play(board, white)
                    board.push(move)
                    # print('Our move:', move, white)
                else:
                    result = engine.play(board, chess.engine.Limit(time=0.0100))
                    board.push(result.move)
                    # print('Stockfish move:', result.move, white)
                white += 1

            print(board.result())
            results.append(board.result())
        all_results.append(results)

    print(results)


class Player:
    def __init__(self, model_path):
        ae = AutoEncoder()
        model = BoardValuator(ae)

        model.load_state_dict(
            torch.load(model_path,
                       map_location=torch.device('cpu')))

        model.eval()
        self.model = model

    def play(self, board, white):
        all_inputs = []
        legal_moves = list(board.legal_moves)
        for move in legal_moves:
            new_board = chess.Board(board.fen())
            new_board.push(move)
            inputs, counts, _ = self.get_inputs(new_board)
            all_inputs.append((inputs, counts, 0))
        new_board = chess.Board()
        inputs, counts, _ = self.get_inputs(new_board)
        all_inputs.append((inputs, counts, 0))

        in_batch, mask_batch, _ = collate_fn(all_inputs)
        values = self.model(in_batch, mask_batch).detach().numpy()[:-1]
        if white:
            return legal_moves[np.argmax(values)]
        else:
            return legal_moves[np.argmin(values)]


    def get_inputs(self, board):
        idxs = get_idxs(board)
        bitboard = idxs_to_bitboard(idxs)
        inputs = defaultdict(lambda: [])
        counts = defaultdict(lambda: [0, 0])
        positions = defaultdict(lambda: [])
        for idx in idxs:
            if idx < RAW_BITBOARD_DIM:
                color, pos, piece = idx_to_piece(idx)
                input = torch.Tensor(append_pos(bitboard, pos))
                if color == 0:
                    inputs[PIECES[piece]].insert(0, input)
                    positions[PIECES[piece]].insert(0, pos_to_str(pos))
                else:
                    inputs[PIECES[piece]].append(input)
                    positions[PIECES[piece]].append(pos_to_str(pos))
                counts[PIECES[piece]][color] += 1
        return inputs, counts, positions

    def eval(self, board):
        inputs, counts, _ = self.get_inputs(board)
        in_batch, mask_batch, _ = collate_fn([(inputs, counts, 0)])
        return self.model(in_batch, mask_batch).item()

    def get_values(self, board):
        inputs, counts, positions = self.get_inputs(board)
        new_inputs, new_counts, _ = self.get_inputs(chess.Board())
        all_inputs = [(inputs, counts, 0), (new_inputs, new_counts, 0)]

        in_batch = collate_fn(all_inputs)[0]
        result = {}
        for piece in PIECES:
            values = self.model.models[piece](in_batch[piece])
            for i, pos in enumerate(positions[piece]):
                result[pos] = values[i].item()
        return result

if __name__ == '__main__':
    main()
