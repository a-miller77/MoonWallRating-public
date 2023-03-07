import numpy as np
import pandas as pd

def create_vocab(file_path = './data/vocab.txt') -> tuple:
    with open(file_path) as f:
        lines = f.readlines()
        lines = [i[:-1] for i in lines]  # Remove new line characters
    hold_to_idx = {hold: idx for idx, hold in enumerate(lines)}
    idx_to_hold = {idx: hold for idx, hold in enumerate(lines)}
    return (hold_to_idx, idx_to_hold)

hold_to_idx, idx_to_hold = create_vocab()
vocab_size = len(hold_to_idx)

def get_x_y(hold: dict):
    x = ord(hold['Position'][0]) - 64
    y = int(hold['Position'][1:])
    return x, y

def get_closest_hold(current_hold: dict, valid_holds: list) -> dict:
    x1, y1 = get_x_y(current_hold)
    distances = list(map(lambda hold: ((x1 - get_x_y(hold)[0])**2 + (y1 - get_x_y(hold)[1])**2)**0.5, valid_holds))
    return valid_holds[np.argmin(distances)]

def get_next_hold(current_hold: dict, holds: list) -> dict:
    x, y = get_x_y(current_hold)
    min_y = 19
    for hold in holds:
        x2, y2 = get_x_y(hold)
        if y2 <= min_y and y2 >= y:
            min_y = y2
    valid_holds = [hold for hold in holds if get_x_y(hold)[1] == min_y]
    return get_closest_hold(current_hold, valid_holds)

def sort_hand_holds(moves: list, sorted_holds = []) -> list:
    holds = moves.copy()

    if len(sorted_holds) == 0:
        sorted_holds = [hold for hold in holds if hold['IsStart']]
        holds = [hold for hold in holds if hold['IsStart'] == False]

    if sorted_holds[-1]['IsEnd']:
        return list(map(lambda x: x['Position'], sorted_holds))
    
    next_hold = get_next_hold(sorted_holds[-1], holds)
    sorted_holds.append(next_hold)
    holds.remove(next_hold)

    return sort_hand_holds(holds, sorted_holds)

def generate_route_sequence(route) -> list:
    climbing_methods = {
        'Feet follow hands': 'FFH',
        'Feet follow hands + screw ons': 'FFHSO',
        'Footless + kickboard': 'FLKB',
        'Screw ons only': 'SOO'
    }
    
    grade = 'V' + str(route.Grade)
    config = route.MoonboardConfiguration[:3]
    method = climbing_methods[route.Method]
    sorted_holds = sort_hand_holds(route.Moves)

    route_sequence = [grade, config, method] + sorted_holds
        
    return route_sequence

def tokenize_sequence(sequence, max_length=22) -> list:
    tokenized_input = [hold_to_idx[i] for i in sequence]
    tokenized_input += [0] * (max_length - len(tokenized_input))
    return tokenized_input

def add_start_stop(sequence) -> list:
    START_TOK = hold_to_idx['[START]']
    END_TOK = hold_to_idx['[END]']

    sequence = np.concatenate(([START_TOK], sequence))
    sequence[sequence.argmin()] = END_TOK

    return sequence