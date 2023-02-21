#!/usr/bin/env python3
"""
    NN agent
"""

def encode_state(state):
    """
        Encode state
    """
    coded_state = None
    Encoded_states = [(0, [0, 0, 1]), (1, [0, 1, 0]), (2, [1, 0, 0])]
    for s in Encoded_states:
        if s[0] == state:
          coded_state = s[1]
    return coded_state
