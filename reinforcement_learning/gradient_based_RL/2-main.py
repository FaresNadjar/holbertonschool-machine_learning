#!/usr/bin/env python3

"""
    Main file
"""

if __name__ == '__main__':
    encode_state = __import__('2-encode_state').encode_state
    state = 1
    enc_state = encode_state(state)
    print("The original state is ", state, " and the encoded state is ", enc_state)