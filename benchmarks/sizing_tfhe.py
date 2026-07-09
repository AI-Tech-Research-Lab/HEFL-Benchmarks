"""
Ciphertext and key sizing for TFHE (Zama Concrete).

For each integer bit width, compiles a minimal circuit, encrypts a value at
the edge of the range, serializes the resulting ciphertext and the evaluation
keys to disk, and reports their sizes. Used to estimate the bandwidth
overhead of exchanging encrypted model updates in HE-FL.
"""

import argparse
import os

import numpy as np
from concrete import fhe

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "tfhe")


def build_circuit(max_value):
    compiler = fhe.Compiler(lambda x: x + 1, {"x": "encrypted"})

    # Input set bounding the circuit to the desired bit width
    inputset = [0, max_value]
    circuit = compiler.compile(inputset)
    circuit.keygen()
    return circuit


def size_for(bits):
    max_value = 2 ** bits - 1
    circuit = build_circuit(max_value)

    value = np.random.randint(0, max_value, dtype=np.uint64)
    np.save(os.path.join(OUTPUT_DIR, f"values_u{bits}.npy"), value)

    ctxt = circuit.encrypt(int(value))

    ctxt_path = os.path.join(OUTPUT_DIR, f"ctxt_u{bits}.bin")
    keys_path = os.path.join(OUTPUT_DIR, f"evalkeys_u{bits}.bin")
    with open(ctxt_path, "wb") as f:
        f.write(ctxt.serialize())
    with open(keys_path, "wb") as f:
        f.write(circuit.client.evaluation_keys.serialize())

    print(f"u{bits:<3d}  "
          f"ciphertext: {os.path.getsize(ctxt_path):12,d} B  "
          f"evaluation keys: {os.path.getsize(keys_path):12,d} B")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bit-widths", type=int, nargs="+", default=[8, 16],
                        help="integer bit widths to evaluate")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for bits in args.bit_widths:
        size_for(bits)


if __name__ == "__main__":
    main()
