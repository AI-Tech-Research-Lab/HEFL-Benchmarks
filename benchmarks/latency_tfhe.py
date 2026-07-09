"""
Latency benchmark for TFHE (Zama Concrete).

Simulates the server-side aggregation step of FedAvg over encrypted model
updates: a compiled circuit accumulates each (encrypted) client contribution
weighted by a clear factor. Reports average encryption, homomorphic
evaluation, plaintext-baseline evaluation, and decryption times.
"""

import argparse
import time

import numpy as np
from concrete import fhe


def fed_avg_step(out, x, w):
    """One FedAvg accumulation step: running sum + weighted client update."""
    return out + x * w


def build_circuit():
    compiler = fhe.Compiler(fed_avg_step, {
        "out": "encrypted",
        "x": "encrypted",
        "w": "clear",
    })

    # Input set bounding the circuit: (running sum, client update, weight)
    inputset = [(0, 0, 2), (10000, 50, 2)]
    circuit = compiler.compile(inputset)
    circuit.keygen()
    return circuit


def run_once(num_clients):
    client_weights = [np.uint16(np.random.randint(10, 50)) for _ in range(num_clients)]
    avg_factor = np.uint16(2)

    circuit = build_circuit()

    # Encryption of the initial accumulator
    start = time.time()
    out, _, _ = circuit.encrypt(0, 0, 0)
    enc_time = time.time() - start

    # Homomorphic aggregation, one run per client
    eval_time = 0.0
    for w in client_weights:
        _, enc_w, _ = circuit.encrypt(0, w, 0)
        start = time.time()
        out = circuit.run(out, enc_w, avg_factor)
        eval_time += time.time() - start

    # Plaintext baseline of the same aggregation
    start = time.time()
    res = 0
    for w in client_weights:
        res = res + avg_factor * w
    plain_time = time.time() - start

    # Decryption
    start = time.time()
    _ = circuit.decrypt(out)
    dec_time = time.time() - start

    return enc_time, plain_time, eval_time, dec_time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clients", type=int, nargs="+", default=[10, 50, 100],
                        help="numbers of federated clients to evaluate")
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args()

    for num_clients in args.clients:
        print(f"\nEvaluation for {num_clients} clients ({args.repetitions} repetitions):")

        enc, plain, eval_, dec = 0.0, 0.0, 0.0, 0.0
        for _ in range(args.repetitions):
            times = run_once(num_clients)
            enc += times[0]
            plain += times[1]
            eval_ += times[2]
            dec += times[3]

        n = args.repetitions
        print(f"Avg encryption time:            {enc / n:.6f} seconds")
        print(f"Avg plain evaluation time:      {plain / n:.6f} seconds")
        print(f"Avg encrypted evaluation time:  {eval_ / n:.6f} seconds")
        print(f"Avg ratio enc/plain:            {eval_ / plain:.6f}")
        print(f"Avg decryption time:            {dec / n:.6f} seconds\n")


if __name__ == "__main__":
    main()
