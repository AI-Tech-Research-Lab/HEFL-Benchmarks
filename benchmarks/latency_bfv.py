"""
Latency benchmark for BFV (OpenFHE).

Simulates the server-side aggregation step of FedAvg over encrypted model
updates: each client contribution is weighted (EvalMult by a plaintext) and
accumulated (EvalAdd). Reports average encryption, homomorphic evaluation,
plaintext-baseline evaluation, and decryption times.
"""

import argparse
import time

import numpy as np
import openfhe as fhe


def build_context(ring_dim, depth, plaintext_modulus):
    params = fhe.CCParamsBFVRNS()
    params.SetSecurityLevel(fhe.SecurityLevel.HEStd_128_classic)
    params.SetRingDim(ring_dim)
    params.SetMultiplicativeDepth(depth)
    params.SetPlaintextModulus(plaintext_modulus)

    cc = fhe.GenCryptoContext(params)
    cc.Enable(fhe.PKESchemeFeature.PKE)
    cc.Enable(fhe.PKESchemeFeature.KEYSWITCH)
    cc.Enable(fhe.PKESchemeFeature.LEVELEDSHE)
    return cc


def run_once(ring_dim, depth, plaintext_modulus, num_clients):
    cc = build_context(ring_dim, depth, plaintext_modulus)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)

    # Integer vector standing in for a (quantized) model update
    values = np.random.randint(0, 1000, size=ring_dim, dtype=np.uint64)

    # Encryption
    start = time.time()
    ptxt = cc.MakePackedPlaintext(values.tolist())
    ctxt = cc.Encrypt(keys.publicKey, ptxt)
    enc_time = time.time() - start

    # Homomorphic aggregation: weight (x1) each client update, then accumulate
    ones = np.ones(ring_dim, dtype=np.uint64)
    pt_one = cc.MakePackedPlaintext(ones.tolist())

    start = time.time()
    res = cc.EvalMult(ctxt, pt_one)
    for _ in range(num_clients - 1):
        temp = cc.EvalMult(ctxt, pt_one)
        res = cc.EvalAdd(res, temp)
    eval_time = time.time() - start

    # Plaintext baseline of the same aggregation
    start = time.time()
    out = values * ones
    for _ in range(num_clients - 1):
        out = out + values * ones
    plain_time = time.time() - start

    # Decryption
    start = time.time()
    _ = cc.Decrypt(keys.secretKey, res)
    dec_time = time.time() - start

    return enc_time, plain_time, eval_time, dec_time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ring-dim", type=int, default=32768)
    parser.add_argument("--depth", type=int, default=2, help="multiplicative depth")
    parser.add_argument("--plaintext-modulus", type=int, default=65537)
    parser.add_argument("--clients", type=int, nargs="+", default=[10, 50, 100],
                        help="numbers of federated clients to evaluate")
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args()

    for num_clients in args.clients:
        print(f"\nEvaluation for {num_clients} clients "
              f"(N={args.ring_dim}, D={args.depth}, {args.repetitions} repetitions):")

        enc, plain, eval_, dec = 0.0, 0.0, 0.0, 0.0
        for _ in range(args.repetitions):
            times = run_once(args.ring_dim, args.depth, args.plaintext_modulus, num_clients)
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
