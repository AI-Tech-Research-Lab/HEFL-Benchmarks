"""
Latency benchmark for CKKS (OpenFHE).

Simulates the server-side aggregation step of FedAvg over encrypted model
updates: each client contribution is weighted (EvalMult by a scalar) and
accumulated (EvalAdd). Reports average encryption, homomorphic evaluation,
plaintext-baseline evaluation, and decryption times.
"""

import argparse
import time

import numpy as np
import openfhe as fhe


def build_context(ring_dim, depth, first_mod_size, scaling_mod_size):
    params = fhe.CCParamsCKKSRNS()
    params.SetSecurityLevel(fhe.SecurityLevel.HEStd_128_classic)
    params.SetRingDim(ring_dim)
    params.SetMultiplicativeDepth(depth)
    params.SetFirstModSize(first_mod_size)
    params.SetScalingModSize(scaling_mod_size)
    params.SetScalingTechnique(fhe.ScalingTechnique.FIXEDAUTO)

    cc = fhe.GenCryptoContext(params)
    cc.Enable(fhe.PKESchemeFeature.PKE)
    cc.Enable(fhe.PKESchemeFeature.KEYSWITCH)
    cc.Enable(fhe.PKESchemeFeature.LEVELEDSHE)
    return cc


def run_once(ring_dim, depth, first_mod_size, scaling_mod_size, num_clients):
    cc = build_context(ring_dim, depth, first_mod_size, scaling_mod_size)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)

    # Real-valued vector standing in for a model update (N/2 CKKS slots)
    num_slots = ring_dim // 2
    x = np.random.uniform(-1.0, 1.0, num_slots)
    ones = np.ones(num_slots)

    # Plaintext baseline of the aggregation
    start = time.time()
    res = np.zeros(num_slots)
    for _ in range(num_clients):
        res = res + x * ones
    plain_time = time.time() - start

    # Encryption
    start = time.time()
    ptxt = cc.MakeCKKSPackedPlaintext(x)
    ctxt = cc.Encrypt(keys.publicKey, ptxt)
    enc_time = time.time() - start

    # Homomorphic aggregation: weight (x1.0) each client update, then accumulate
    start = time.time()
    out = cc.EvalMult(ctxt, 1.0)
    for _ in range(num_clients - 1):
        temp = cc.EvalMult(ctxt, 1.0)
        out = cc.EvalAdd(out, temp)
    eval_time = time.time() - start

    # Decryption
    start = time.time()
    _ = cc.Decrypt(keys.secretKey, out)
    dec_time = time.time() - start

    return enc_time, plain_time, eval_time, dec_time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ring-dim", type=int, default=65536)
    parser.add_argument("--depth", type=int, default=2, help="multiplicative depth")
    parser.add_argument("--first-mod-size", type=int, default=60)
    parser.add_argument("--scaling-mod-size", type=int, default=50)
    parser.add_argument("--clients", type=int, nargs="+", default=[10, 50, 100],
                        help="numbers of federated clients to evaluate")
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args()

    for num_clients in args.clients:
        print(f"\nEvaluation for {num_clients} clients "
              f"(N={args.ring_dim}, D={args.depth}, {args.repetitions} repetitions):")

        enc, plain, eval_, dec = 0.0, 0.0, 0.0, 0.0
        for _ in range(args.repetitions):
            times = run_once(args.ring_dim, args.depth, args.first_mod_size,
                             args.scaling_mod_size, num_clients)
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
