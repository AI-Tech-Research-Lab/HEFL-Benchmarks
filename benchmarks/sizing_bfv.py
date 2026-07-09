"""
Ciphertext and public-key sizing for BFV (OpenFHE).

For each ring dimension, builds a 128-bit-secure BFV context, encrypts a
random integer vector, serializes the ciphertext and the public key to disk,
and reports their sizes. Used to estimate the bandwidth overhead of
exchanging encrypted model updates in HE-FL.
"""

import argparse
import os

import numpy as np
import openfhe as fhe

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "bfv")


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


def size_for(ring_dim, depth, plaintext_modulus):
    cc = build_context(ring_dim, depth, plaintext_modulus)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)

    # Random integer vector filling all plaintext slots
    values = np.random.randint(0, 1000, size=ring_dim, dtype=np.uint64)
    np.save(os.path.join(OUTPUT_DIR, f"values_N{ring_dim}_D{depth}.npy"), values)

    ptxt = cc.MakePackedPlaintext(values.tolist())
    ctxt = cc.Encrypt(keys.publicKey, ptxt)

    ctxt_path = os.path.join(OUTPUT_DIR, f"ctxt_N{ring_dim}_D{depth}.bin")
    pk_path = os.path.join(OUTPUT_DIR, f"pk_N{ring_dim}_D{depth}.bin")
    fhe.SerializeToFile(ctxt_path, ctxt, fhe.BINARY)
    fhe.SerializeToFile(pk_path, keys.publicKey, fhe.BINARY)

    print(f"N={ring_dim:7d}  D={depth}  "
          f"ciphertext: {os.path.getsize(ctxt_path):12,d} B  "
          f"public key: {os.path.getsize(pk_path):12,d} B")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ring-dims", type=int, nargs="+",
                        default=[8192, 16384, 32768, 65536])
    parser.add_argument("--depth", type=int, default=2, help="multiplicative depth")
    parser.add_argument("--plaintext-modulus", type=int, default=65537)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for ring_dim in args.ring_dims:
        size_for(ring_dim, args.depth, args.plaintext_modulus)


if __name__ == "__main__":
    main()
