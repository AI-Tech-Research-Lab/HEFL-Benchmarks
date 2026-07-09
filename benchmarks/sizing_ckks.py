"""
Ciphertext and public-key sizing for CKKS (OpenFHE).

For each ring dimension, builds a 128-bit-secure CKKS context, encrypts a
random real-valued vector, serializes the ciphertext and the public key to
disk, and reports their sizes. Used to estimate the bandwidth overhead of
exchanging encrypted model updates in HE-FL.
"""

import argparse
import os

import numpy as np
import openfhe as fhe

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "ckks")


def build_context(ring_dim, depth, first_mod_size, scaling_mod_size):
    params = fhe.CCParamsCKKSRNS()
    params.SetSecurityLevel(fhe.SecurityLevel.HEStd_128_classic)
    params.SetRingDim(ring_dim)
    params.SetMultiplicativeDepth(depth)
    params.SetFirstModSize(first_mod_size)
    params.SetScalingModSize(scaling_mod_size)
    params.SetScalingTechnique(fhe.ScalingTechnique.FLEXIBLEAUTO)

    cc = fhe.GenCryptoContext(params)
    cc.Enable(fhe.PKESchemeFeature.PKE)
    cc.Enable(fhe.PKESchemeFeature.KEYSWITCH)
    cc.Enable(fhe.PKESchemeFeature.LEVELEDSHE)
    return cc


def size_for(ring_dim, depth, first_mod_size, scaling_mod_size):
    cc = build_context(ring_dim, depth, first_mod_size, scaling_mod_size)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)

    # Random real-valued vector filling all N/2 CKKS slots
    x = np.random.uniform(-1.0, 1.0, ring_dim // 2)
    np.save(os.path.join(OUTPUT_DIR, f"values_N{ring_dim}_D{depth}.npy"), x)

    ptxt = cc.MakeCKKSPackedPlaintext(x)
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
                        default=[16384, 32768, 65536, 131072])
    parser.add_argument("--depth", type=int, default=2, help="multiplicative depth")
    parser.add_argument("--first-mod-size", type=int, default=60)
    parser.add_argument("--scaling-mod-size", type=int, default=59)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for ring_dim in args.ring_dims:
        size_for(ring_dim, args.depth, args.first_mod_size, args.scaling_mod_size)


if __name__ == "__main__":
    main()
