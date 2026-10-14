# Homomorphic Encryption: The Missing Piece in Federated Learning — Benchmarks

Companion code for IEEE TAI survey **"Homomorphic Encryption: The Missing Piece in Federated Learning"**. It provides micro-benchmarks used to quantify the computational and bandwidth overhead of integrating Homomorphic Encryption (HE) into Federated Learning (FL), covering three HE schemes:

| Scheme | Family | Library |
|--------|--------|---------|
| **BFV** | Leveled HE (exact integers) | [OpenFHE](https://github.com/openfheorg/openfhe-development) v1.2.4 ([openfhe-python](https://github.com/openfheorg/openfhe-python) v0.8.10) |
| **CKKS** | Leveled HE (approximate reals) | [OpenFHE](https://github.com/openfheorg/openfhe-development) v1.2.4 ([openfhe-python](https://github.com/openfheorg/openfhe-python) v0.8.10) |
| **TFHE** | Fully HE (bootstrapped) | [Zama Concrete](https://github.com/zama-ai/concrete) (concrete-ml v1.9.0 / concrete-python v2.10.0) |


## Running

To run benchmarks, build the .devcontainer, which provides OpenFHE and Concrete.

Make sure dependencies have been installed successfully, or run:

```bash
bash .devcontainer/after_creation.sh
```

Then, run any benchmark directly:

```bash
python3 benchmarks/latency_ckks.py                     # paper defaults
python3 benchmarks/latency_bfv.py --clients 10 50 100 --repetitions 10
python3 benchmarks/sizing_ckks.py --ring-dims 16384 32768 65536 131072
python3 benchmarks/sizing_tfhe.py --bit-widths 8 16
```

Every script accepts `--help` for the full list of parameters (ring dimension, multiplicative depth, moduli, federation sizes, repetitions).