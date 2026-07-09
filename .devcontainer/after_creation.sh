#!/bin/bash

# Inside the dev container, use the prebuilt openfhe-python bindings from the
# image; in a manual setup, openfhe-python must already be at the repo root
# (built against OpenFHE v1.2.4).
if [ -d "/openfhe-python" ]; then
    rm -rf "/workspaces/Test/openfhe-python"
    cp -r "/openfhe-python" "/workspaces/Test"
fi
python3 -m pip install -U setuptools pip
python3 -m pip install -e openfhe-python
python3 -m pip install numpy==1.26.4
python3 -m pip install -q concrete-ml==1.9.0
