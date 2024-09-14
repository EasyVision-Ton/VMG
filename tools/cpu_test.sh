#!/usr/bin/env bash



PYTHONPATH="$(dirname $0)/..":PYTHONPATH \
python \
$(dirname $0)/../models/vmg.py
