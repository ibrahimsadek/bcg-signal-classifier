#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Entry point for BCG signal classification pipeline."""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress INFO + WARNING logs

from bcg_signal_classifier.pipeline import main

if __name__ == "__main__":
    main()
