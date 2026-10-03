# Importing dataset.latent (or any of its submodules, since Python runs this file
# first) must register every encoder, regardless of which one a caller asks for by
# name -- otherwise latent_registry.get("bigvgan-mel") fails unless something already
# happened to import bigvgan_mel.py directly first.
from dataset.latent import sao3  # noqa: F401

# bigvgan_mel was deregistered when the mel-vocoder arm was archived (it had no caller but
# this import). It now lives at experiments/oneoff_analyses/bigvgan_mel.py; the archived
# config/archive/*-mel.yaml runs need that import restored here to resolve the encoder name.
