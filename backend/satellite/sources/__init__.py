"""Satellite source plugins. Each source implements the small interface in
base.py: get_latest_metadata() and download(metadata) -> PIL.Image plus the
observation timestamp. pipeline.py doesn't know or care which source is
active."""
