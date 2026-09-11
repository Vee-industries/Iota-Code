"""
resume_policy.py -- one fresh-start switch for every resume cache in the pipeline (v1.0.2).

The per-cell resume caches (Q0057 cells, kraskov anchors.json, the partition-B phase files with their
LIVE_SENTINEL stamps) are right for a collection you walk away from and re-enter after a crash. They are
wrong for a re-run: a fire of Run 0057 / Run 0058 on refreshed inputs silently re-emits cached cells with a
new timestamp. IOTA_FRESH=1 makes every resume site treat its cache as absent. reproduce.py sets it.
"""
import os


def fresh():
    return os.environ.get('IOTA_FRESH', '') == '1'
