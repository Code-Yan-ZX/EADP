"""Restore only one missing, unrelated submission-helper import.

The EADP checkout's tracked files remain byte-identical. The replacement file is
an unmodified, pinned official VLMEvalKit source; see provenance.json. It is not
used for generation or scoring on DocVQA/AI2D.
"""
import importlib.abc
import importlib.util
from pathlib import Path
import sys


class MissingSubmissionHelper(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "vlmeval.utils.result_transfer":
            return importlib.util.spec_from_file_location(fullname, Path(__file__).with_name("result_transfer.py"))
        return None


sys.meta_path.insert(0, MissingSubmissionHelper())
