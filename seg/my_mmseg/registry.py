# -*- coding: utf-8 -*-
"""
Bridge registry for local mmseg package.
It re-exports mmengine.registry so that
`from mmseg.registry import MODELS` still works.
"""

from mmengine.registry import MODELS, DATASETS, RUNNERS, TASK_UTILS, VISUALIZERS

__all__ = ['MODELS', 'DATASETS', 'RUNNERS', 'TASK_UTILS', 'VISUALIZERS']
