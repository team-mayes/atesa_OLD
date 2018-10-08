import pytest
import importlib
from aimless_shooting import *

def test_handle_groupfile():
    groupfile_list = []
    result = handle_groupfile('test_job')
    assert result == 'groupfile_1'
