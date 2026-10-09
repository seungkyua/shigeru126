# -*- coding: utf-8 -*-
"""디렉토리 구조 변경 후에도 각 프로그램을 어디서 실행하든 import 에러 없이 뜨는지 확인"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ["common/shigeru_126.py", "firstbuy/shigeru_start.py", "buynsell/shigeru_portfolio.py"]


@pytest.mark.parametrize("script", SCRIPTS)
def test_shouldShowHelpWhenRunDirectlyFromOtherDirectory(script, tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / script), "--help"],
                       cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("module", ["common.shigeru_126", "firstbuy.shigeru_start", "buynsell.shigeru_portfolio"])
def test_shouldShowHelpWhenRunAsModuleFromRoot(module):
    r = subprocess.run([sys.executable, "-m", module, "--help"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_shouldReadSampleByFileNameFromRoot():
    """portfolio_sample.csv 처럼 파일 이름만 줘도 buynsell/ 에서 찾아 읽는다 (네트워크 없이 --help 대신 load 확인)."""
    r = subprocess.run([sys.executable, "-c",
                        "from buynsell import shigeru_portfolio as pf; "
                        "print(len(pf.load_trades(pf.resolve_csv('portfolio_sample.csv'))))"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and int(r.stdout.strip()) > 0, r.stderr
