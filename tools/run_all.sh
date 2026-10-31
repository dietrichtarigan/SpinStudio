#!/bin/sh
# Full regression suite. Usage on the GPU node:  sh tools/run_all.sh   (log: tests_all.log)
cd ~/micromagnetics
PY=".venv/bin/python -u"
{
  echo "===== verify vs SpinTorch"; $PY tools/verify.py 2>&1 | grep -E "rel\. error|cosine|epoch time|steps/s|loss|legacy|fixed"
  echo "===== API smoke";          $PY tools/smoke_test.py 2>&1 | grep -E "^(simulate|frames|train|history|invalid)"
  echo "===== dispersion";         $PY tools/dispersion_test.py 2>&1 | grep -v -i warn
  echo "===== micromagnetic physics"; $PY tools/physics_tests.py 2>&1 | grep -E "PASS|FAIL|ALL|SOME"
  echo "===== macrospin physics";  $PY tools/macrospin_tests.py 2>&1 | grep -E "PASS|FAIL|ALL|SOME|steps/s"
  echo "===== neuromorphic apps";  $PY tools/apps_tests.py 2>&1 | grep -E "PASS|FAIL|ALL|SOME"
  echo "===== GUI 1";              $PY tools/ui_test.py 2>&1 | tail -2
  echo "===== GUI 2";              $PY tools/ui_test2.py 2>&1 | tail -2
  echo "===== DONE"
} > tests_all.log 2>&1
