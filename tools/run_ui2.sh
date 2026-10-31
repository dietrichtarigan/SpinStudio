#!/bin/sh
cd ~/micromagnetics
rm -rf ui_shots2
node --check spinstudio/static/app.js && node --check spinstudio/static/macro.js && echo JS-OK
.venv/bin/python -u tools/ui_test2.py 2>&1 | grep -v -i "userwarning\|warnings.warn" | tail -15
