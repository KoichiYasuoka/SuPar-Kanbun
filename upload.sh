#! /bin/sh
rm -fr build dist *.egg-info
python3 setup.py bdist_wheel
git status
twine upload --repository pypi dist/*
exit 0
