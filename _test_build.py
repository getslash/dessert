#! /usr/bin/python
import sys
import subprocess

if __name__ == '__main__':
    modules = ["pytest"]
    subprocess.check_call("pip install --use-mirrors {0}".format(" ".join(modules)), shell=True)
    subprocess.check_call("python setup.py develop", shell=True)
    subprocess.check_call("py.test tests", shell=True)
