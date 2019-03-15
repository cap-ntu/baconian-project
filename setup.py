from setuptools import setup
from pip.req import parse_requirements
import os
import subprocess

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
gpu_flag = None
try:
    nvcc_res = subprocess.run(['nvcc', '-V'])
    if nvcc_res.returncode == 0:
        gpu_flag = True
    else:
        gpu_flag = False
except Exception:
    gpu_flag = False

if gpu_flag is True:
    req_file = 'requirement_gpu.txt'
else:
    req_file = 'requirement_nogpu.txt'

req = parse_requirements(filename=os.path.join(CURRENT_PATH, req_file), session='hack')

req = [str(ir.req) for ir in req]

# print(req)
with open(os.path.join(CURRENT_PATH, 'README.md'), 'r') as f:
    long_description = f.read()
exec(open('./baconian/__version__.py').read())

ver = __version__
setup(
    name='baconian',
    version=ver,
    url='https://git.withcap.org/Lukeeeeee/mobrl',
    license='MIT License',
    author='Dong Linsen',
    author_email='linsen001@e.ntu.edu.sg',
    description='model-based reinforcement learning toolbox',
    install_requires=req,
    long_description=long_description,
    long_description_content_type='text/markdown'
)
