from setuptools import setup
import os
import subprocess


def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


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

req = parse_requirements(filename=os.path.join(CURRENT_PATH, req_file))

# req = [str(ir.req) for ir in req]

# print(req)
with open(os.path.join(CURRENT_PATH, 'README.md'), 'r', encoding='utf-8') as f:
    long_description = f.read()
exec(open('./baconian/__version__.py').read())

ver = __version__
setup(
    name='baconian',
    version=ver,
    url='https://github.com/cap-ntu/baconian-project',
    license='MIT License',
    author='Linsen Dong',
    author_email='linsen001@e.ntu.edu.sg',
    description='model-based reinforcement learning toolbox',
    install_requires=req,
    long_description=long_description,
    long_description_content_type='text/markdown'
)
