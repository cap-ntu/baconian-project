from setuptools import setup
from pip.req import parse_requirements

req = parse_requirements(filename='requirement.txt', session='hack')
req = [str(ir.req) for ir in req]
print(req)
setup(
    name='baconian-internal',
    version='0.1',
    url='https://git.withcap.org/Lukeeeeee/mobrl',
    license='MIT License',
    author='Dong Linsen',
    author_email='linsen001@e.ntu.edu.sg',
    description='model-based reinforcement learning toolbox',
    install_requires=req
)
