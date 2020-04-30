from setuptools import setup, find_packages
import os


def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

req = parse_requirements(filename=os.path.join(CURRENT_PATH, 'requirements.txt'))

# req = [str(ir.req) for ir in req]

# print(req)
with open(os.path.join(CURRENT_PATH, 'README.md'), 'r', encoding='utf-8') as f:
    long_description = f.read()
exec(open('baconian/version.py').read())

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
    long_description_content_type='text/markdown',
    packages=find_packages(include=['baconian', 'baconian.*'], exclude=[]),
    python_requires='>=3.5',
    include_package_data=True,
    package_data={'baconian': ['config/required_keys/*', 'benchmark/**/*', './*']}
)
