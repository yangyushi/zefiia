from setuptools import setup
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'readme.md'), encoding='utf-8') as f:
    long_description = f.read()

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name='zefiia',
    version='0.1.0',
    description='A Python module for zebrafish image analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yangyushi/zefiia',
    author='Yushi Yang',
    author_email='yangyushi1992@icloud.com',  # Optional
    packages=["zefiia"],
    package_dir={'zefiia': 'zefiia'},
    install_requires=[
        'numpy', 'scipy', 'scikit_posthocs', 'numba', 'tqdm'
    ],
    python_requires='>=3.5'
)
