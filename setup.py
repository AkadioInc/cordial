from setuptools import setup, find_packages
from pathlib import Path
import cordial

with Path('./README.md').open('rt', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='cordial',
    version=cordial.__version__,
    description='Classes, scripts, and tools for the CoRDIAL project',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ajelenak/cordial',
    author='Akadio Inc.',
    author_email='aleksandar.jelenak@gmail.com',
    classifiers=[
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD 3-Clause License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'],
    # keywords='',
    # py_modules=[],
    packages=find_packages(exclude=['notebooks', 'docs', 'tests']),
    python_requires='>=3.6',
    install_requires=[
        'obspy>=1.1.1',
        'h5py_switch @ git+https://github.com/ajelenak/h5py_switch.git@master#egg=h5py_switch',
        's3fs'],
    # extras_require={},
    # package_data={},
    # data_files=[],
    # entry_points={},
    # project_urls={},
)
