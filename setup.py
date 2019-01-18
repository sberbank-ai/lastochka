from setuptools import setup, find_packages
from lastochka import __version__

DISTNAME = 'lastochka'
DESCRIPTION = 'Weight Of Evidence Transformer with scikit-learn API'

with open('README.rst') as f:
    LONG_DESCRIPTION = f.read()

MAINTAINER = 'Ivan Trusov'
MAINTAINER_EMAIL = 'polarpersonal@gmail.com'
URL = 'http://scikit-learn.org'
DOWNLOAD_URL = 'https://pypi.org/project/lastochka/#files'
LICENSE = 'MIT'


CLASSIFIERS = [
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX",
        "Operating System :: POSIX :: BSD",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: CPython",
]

setup(
    name="lastochka",
    version=__version__,
    description="Lastochka - Weight Of Evidence Transformer",
    long_description=LONG_DESCRIPTION,
    author=MAINTAINER,
    url='https://github.com/sberbank-ai/lastochka',
    license=LICENSE,
    packages=find_packages(exclude=('tests',)),
    include_package_data=True,
    install_requires=["numpy", "pandas", "scikit-learn", "tqdm"],
    zip_safe=False
)
