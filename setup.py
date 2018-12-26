from setuptools import setup, find_packages
from lastochka import __version__

DISTNAME = 'lastochka'
DESCRIPTION = 'Weight Of Evidence Transformer with scikit-learn API'

with open('README.md') as f:
    LONG_DESCRIPTION = f.read()

MAINTAINER = 'Ivan Trusov'
MAINTAINER_EMAIL = 'polarpersonal@gmail.com'
URL = 'http://scikit-learn.org'
DOWNLOAD_URL = 'https://pypi.org/project/lastochka/#files'
LICENSE = 'MIT'

setup(
    name="lastochka",
    version=__version__,
    description="Lastochka - Weight Of Evidence Transformer",
    long_description='',
    author='Trusov Ivan',
    url=None,
    license='',
    packages=find_packages(),
    include_package_data=True,
    install_requires=["numpy", "pandas", "scikit-learn"],
    zip_safe=False
)
