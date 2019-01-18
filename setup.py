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

setup(
    name="lastochka",
    version=__version__,
    description="Lastochka - Weight Of Evidence Transformer",
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/rst',
    author=MAINTAINER,
    url='https://github.com/sberbank-ai/lastochka',
    license=LICENSE,
    packages=find_packages(exclude=('tests',)),
    include_package_data=True,
    install_requires=["numpy", "pandas", "scikit-learn", "tqdm"],
    zip_safe=False
)
