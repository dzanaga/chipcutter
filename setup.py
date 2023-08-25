from setuptools import setup, find_packages
from chipcutter import __version__

setup(
    name="chipcutter",
    version=f"{__version__}",
    author="Daniele Zanaga",

    # adding packages
    packages=find_packages(),
    package_data={
    },
    install_requires=[
        'numpy'
    ],
    zip_safe=True,
)
