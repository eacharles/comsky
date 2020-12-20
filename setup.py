from setuptools import setup

import comsky

setup(
    name="comsky",
    version=comsky.__version__,
    author=comsky.__author__,
    author_email=comsky.__author_email__,
    url = comsky.__url__,
    packages=["comsky"],
    description=comsky.__desc__,
    long_description=open("README.md").read(),
    package_data={"": ["README.md", "LICENSE"]},
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        ],
    install_requires=["matplotlib", "numpy", "astropy", "healpy"],
)
