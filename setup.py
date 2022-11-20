import codecs
import setuptools


setuptools.setup(
    name="bnaug",
    version="1.1.0",
    author="Sagor Sarker",
    author_email="sagorhem3532@gmail.com",
    description="bnaug is a text augmentation tool for Bangla text.",
    long_description=codecs.open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sagorbrur/bnaug",
    license="MIT",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        "bnlp_toolkit==3.2.0",
        "transformers==4.24.0"
    ]
)
