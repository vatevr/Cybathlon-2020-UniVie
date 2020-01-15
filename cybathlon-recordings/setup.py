import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cybathlon-2019-eeg-recordings-vatevr", # Replace with your own username
    version="1.0.0",
    author="Hamlet Mkrtchyan",
    author_email="hamlet.mkrtchyan@protonmail.ch",
    description="A database CRUD layer for Cybathlon 2020 Challenge",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vatevr/Cybathlon-2020-UniVie",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)