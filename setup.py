import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="titansurv-abhi8893", # Replace with your own username
    version="0.0.1",
    author="Abhishek Bhatia",
    author_email="bhatiaabhishek8893@gmail.com",
    description="Titanic Survival Kaggle Competition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abhi8893/Titanic-Survival",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)