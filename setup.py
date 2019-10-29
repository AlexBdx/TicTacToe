import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="TicTacToe",
    version="0.0.2",
    author="Alex Bondoux",
    author_email="alexandre.bdx@gmail.com",
    description="A Reinforcement Learning approach to the general TicTacToe games ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AlexBdx/TicTacToe",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: POSIX :: Linux",
    ],
)
