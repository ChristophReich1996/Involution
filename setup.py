from setuptools import setup

requirements = [
    'torch>=1.7.0'
]

setup(
    name="involution",
    version="0.1",
    url="https://github.com/ChristophReich1996/Involution",
    license='MIT License',
    author="Christoph Reich",
    author_email='ChristophReich@gmx.net',
    description="PyTorch 2d Involution",
    install_requires=requirements,
)
