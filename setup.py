from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

with open('requirements.txt', 'r') as f:
    requirements = f.readlines()

setup(
    name='RoboSurvStratOpt',
    version='1.0.0',
    author=['Connor Hughes', 'Yohan John'],
    author_email=['connorhughes@ucsb.edu', 'yohanjohn@ucsb.edu'],
    description='Repository for research code related to robotic surveillance problems.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/conhugh/RoboSurvStratOpt',
    packages=['src'],
    install_requires=requirements,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
)
