from setuptools import setup

authors = [
    ('Yohan John', 'yohanjohn@ucsb.edu'),
    ('Connor Hughes', 'connorhughes@ucsb.edu'),
]

with open("README", 'r') as f:
    long_description = f.read()

with open('requirements.txt', 'r') as f:
    requirements = f.readlines()

setup(
    name='RoboSurvStratOpt',
    version='1.0.0',
    author=', '.join([author[0] for author in authors]),
    author_email=', '.join([author[1] for author in authors]),
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
