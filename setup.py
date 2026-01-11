from setuptools import find_packages, setup

def load_requirements(filename="requirements.txt"):
    with open(filename, "r") as f:
        return [
            line.strip()
            for line in f
            if line.strip() 
            and not line.startswith("#")
            and not line.startswith("-")
        ]

setup(
    name='ml-project',
    version='0.0.1',
    author='himanshu',
    author_email='kumarehimanshu1@gmail.com',
    packages=find_packages(),
    install_requires=load_requirements()
)