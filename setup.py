from setuptools import setup

install_requires = []
with open("./requirements.txt", encoding="utf-8") as requirements_file:
    reqs = [r.strip() for r in requirements_file.readlines()]
    reqs = [r for r in reqs if r and r[0] != "#"]
    for r in reqs:
        install_requires.append(r)

setup(
    name="scaled-rope",
    version="0.1",
    packages=["scaled_rope"],
    install_requires=install_requires,
    url='https://github.com/jquesnelle/scaled-rope/',
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ]
)