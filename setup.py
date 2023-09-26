import os
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """
    Overrides the base extension class so that setuptools
    doesn't try to build your sources for you
    """

    def __init__(self, name, sources=[]):
        super().__init__(name=name, sources=sources)


class BuildCMakeExt(build_ext):
    def run(self):
        for extension in self.extensions:
            if extension.name == "xft":
                self.build_cmake()
            else:
                super().run()

    def build_cmake(self):
        self.announce("Preparing the build environment", level=3)
        build_dir = os.path.abspath(os.path.dirname(self.build_temp))
        os.makedirs(build_dir, exist_ok=True)

        self.announce("Building xft binaries", level=3)
        self.spawn(["cmake", f"-B {build_dir}", "."])
        self.spawn(["make", "-C", build_dir, "-j"])


setup(
    name="xfastertransformer",
    version="1.0.0",
    keywords="LLM",
    description="Boost large language model inference performance on CPU platform.",
    long_description="Boost large language model inference performance on CPU platform.",
    license="Apache 2.0",
    url="https://github.com/intel/xFasterTransformer",
    author="xFasterTransformer",
    python_requires=">=3.8",
    package_dir={"": "src"},
    packages=find_packages(where="src", include=["xfastertransformer"]),
    package_data={"xfastertransformer": ["*.so"]},
    platforms="x86_64",
    ext_modules=[CMakeExtension(name="xft")],
    cmdclass={"build_ext": BuildCMakeExt},
    install_requires=[
        'torch>=2.0.0'
    ]
)
