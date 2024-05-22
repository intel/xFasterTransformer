import os
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from pathlib import Path

# read the contents of your README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


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


def get_xft_version() -> str:
    xft_root = Path(__file__).parent
    version = open(xft_root / "VERSION").read().strip()

    return version


setup(
    name="xfastertransformer",
    version=get_xft_version(),
    keywords="LLM",
    description="Boost large language model inference performance on CPU platform.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache 2.0",
    url="https://github.com/intel/xFasterTransformer",
    author="xFasterTransformer",
    author_email="xft.maintainer@intel.com",
    python_requires=">=3.8",
    package_dir={"": "src"},
    packages=find_packages(where="src", include=["xfastertransformer"]),
    package_data={
        "xfastertransformer": ["*.so", "tools/*.*"],
    },
    platforms="x86_64",
    ext_modules=[CMakeExtension(name="xft")],
    cmdclass={"build_ext": BuildCMakeExt},
    install_requires=["torch>=2.3.0, <2.4.0"],
)
