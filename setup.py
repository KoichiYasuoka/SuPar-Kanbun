import os,setuptools
with open("README.md","r",encoding="utf-8") as r:
  long_description=r.read()
URL="https://github.com/KoichiYasuoka/SuPar-Kanbun"

setuptools.setup(
  name="suparkanbun",
  version="1.5.6",
  description="Tokenizer POS-tagger and Dependency-parser for Classical Chinese",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url=URL,
  author="Koichi Yasuoka",
  author_email="yasuoka@kanji.zinbun.kyoto-u.ac.jp",
  license="MIT",
  keywords="NLP Chinese",
  packages=setuptools.find_packages(),
  install_requires=[
    "supar>=1.1.4",
    "torch<2.6",
    "transformers<4.45",
    "spacy>=2.2.2",
    "deplacy>=2.1.0"
  ],
  python_requires=">=3.7",
  package_data={"suparkanbun":["models/*.txt","models/*/*.txt","models/*/*.json"]},
  classifiers=[
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Operating System :: OS Independent",
    "Topic :: Text Processing :: Linguistic"
  ],
  project_urls={
    "Source":URL,
    "Tracker":URL+"/issues",
  }
)
