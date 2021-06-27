# rp-group-43-common

## SVM

svm_binary.ipynb contains binary classification.
ordinal.ipynb contains ordinal classification.
per_questionnaire.ipynb contains per-questionnaire classification.

In order to run these classifiers, you need fastText model.
You also need to adjust the file structure if you want to run these classifiers.

### How to download fastText model?

>>> import fasttext.util
>>> fasttext.util.download_model('en', if_exists='ignore')  # English
>>> ft = fasttext.load_model('cc.en.300.bin')


For detailed download information, please check this link: https://fasttext.cc/docs/en/crawl-vectors.html

### Data
Data used for this text classification is not available, as it was not collected to be publicly shared.

