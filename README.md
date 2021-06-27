# rp-group-43-common

## SVM

svm_binary.ipynb contains binary classification.
ordinal.ipynb contains ordinal classification.
per_questionnaire.ipynb contains per-questionnaire classification.

In order to run these classifiers, you need fastText model.

### How to download fastText model?

>>> import fasttext.util
>>> fasttext.util.download_model('en', if_exists='ignore')  # English
>>> ft = fasttext.load_model('cc.en.300.bin')


For detailed download information, please check this link: https://fasttext.cc/docs/en/crawl-vectors.html

