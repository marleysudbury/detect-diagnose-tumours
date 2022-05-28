# Detecting metastatic tissue in lymph nodes with deep learning

Individual project worth 40 credits as part of my final year studying BSc Computer Science at Cardiff University.

## Abstract

To track the spread of a tumour, pathologists examine lymph nodes in the surrounding area for metastatic growths. This is done by looking at high resolution scans of slides with tissue taken from the lymph nodes. Deep learning systems have demonstrated a great ability to classify images, and so researchers have looked at applying these techniques to the problem of tissue classification, including lymph node classification, with the aim of creating a system which can aid pathologists. In this project a convolutional neural network (CNN) was used to evaluate two methods for classifying lymph nodes: a whole-slide method, which uses a down scaled version of the entire slide image as its input, and a patch-based method which takes small patches of the image as the input. The effects of data augmentation and stain normalisation were also investigated. Two datasets were used in this project, with one provided by the Head and Neck 5000 Study, and the other being the Camleyon16 training dataset. The best performance obtained from the whole-slide method was an area under the receiver operating characteristic curve (AUC) of 0.727, with a false positive rate (FPR) of 0.36 and a false negative rate (FNR) of 0.22. The patch-based method gave an AUC of 0.890, with an FPR of 0.21 and FNR of 0.24. This level of performance falls slightly short of the state of the art, and suggestions are provided for why this is and how it can be improved in future.

## Links

* Report (PDF): https://marleysudbury.github.io/final-year-project/report.pdf
* Source code: https://github.com/marleysudbury/detect-diagnose-tumours

## Key people

* Author: Marley Sudbury
* Supervisor: Paul Rosin
* Moderator: Dave Marshall

## Credits and licenses

This project used data from the Head and Neck 5000 study. More information about the study is available here: http://www.headandneck5000.org.uk/

Additional data was used from the Camelyon16 challenge. More information about Camelyon16 can be found here: https://camelyon16.grand-challenge.org/ and the data set can be found here: https://camelyon17.grand-challenge.org/Data/

This project uses the stain normalisation code available here: https://github.com/schaugf/HEnorm_python/blob/master/normalizeStaining.py
