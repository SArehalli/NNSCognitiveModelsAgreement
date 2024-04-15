# RNN AgreementAttraction project code

Here's a quick outline of what files do what:

### download_files.sh

grab all of the necessary lm data for the Gulordava et al. (2018) model. 

### eval_gulordava.py 

Python 3 script. Evaluates an LSTM model trained using the Gulordava et al. (2018) code (the colorlessgreenRNNs submodule) on a set of experimental materials (see the evalsets directory items.csv files for examples of the format)

### eval_rnng.py

Python 3 script. Same as eval_gulordava.py, but for the Dyer et al. 2016 RNNGs. Mostly delegates work to the Stanford parser and the executables in the rnng submodule.

### get_ptb.py

Extracts the RNNG training data from a local installation of the Penn Treebank using nltk. "-s" option reduces the POS tagset to be more manageable (following the preprocessing guidlines from Jason Eisner's Dissertation)

### small-*.ptb

Produced by get_ptb.py. Parse trees that serve as training input for the RNNGs.

### scripts/*_all.sh

scripts to evaluate all of the models of the chosen type on all of the experimental materials (note: hardcoded, won't actually scan directories for new evalsets)
