# Sorghum_Yield_Prediction_2025
This repository contains the materials necessary to test the models and data relevant to a submitted manuscript (*in review*).

Python script execution is simple. Run the script and follow the prompts in the dialog boxes. 

The raw data necessary to train the machine learning models in the manuscript are included, as well as the results for the AFTC test used for cross-validation. These will need to be converted to a CSV format to use them with the Python script.

The "pickle" files are also included so that the model weights used for the breeding portion of the paper can be replicated. To test this, you will just need to use the pickle file when running the script and run only the AFTC data. The pickle files/weights were created by training on the data that were not part of the AFTC test, so that the AFTC test could be used for validation.

Entries have been anonymized because they are being used as part of an ongoing breeding program and associated research effort.

If you find the Python script, the model weights, or the data helpful, please cite our work:
*Citation here*
