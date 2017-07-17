# neuralCI
---
In these two notebooks I outline the Word2Vec neural net model, and run it against CI Log Data from Red Hat. 

## Usage:
**If you have the `rchi-moby` parquet:**

Place the rchi parquet into the same directory as the notebooks. Then, run the [dataFormatter](https://github.com/RobGeada/neuralCI/blob/master/dataFormatter.ipynb) notebook followed by the [neuralCI](https://github.com/RobGeada/neuralCI/blob/master/neuralCI.ipynb) notebook.

**If you _don't_ have the `rchi-moby` parquet:**

Since you have no raw data, you can't run the dataFormatter, but I've included sample formatted data that is read by the [neuralCI](https://github.com/RobGeada/neuralCI/blob/master/neuralCI.ipynb) notebook. Therefore, you can simply run neuralCI.

Both of the notebooks are well documented enough that they should be self-explanatory!

## Disclaimer:
For general corporate compliance practices, I haven't committed the Red Hat dataset that the notebooks are configured to run on. However, I have included sample output of the dataFormatter notebook, which allows you to run the neuralCI notebook without having the original data. This sample data is numeric and uninterpretable, and therefore _shouldn't_ be a confidentiality issue. 
