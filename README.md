# Deep Medical Diagnosis

QA solving using DL methods applied to medical diagnosis. The main models used are a RNN and a memnet.

## Dataset

We have a dataset of about 1500 diseases with their symptoms. The data format is similar to the dataset from 
babi task (Facebook). For example, for the disease Achasia:

![Achasia](http://www.diegoacuna.me/factlist.png)

### Exploratory analysis of the dataset

We have 2350 diseases on the dataset. Augmentation has been applied to the original dataset by replacing words by their synonyms and deleting and permuting
facts randomly. The final dataset consists of:

* Vocabulary size: 21520
* Story max length: 1647
* Number of training stories: 133093
* Number of test stories: 59394


Here's how a sample looks like (input, disease):


([[u'adult', u'symptoms', u'of', u'likewise', u'begin', u'to', u'be', u'far', u'more', u'subtle', u'than', u'childhood', u'symptoms', u'.'], [u'being', u'unable', u'to', u'stick', u'at', u'tasks', u'that', u'are', u'borings', u'or', u'clip', u'down', u'.'], [u'but', u'it', u's', u'recognise', u'that', u'symptoms', u'of', u'prevail', u'from', u'childhood', u'into', u'a', u'person', u's', u'teenage', u'and', u'then', u'adulthood', u'.'], [u'carelessnesses', u'and', u'want', u'of', u'to', u'detail', u'.'], [u'continually', u'starting', u'new', u'tasks', u'before', u'finish', u'old', u'ones', u'.'], [u'being', u'unable', u'to', u'postponement', u'their', u'bend', u'.'], [u'some', u'specialists', u'have', u'suggest', u'the', u'followers', u'list', u'of', u'symptoms', u'associated', u'with', u'in', u'adults', u'.'], [u'the', u'symptoms', u'of', u'can', u'be', u'categorize', u'into', u'two', u'types', u'of', u'behavioral', u'jobs', u'.'], [u'interrupt', u'conversations', u'.'], [u'constantly', u'fidget', u'.'], [u'having', u'difficulty', u'coordinate', u'tasks', u'.'], [u'the', u'main', u'signs', u'of', u'and', u'impulsivenesses', u'are', u'.']],[u'attention deficit hyperactivity disorder'])


The next plot show 1) the number of facts in each sample and 2) the number of words in each fact:



 Facts x sample                   | Words x fact
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/MedicalDiagnosis/plots/facts_by_disease.png" width="350">  | <img src="https://github.com/jgpavez/MedicalDiagnosis/plots/word_by_fact.png" width="350" >
