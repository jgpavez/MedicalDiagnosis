# Deep Medical Diagnosis

QA solving using DL methods applied to medical diagnosis. The main models used are a RNN and a memnet.

## Dataset

We have a dataset of 2350 diseases with their symptoms. The data format is similar to the dataset from 
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

The next plots show 1) the number of facts in each sample and 2) the number of words in each fact:



 Facts x sample                   | Words x fact
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/MedicalDiagnosis/blob/master/plots/facts_by_disease.png" width="350">  | <img src="https://github.com/jgpavez/MedicalDiagnosis/blob/master/plots/word_by_fact.png" width="350" >

## Training a RNN 

A GRU network is trained on the dataset. The GRU consists of two layers of 128 units with a dropout of 0.5. The output is a softmax layer. Pre-trained word vectors are used as embedding layer. 
The network works very well on the training and test dataset but is suboptimal on self-made data. There are various possible ways to solve that which must be studied.

Accuracy curves on the validation set are shown in the next image

![Accuracy](https://github.com/jgpavez/MedicalDiagnosis/blob/master/plots/accuracy.png)

An example of output for data from the dataset is shown next. It is interesting to notice that the network outputs related diseases as the most probable 5, meaning that is understanding that those diseases are related (in this case all are psychological disorders).

```
Predictions for data:
[[u'is', u'easily', u'distract', u'.'], [u'is', u'oftentimes', u'short', u'in', u'daily', u'activities', u'.'], [u'hyperactivity', u'symptoms', u'.'], [u'fidgetinesses', u'with', u'custodies', u'or', u'pess', u'or', u'wriggles', u'in', u'place', u'.'], [u'leaves', u'place', u'when', u'remain', u'sit', u'is', u'expect', u'.'], [u'runs', u'about', u'or', u'rises', u'in', u'inappropriate', u'situations', u'.'], [u'has', u'jobs', u'playing', u'or', u'workings', u'softly', u'.'], [u'is', u'oft', u'on', u'the', u'turn', u'acts', u'as', u'if', u'drive', u'by', u'a', u'motor', u'.'], [u'negotiations', u'excessively', u'.'], [u'impulsivity', u'symptoms', u'.'], [u'blurts', u'out', u'replies', u'before', u'enquiries', u'have', u'been', u'complete', u'.'], [u'has', u'trouble', u'look', u'crook', u'.'], [u'interrupts', u'or', u'irrupts', u'on', u'others', u'butts', u'into', u'conversations', u'or', u'games', u'.']]
Disease: attention deficit hyperactivity disorder
5 most prob. diseases: [u'attention deficit hyperactivity disorder', u'oppositional defiant disorder', u'seasonal affective disorder', u'anorexia nervosa', u'language disorder children']
```

Next, we test the neural network on self-made symptoms, some examples are shown next. While in the first case the network correctly identify the disease, in the second case does not.

```
[u'oily skin', u'painful touch skin', u'face affected almost everywhere', u'chest affected', u'some blackheads', u'a lot of papules', u'papules', u'nodules', u'cysts']
Disease: acne
5 most prob. diseases: [u'acne', u'genital warts', u'scarlet fever', u'erythema multiforme', u'measles']

[u'pulsating feeling in stomach', u'persistent back pain', u'abdominal pain', u'severe pain in the middle abdomen', u'dizziness', u'clammy skin', u'tachycardia', u'loss of consciousness']
Disease: abdominal aortic aneurysm
5 most prob. diseases: [u'pulmonary actinomycosis', u'mucormycosis', u'sleeping sickness', u'acute myeloid leukemia', u'bile duct obstruction']
```


