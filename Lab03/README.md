# NER Lab for IIITL Class

## Data Sharing

Before we move on, an important disclaimer:
These datasets have not been created by us (the "Teachers" of this class) and we do not claim any copyright on it. However, the original creators of these datasets have released this for the purpose of research. **We are only using it for teaching the importance- and the task- of NER itself.**<br/>

**Please be aware that you are not allowed to re-create, re-share, or re-purpose the data (or parts of it) resulting into any kind of commercial gains. Please also be aware that if you publish anything based on these datasets, you MUST CITE the orginial creators of the dataset.** <br/>

Let us now move on to the dataset, if the disclaimer shared above is clear.
## Link to data

* Please [follow this link](https://drive.google.com/drive/folders/19QbfqC-E-LFcAfnRe_DtP_co9m91p_mC?usp=sharing) to get access to the data we have uploaded in a GDrive folder. You will need your IIITL organizational access via GDrive to access the link.

* Please download the dataset (the one language you choose) and have a look at the files you need to process.

### Dataset Description

* Some data has been acquired from a recent shared task (the files with .zip extension) and can be used as-it-is for the NER task. Ref.: MultiCoNER Shared Task 2022.
* Some data has been acquired from the WikiANN (sometimes known as PAN-X) dataset and have a langauge identifier present with the tokens (the files with the tar.gz extension). You need to make sure that the language identifier is not passed in the model training as you will be training a monolingual NER model, i.e., an NER model for a Single Language.

Here is a description for the files on the folder link shared:

bn.zip -> Bengali NER Data<br/>
de.zip -> German NER Data<br/>
en.zip -> English NER Data<br/>
es.zip -> Spanish NER Data<br/>
fa.zip -> Farsi NER Data<br/>
gu.tar.gz -> Gujarati NER Data<br/>
hi.zip -> Hindi NER Data<br/>
kn.tar.gz -> Kannada NER Data<br/>
ko.zip -> Korean NER Data<br/>
ml.tar.gz -> Malayalam NER Data<br/>
mr.tar.gz -> Marathi NER Data<br/>
ne.tar.gz -> Nepali NER Data<br/>
nl.zip -> Dutch NER Data<br/>
or.tar.gz -> Oriya NER Data<br/>
pa.tar.gz -> Punjabi NER Data<br/>
ru.zip -> Russian NER Data<br/>
ta.tar.gz -> Tamil NER Data<br/>
te.tar.gz -> Telugu NER Data<br/>
tr.zip -> Turkish NER Data<br/>
zh.zip -> Chinese NER Data<br/>

You are required to choose only ONE language as you dataset for the task. Further instructions will follow.