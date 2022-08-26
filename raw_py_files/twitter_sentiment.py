import numpy as np
import re
import pandas as pd
import html
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                            pipeline, TrainingArguments, Trainer)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from datasets import Dataset

def lite_parser(post):
    # remove html chars %xx
    post = html.unescape(post)

    # look at each word
    t_post = post.split()
    
    # make links, mentions, cashtags, and numbers uniform
    t_post = [re.sub('^http.*', '#link',z) for z in t_post]
    t_post = [re.sub('^\@.*', '@mention',z) for z in t_post]
    t_post = [re.sub('^\$.*', '$cashtag', z) for z in t_post]
    t_post = [re.sub('^\d+\.*\d+', '#number', z) for z in t_post]

    # blunt instrument used to remove things that are not in the set of symbols below
    # more can be added here
    t_post = [re.sub("[^a-zA-Z@#$0-9.,!?']", ' ', z) for z in t_post]
  
    #remove blanks
    t_post = ' '.join([z.strip() for z in t_post if len(z.strip()) > 0]).strip().split('\w+')
    return ' '.join(t_post)

tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert', num_labels = 3)
nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

df = pd.read_csv('in/jfr_sample.txt', delimiter='\t')

# use lite_parser on each line of the input file
df['texts'] = df['texts'].apply(lite_parser)

# classified is a json-like data structure with the results of the classification
classified = nlp(df['texts'].to_list())

# map sentiment to jfr_sample label
xmap = {'neutral': 2,'positive': 1,'negative': 0}
y_predict = [xmap[z['label']] for z in classified]
y_true = df['label'].to_list()

# Get model performance
report = classification_report(y_pred=np.array(y_predict),y_true=np.array(y_true))
print(report)

tokenizer.add_tokens(['#link', '@mention', '$cashtag', '#number'])
model.resize_token_embeddings(len(tokenizer))

# check model labels and align with fine tuning data labels (here, we switch tuning positive and negative)
print(model.config.id2label.items())
sent_map = {1:0,0:1,2:2}
df['label'] = df['label'].map(sent_map)

# split data
df_train, df_test, = train_test_split(df, stratify=df['label'], test_size=0.1, random_state=42)
df_train, df_val = train_test_split(df_train, stratify=df_train['label'],test_size=0.1, random_state=42)

# translate data
dataset_train = Dataset.from_pandas(df_train)
dataset_val = Dataset.from_pandas(df_val)
dataset_test = Dataset.from_pandas(df_test)

dataset_train = dataset_train.map(lambda e: tokenizer(e['texts'], truncation=True, padding='max_length', max_length=128), batched=True)
dataset_val = dataset_val.map(lambda e: tokenizer(e['texts'], truncation=True, padding='max_length', max_length=128), batched=True)
dataset_test = dataset_test.map(lambda e: tokenizer(e['texts'], truncation=True, padding='max_length' , max_length=128), batched=True)

dataset_train.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
dataset_val.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
dataset_test.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])

# train/save model
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy' : accuracy_score(predictions, labels)}

args = TrainingArguments(
        output_dir = 'models/',
        evaluation_strategy = 'epoch',
        save_strategy = 'epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
)

trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=args,                  # training arguments, defined above
        train_dataset=dataset_train,         # training dataset
        eval_dataset=dataset_val,            # evaluation dataset
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
)

trainer.train()  

tokenizer = AutoTokenizer.from_pretrained('models/checkpoint-45/')
model = AutoModelForSequenceClassification.from_pretrained('models/checkpoint-45/', num_labels = 3)
nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

df = pd.read_csv('in/jfr_sample.txt', delimiter='\t')
df['texts'] = df['texts'].apply(lite_parser)

x = nlp(df['texts'].to_list())

xmap = {'neutral': 2,'positive': 1,'negative': 0}

y_predict = [xmap[z['label']] for z in x]
y_true = df['label'].to_list()

report = classification_report(y_pred=np.array(y_predict),y_true=np.array(y_true))
print(report)


