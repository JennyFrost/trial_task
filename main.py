from PhraseExtractor import PhraseExtractor, mean_pooling

import numpy as np
import re
import torch
import spacy
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, RobertaModel
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('punkt')
nlp = spacy.load("en_core_web_lg")

# READ INPUT TEXT

filename = ''
while not filename:
    filename = input('Enter the path to the file with the text you wish to improve: ')

with open(file=filename, mode='r') as f:
    lines = f.readlines()
input_text = ' '.join([line.strip() for line in lines if not re.match(r'\s+', line)])

# READ STANDARD PHRASES

filename_phrases = '/content/business phrases.txt'
with open(file=filename_phrases, mode='r') as f:
    lines = f.readlines()
standard_phrases = list(set([phrase.strip().lower() for phrase in lines]))

# PROCESS PHRASES AND SENTENCES WITH SPACY

standard_phrases_docs = list(nlp.pipe(standard_phrases))
input_docs = list(nlp.pipe(sent_tokenize(input_text)))

# SORT OUT VERB AND NOUN STANDARD PHRASES

verb_standard_phrases, noun_standard_phrases = [], []
for doc in standard_phrases_docs:
    root = [tok for tok in doc if tok.dep_ == 'ROOT'][0]
    if root.pos_ in ('VERB', 'AUX'):
        verb_standard_phrases.append(doc.text)
    else:
        noun_standard_phrases.append(doc.text)

# EXTRACT VERB AND NOUN PHRASES FORM TEXT

phrase_extractor = PhraseExtractor(input_docs)
verb_phrases_from_text, noun_phrases_from_text = phrase_extractor.get_phrases_from_text()

# GET PHRASE EMBEDDINGS

# The use of sentence transformers, distilroberta model
model = SentenceTransformer('all-distilroberta-v1')

verb_phrases_from_text_embeddings = model.encode(verb_phrases_from_text, convert_to_tensor=True)
noun_phrases_from_text_embeddings = model.encode(noun_phrases_from_text, convert_to_tensor=True)
verb_standard_phrases_embeddings = model.encode(verb_standard_phrases, convert_to_tensor=True)
noun_standard_phrases_embeddings = model.encode(noun_standard_phrases, convert_to_tensor=True)

verb_cosine_scores = util.cos_sim(verb_phrases_from_text_embeddings,
                                  verb_standard_phrases_embeddings).numpy()
noun_cosine_scores = util.cos_sim(noun_phrases_from_text_embeddings,
                                  noun_standard_phrases_embeddings).numpy()

print('The use of sentence-transformers model to get sentence embeddings', end='\n\n')

for i, phrase in enumerate(verb_phrases_from_text):
    j = np.argmax(verb_cosine_scores[i])
    if verb_cosine_scores[i][j] > 0.5:
        print("{:50s} {:30s} Score: {:.4f}".format(phrase, verb_standard_phrases[j],
                                                   verb_cosine_scores[i][j]))

for i, phrase in enumerate(noun_phrases_from_text):
    j = np.argmax(noun_cosine_scores[i])
    if noun_cosine_scores[i][j] > 0.5:
        print("{:50s} {:50s} Score: {:.4f}".format(phrase, noun_standard_phrases[j],
                                                   noun_cosine_scores[i][j]))

# The use of  regular transformers, roberta model

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
model.eval()

verb_phrases_from_text_emb = torch.Tensor().to(device)
verb_standard_phrases_emb = torch.Tensor().to(device)
noun_phrases_from_text_emb = torch.Tensor().to(device)
noun_standard_phrases_emb = torch.Tensor().to(device)

with torch.no_grad():
    for vp in verb_phrases_from_text:
        inputs = tokenizer(vp, return_tensors="pt")
        outputs = model(**inputs)
        verb_phrases_from_text_emb = torch.cat([verb_phrases_from_text_emb,
                                                mean_pooling(outputs, inputs['attention_mask'])])
    verb_phrases_from_text_emb = verb_phrases_from_text_emb.cpu().numpy()
    for vp in verb_standard_phrases:
        inputs = tokenizer(vp, return_tensors="pt")
        outputs = model(**inputs)
        verb_standard_phrases_emb = torch.cat([verb_standard_phrases_emb,
                                               mean_pooling(outputs, inputs['attention_mask'])])
    verb_standard_phrases_emb = verb_standard_phrases_emb.cpu().numpy()
    for np in noun_phrases_from_text:
        inputs = tokenizer(np, return_tensors="pt")
        outputs = model(**inputs)
        noun_phrases_from_text_emb = torch.cat([noun_phrases_from_text_emb,
                                                mean_pooling(outputs, inputs['attention_mask'])])
    noun_phrases_from_text_emb = noun_phrases_from_text_emb.cpu().numpy()
    for np in noun_standard_phrases:
        inputs = tokenizer(np, return_tensors="pt")
        outputs = model(**inputs)
        noun_standard_phrases_emb = torch.cat([noun_standard_phrases_emb,
                                               mean_pooling(outputs, inputs['attention_mask'])])
    noun_standard_phrases_emb = noun_standard_phrases_emb.cpu().numpy()

verb_cosine_scores = cosine_similarity(verb_phrases_from_text_emb, verb_standard_phrases_emb)
noun_cosine_scores = cosine_similarity(noun_phrases_from_text_emb, noun_standard_phrases_emb)

print('The use of averaged word embeddings to get sentence embeddings', end='\n\n')

for i, phrase in enumerate(verb_phrases_from_text):
    j = np.argmax(verb_cosine_scores[i])
    if verb_cosine_scores[i][j] > 0.5:
        print("{:50s} {:30s} Score: {:.4f}".format(phrase, verb_standard_phrases[j],
                                                   verb_cosine_scores[i][j]))

for i, phrase in enumerate(noun_phrases_from_text):
    j = np.argmax(noun_cosine_scores[i])
    if noun_cosine_scores[i][j] > 0.5:
        print("{:50s} {:50s} Score: {:.4f}".format(phrase, noun_standard_phrases[j],
                                                   noun_cosine_scores[i][j]))
