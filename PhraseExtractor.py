import re
import torch
import spacy
from itertools import chain


class PhraseExtractor:

    def __init__(self, docs: list[spacy.tokens.doc.Doc]):
        self.docs = docs

    @staticmethod
    def get_action_verbs(doc: spacy.tokens.doc.Doc) -> list[spacy.tokens.token.Token]:
        """
            For a sentence processed with spaCy, finds all the verbs

            Input
            -----
            doc : Doc
                sentence processed with spaCy
            Output
            ------
            verbs: list[Token]
                list of the verb tokens found (in spaCy format)

        """
        verbs = []
        for token in doc:
            if token.pos_ in ('VERB', 'AUX'):
                if token.dep_ == 'amod' and token.head.dep_ != 'ROOT':
                    continue
                if token.i >= 2:
                    if (re.search(r'ed\b', token.text
                                            and (((doc[token.i-1].text == ','
                                            and doc[token.i-2].pos_ == 'ADJ')
                                            or doc[token.i-1].pos_ == 'ADJ')
                                            or (token.dep_ in ('conj','appos')
                                            and not re.search(r'ed\b', token.head.text))))):
                        continue
                verbs.append(token)
            elif token.pos_ == 'ADJ' and bool(re.search(r'ed\b|ing\b', token.text)) \  # considering the cases when the parser mistakes a verb                                                                           # for
                    and token.head.dep_ in ('ROOT', 'nsubj') \  # for an adjective and the direct verb objected for a noun, like in 'maximized revenue'
                    and not ((doc[token.i-1].text == ',' and  # the parser treats 'maximized' as an adjective and 'revenue' as its parent, while
                              doc[token.i-2].pos_ == 'ADJ') or  # 'maximize' is a verb and the root and 'revenue' is its direct object
                              doc[token.i-1].pos_ == 'ADJ'):
                verbs.append(token)
        return verbs

    @staticmethod
    def get_verb_objects(doc: spacy.tokens.doc.Doc,
                         verb: spacy.tokens.token.Token) -> list[str]:
        """
            For sentence processed with spacy and verb processed with spacy finds direct objects
            and prepositional phrases of the verb

            Input
            -----
            doc : Doc
                sentence processed with spacy
            verb: Doc
                verb processed with spacy
            Output
            ------
            obj_text/pobj_text: list
                a list of strings (objects of the verb) with left children
        """
        if 'dobj' in list(map(lambda x: x.dep_, verb.rights)):
            obj = [tok for tok in verb.rights if tok.dep_ == 'dobj'][0]
            rights = list(chain([x for x in verb.rights if x.i > obj.i], list(obj.rights)))
            objs = [obj]
            obj_text = [' '.join([x.text for x in verb.rights if x.i < obj.i])]
            if obj.conjuncts:  # consider the case when there are several objects
                conjs = [doc[conj.i:conj.i+1] for conj in obj.conjuncts]
                rights += list(chain.from_iterable([list(conj[-1].rights) for conj in conjs]))
                objs += conjs
                obj_text += [conj.text for conj in conjs]
            for i, obj in enumerate(objs):
                lefts = [tok.text for tok in obj.subtree if tok.i <= obj.i]
                if lefts:
                    obj_text[i] += ' '.join(lefts)
            if 'ADP' in list(map(lambda x: x.pos_, rights)):
                prep = list(filter(lambda x: x.pos_ == 'ADP', rights))[0]
                for i, obj in enumerate(objs):
                    obj_text[i] += ' ' + prep.text
                if [tok for tok in prep.rights if tok.dep_ == 'pobj']:
                    pobj = [tok for tok in prep.rights if tok.dep_ == 'pobj'][0]
                    pobj_lefts = [tok.text for tok in pobj.subtree if tok.i <= pobj.i or tok.pos_ == 'ADP']
                    for i, obj in enumerate(objs):
                        obj_text[i] += ' ' + ' '.join(pobj_lefts)
            return obj_text
        if any(list(map(lambda word: word.pos_ == 'ADP', verb.rights))):
            prep = [word for word in verb.rights if word.pos_ == 'ADP'][0]
            if [word for word in prep.rights if word.dep_ == 'pobj']:
                pobj = [word for word in prep.rights if word.dep_ == 'pobj'][0]
                pobjs = [pobj]
                pobj_text = [' '.join([x.text for x in verb.rights if x.i < pobj.i])]
                if pobj.conjuncts:  # consider the case when there are several objects
                    pobjs += list(pobj.conjuncts)
                    pobj_text.append(' '.join([conj.text for conj in pobj.conjuncts]))
                for i, obj in enumerate(pobjs):
                    lefts = [tok.text for tok in pobj.subtree if tok.i <= pobj.i]
                    if lefts:
                        pobj_text[i] += ' ' + ' '.join(lefts)
                    if 'ADP' in list(map(lambda x: x.pos_, pobj.rights)):
                        prep2 = list(filter(lambda x: x.pos_ == 'ADP', pobj.rights))[0]
                        if [tok for tok in prep2.rights if tok.dep_ == 'pobj']:
                            pobj2 = [tok for tok in prep2.rights if tok.dep_ == 'pobj'][0]
                            pobj_lefts2 = [tok.text for tok in pobj2.subtree if tok.i <= pobj2.i or tok.pos_ == 'ADP']
                            if pobj_lefts2:
                                for i, pobj in enumerate(pobjs):
                                    pobj_text[i] += ' ' + prep2.text + ' ' + ' '.join(pobj_lefts2)
                return pobj_text

    @staticmethod
    def get_noun_phrases(doc: spacy.tokens.doc.Doc) -> list[str]:
        """
            For sentence processed with spacy finds noun phrases (noun and its children)

            Input
            -----
            doc : Doc
                sentence processed with spacy
            Output
            ------
            noun_phrases: list[str]
                a list of strings of noun phrases
        """
        noun_phrases = []
        for token in doc:
            if token.pos_ in ('NOUN', 'PROPN'):
                conj = [tok for tok in token.subtree if tok.dep_ == 'conj']
                conj_ind = len(doc)
                if conj:
                    conj_ind = conj[0].i
                    if doc[conj_ind-1].pos_ in ('CCONJ', 'PUNCT'):
                        conj_ind -= 1
                else:
                    if [tok for tok in token.subtree if tok.pos_ == 'PUNCT']:
                        conj_ind = [tok for tok in token.subtree if tok.pos_ == 'PUNCT'][0].i
                noun_phrase = [tok for tok in token.subtree if tok.pos_ != 'PRON' and tok.i < conj_ind]
                if noun_phrase:
                    if noun_phrase[0].pos_ == 'DET':
                        noun_phrase = noun_phrase[1:]
                    if len(noun_phrase) > 1:
                        noun_phrases.append(' '.join(list(map(lambda x: x.text, noun_phrase))))
        return noun_phrases

    def get_phrases_from_text(self) -> tuple[list[str], list[str]]:
        """
            For a list of sentences processed with spacy, makes verb phrases of verb + its object
            and collects all the verb phrases and noun phrases

            Input
            -----
            doc : Doc
                sentence processed with spacy
            Output
            ------
            noun_phrases: list[str]
                a list of strings of noun phrases
            verb_phrases: list[str]
                a list of strings of verb phrases
        """
        verb_phrases_from_text = []
        noun_phrases_from_text = []
        for doc in self.docs:
            verbs = self.get_action_verbs(doc)
            for verb in verbs:
                if self.get_verb_objects(doc, verb):
                    phrase = verb.lemma_ + ' ' + self.get_verb_objects(doc, verb)[0]
                    verb_phrases_from_text.append(phrase)
            noun_phrases = self.get_noun_phrases(doc)
            if noun_phrases:
                noun_phrases_from_text.extend(noun_phrases)
        return verb_phrases_from_text, noun_phrases_from_text


def mean_pooling(model_output, attention_mask):
    """
        Obtains word embeddings from the model and averages them to get a sentence embedding
    """
    token_embeddings = model_output['last_hidden_state']
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask
