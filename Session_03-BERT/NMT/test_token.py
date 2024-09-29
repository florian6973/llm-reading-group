from model import load_all_vocab, tokenize

# spacy_de, spacy_en, vocab_src, vocab_tgt = load_all_vocab()
# print(spacy_de, spacy_en, vocab_src, vocab_tgt)

# # spacy tokenizer
# import spacy
# from spacy.lang.fr.examples import sentences 

# nlp = spacy.load("fr_core_news_sm")
# doc = nlp("Il fait beau aujourd'hui.\n")
# print(doc.text)
# for token in doc:
#     print(token.text, token.pos_, token.dep_)


# def build_vocabulary(spacy_de, spacy_en):
#     def tokenize_de(text):
#         return tokenize(text, spacy_de)

#     def tokenize_en(text):
#         return tokenize(text, spacy_en)

#     print("Building German Vocabulary ...")
#     train, val, test = datasets.Multi30k(root='.', language_pair=("de", "en"))
#     vocab_src = build_vocab_from_iterator(
#         yield_tokens(train + val + test, tokenize_de, index=0),
#         min_freq=2,
#         specials=["<s>", "</s>", "<blank>", "<unk>"],
#     )

#     print("Building English Vocabulary ...")
#     train, val, test = datasets.Multi30k(root='.', language_pair=("de", "en"))
#     vocab_tgt = build_vocab_from_iterator(
#         yield_tokens(train + val + test, tokenize_en, index=1),
#         min_freq=2,
#         specials=["<s>", "</s>", "<blank>", "<unk>"],
#     )

#     vocab_src.set_default_index(vocab_src["<unk>"])
#     vocab_tgt.set_default_index(vocab_tgt["<unk>"])

#     return vocab_src, vocab_tgt


# def load_vocab(spacy_de, spacy_en):
#     if not exists("vocab.pt"):
#         vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en)
#         torch.save((vocab_src, vocab_tgt), "vocab.pt")
#     else:
#         vocab_src, vocab_tgt = torch.load("vocab.pt")
#     print("Finished.\nVocabulary sizes:")
#     print(len(vocab_src))
#     print(len(vocab_tgt))
#     return vocab_src, vocab_tgt




def read_file(path):
    with open(path, 'r') as f:
        return f.readlines()
    
file_en = 'datasets/europarl-v7.fr-en.en'
file_fr = 'datasets/europarl-v7.fr-en.fr'
dataset = list(zip(read_file(file_en), read_file(file_fr)))
print(len(dataset))