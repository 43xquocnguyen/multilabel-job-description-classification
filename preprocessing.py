from setup import *

# Check if element is word and '.' or not
def is_word(element):
    if element == '.':
        return True
    pattern = r'^\w+$'
    return re.match(pattern, element) is not None

# Punctuation & stopwords removal
def remove_characters(tokens):
    cleaned_tokens = []
    for token in tokens:
        if token not in STOPWORDS and is_word(unidecode(token)):
                cleaned_tokens.append(token)
    return cleaned_tokens

def preprocess(txt):
    # Lower case
    txt = txt.lower()
    # Tokenization
    txt = vncorenlp.tokenize(txt)
    txt = list(chain.from_iterable(txt))
    # Punctuation & stopwords removal
    txt = remove_characters(txt)
    txt = list(filter(lambda x: x != '', txt))
    # Join
    txt = ' '.join(txt)
    txt = txt.replace('_', ' ')
    txt = txt.replace(' . ', '. ')
    return txt

# def tokenize_data(text_data, tokenizer, max_length):
#     input_ids = []
#     attn_mask = []
#     # token_type_ids = []

#     for txt_data in tqdm(text_data):

#         tokenized = tokenizer.encode_plus(
#             txt_data,
#             padding='max_length',
#             truncation=True,
#             max_length=max_length,
#             add_special_tokens=True
#         )

#         input_ids.append(tokenized['input_ids'])
#         attn_mask.append(tokenized['attention_mask'])
#         # token_type_ids.append(tokenized['token_type_ids'])

#     return torch.tensor(input_ids), torch.tensor(attn_mask)#, torch.tensor(token_type_ids)


# def embed(input_ids, attn_mask, encoder):
#     with torch.no_grad():
#         outputs = encoder(input_ids, attention_mask=attn_mask)
#     return outputs[0]


# def extract_feature_for_DL(X, tokenizer, max_length, encoder):
#     X = [preprocess(x) for x in X]
#     ids, attn_mask = tokenize_data(X, tokenizer, max_length=max_length)
#     embeddeds = embed(ids, attn_mask, encoder=encoder)
#     return embeddeds



# def extract_feature_for_ML(X):
#     X = [preprocess(x) for x in X]
#     with open('model/tfidf_5000_ML.pkl', 'rb') as file:
#         tfidf_loaded = pickle.load(file)
#     return tfidf_loaded.transform(X)
