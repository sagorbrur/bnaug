
def replace_special_token_to_empty(text, token_list):
    for token in token_list:
        text = text.replace(token, '')
    
    return text