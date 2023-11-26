    addresses = []
        current_address = []

        for token in ner_result:
            if token['entity'].startswith('B-LOC'):
                if current_address:
                    addresses.append(" ".join(current_address))
                    current_address = [token['word']]
                else:
                    current_address.append(token['word'])
            elif token['entity'].startswith('I-LOC'):
                current_address.append(token['word'])

        if current_address:
            addresses.append(" ".join(current_address))

        return addresses