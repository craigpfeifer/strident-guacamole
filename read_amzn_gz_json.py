__author__ = 'cpfeifer'

import json, sys,gzip
from spacy.en import English

nlp = English()

with gzip.open(sys.argv[1], "rb") as f:
    for a_line in f:
        a_line = a_line.strip().decode("utf8")
        d = json.loads(a_line)["reviewText"].lower()

        s_doc = nlp(d,parse=False, entity=False, tag=False)

        for a_tok in s_doc:
            print ("{} ".format(a_tok.strip()),end="")
        print ("")
