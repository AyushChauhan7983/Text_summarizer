import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

text = "The Avengers, a team of extraordinary superheroes assembled by Marvel Comics, comprises iconic characters like Iron Man, Captain America, Thor, Hulk, Black Widow, and Hawkeye. These superheroes, each with unique abilities and backgrounds, unite to defend the world against formidable threats.Led by the enigmatic Nick Fury and the intelligence agency S.H.I.E.L.D., the Avengers stand as Earth's mightiest protectors. With their unwavering dedication and unparalleled powers, they safeguard humanity from villains, aliens, and cosmic forces that threaten global peace.The team's adventures span across the Marvel Cinematic Universe, showcasing epic battles, alliances, and personal struggles. From facing Loki's devious schemes to confronting the powerful Thanos and his quest for universal dominance, the Avengers epitomize courage, sacrifice, and the spirit of heroism.Their camaraderie, resilience, and unwavering commitment to defending the world have cemented the Avengers as legendary figures in the realm of superheroes, inspiring generations with their valor and the enduring belief that, together, Earth's mightiest heroes can overcome any adversity."


def summarizer(rawdocs):
    stopwords = list(STOP_WORDS)

    nlp = spacy.load('en_core_web_sm')  # smallest spacy english trained model
    doc = nlp(rawdocs)

    # creating token of each word in the text
    tokens = [token.text for token in doc]

    # calculating the frequency of each word
    word_freq = {}
    for word in doc:
        if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
            if word.text not in word_freq.keys():
                word_freq[word.text] = 1
            else:
                word_freq[word.text] += 1

    max_freq = max(word_freq.values())

    # normalised frequency
    for word in word_freq.keys():
        word_freq[word] = word_freq[word]/max_freq

    sent_tokens = [sent for sent in doc.sents]

    sent_scores = {}
    for sent in sent_tokens:
        for word in sent:
            if word.text in word_freq.keys():
                if sent not in sent_scores.keys():
                    sent_scores[sent] = word_freq[word.text]
                else:
                    sent_scores[sent] += word_freq[word.text]

    # this will select the number of lines which are needed to create the summary of text
    select_len = int(len(sent_tokens) * 0.3)

    # this function will take number of sentences which we get output from the above line of code (for eg. 2) with max frequency
    summary = nlargest(select_len, sent_scores, key=sent_scores.get)

    final_summary = [word.text for word in summary]

    summary = ' '.join(final_summary)

    return summary, doc, len(rawdocs.split(' ')), len(summary.split(' '))
