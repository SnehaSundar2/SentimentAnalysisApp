"""Synthetic reviews for the project.

Real review collections are licensed and carry real people's words, so the project ships a generator that
builds product, hotel, restaurant, app, film and service reviews from many combinable pieces. Each clause has a
KNOWN polarity (+2 strong praise ... -2 strong criticism), and a review's label follows from its clauses:

* positive and negative clauses add up; a clause after "but" counts one and a half times, and a tie is Neutral,
* negation flips a clause ("not good" is negative, "not bad" is positive, "can't complain" is positive),
* sarcasm uses positive words for a negative point ("Oh great, it crashed again"),
* emoji, exclamation marks, stretched words ("sooo good"), SHOUTING and typos decorate the text,
* a few percent of labels are shifted the way human annotators disagree (mostly to the neighbouring class).

The data is clearly SYNTHETIC: it teaches the workflow, and real reviews are messier, so real accuracy would be lower.
The generated file is also deliberately DIRTY so the preprocessing stage has real work to do.
"""

import numpy as np
import pandas as pd

from . import schema as S

LABEL_NOISE = 0.03           # share of labels shifted by "annotator disagreement" (most to the neighbouring class)
POLAR_FLIP_SHARE = 0.15      # of those, the share that jump straight from positive to negative or back

DOMAINS = {
    "restaurant": ["food", "service", "staff", "ambience", "portions", "prices", "pizza", "coffee", "dessert", "menu"],
    "hotel": ["room", "bed", "staff", "location", "breakfast", "cleanliness", "wifi", "view", "check-in", "bathroom"],
    "product": ["battery", "screen", "build quality", "delivery", "camera", "packaging", "sound", "price", "fit", "charger"],
    "app": ["interface", "update", "support team", "speed", "search", "notifications", "login", "design", "sync", "pricing"],
    "film": ["acting", "plot", "ending", "soundtrack", "pacing", "script", "visuals", "cast", "humour", "direction"],
    "service": ["support", "technician", "response time", "billing", "installation", "website", "refund process", "agent",
                "booking", "delivery"],
}
POS_STRONG = ["amazing", "fantastic", "excellent", "outstanding", "superb", "brilliant", "perfect", "wonderful",
              "incredible", "delightful", "gorgeous", "flawless", "great", "awesome", "lovely", "fabulous"]
POS_MILD = ["good", "nice", "decent", "pleasant", "solid", "fine", "reliable", "smooth", "helpful", "comfortable",
            "tasty", "friendly"]
NEG_STRONG = ["terrible", "awful", "horrible", "atrocious", "dreadful", "appalling", "unbearable", "useless", "pathetic",
              "disgusting", "worthless", "shocking", "rubbish", "garbage"]
NEG_MILD = ["poor", "mediocre", "disappointing", "slow", "bland", "clunky", "noisy", "overpriced", "confusing", "flimsy",
            "bad", "sloppy", "lacking", "underwhelming", "dull"]
NEUTRAL_ADJ = ["average", "okay", "standard", "as described", "ordinary", "typical", "acceptable", "adequate", "so-so",
               "normal", "unremarkable"]
INTENSIFIERS = ["really", "very", "extremely", "absolutely", "totally", "incredibly"]
POS_VERBS_STRONG = ["loved", "adored", "was blown away by"]
POS_VERBS_MILD = ["liked", "enjoyed", "appreciated", "was happy with"]
NEG_VERBS_STRONG = ["hated", "despised", "cannot stand"]
NEG_VERBS_MILD = ["disliked", "was let down by", "was unhappy with", "was disappointed by"]
NEUTRAL_FACTS = ["I bought it on {day}", "It arrived after {n} days", "I have been using it for {n} weeks",
                 "We stayed for {n} nights", "I tried it on {day}", "A friend suggested it to me", "It comes in {n} colours",
                 "I read the description before ordering", "I visited on {day} with my family", "It was delivered to my office",
                 "I signed up last {day}", "The box contained the usual items", "I ordered the {colour} one on {day}",
                 "It showed up {n} days later", "I picked the {colour} version", "We came on {day} evening",
                 "I have owned it since {day}", "It was my first time ordering from here", "I asked for the {colour} one",
                 "It is the {colour} model", "I used it every day for {n} weeks", "The order number was in the email",
                 "I took it on a trip last {day}", "My brother had the same one"]
NEUTRAL_GENERIC = ["It is what it is", "Pretty standard stuff", "It was {neutral}", "Nothing unusual to report", "It does what it says",
                   "Everything was {neutral}", "Overall it was {neutral}", "I think it's {neutral}"]
GENERIC = ["It was {adj}", "Everything was {adj}", "Honestly {adj}", "I think it's {adj}", "The whole thing was {adj}",
           "Overall it was {adj}"]
COLOURS = ["blue", "black", "red", "green", "white", "grey", "silver"]
NUMBER_WORDS = ["two", "three", "four", "five", "six", "a few", "2", "3", "5", "7"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
SARCASM = ["Oh great, the {a} broke again", "Wow, thanks for the {a}, truly the highlight of my week",
           "Just what I needed, another problem with the {a}", "Fantastic, the {a} stopped working after {n} days",
           "Brilliant, the {a} failed right on time", "Love how the {a} never works when I need it"]
CLOSERS_POS = ["Would recommend", "Will order again", "Five stars", "Worth every rupee", "Highly recommended"]
CLOSERS_NEG = ["Would not recommend", "Never buying again", "Avoid", "Complete waste of money", "Do not bother"]
CLOSERS_NEUTRAL = ["Time will tell how it lasts", "Let's see how it goes", "Nothing more to add", "Will update later"]
POS_EMOJI = ["😊", "😍", "👍", "🎉", "❤️"]
NEG_EMOJI = ["😡", "👎", "😠", "😞", "😒"]


def pick(rng, items):
    return items[int(rng.integers(len(items)))]


def clause(rng, polarity, aspects):
    """One clause with the requested polarity (-2, -1, 0, +1, +2). Returns (text, polarity)."""
    a, n, day = pick(rng, aspects), pick(rng, NUMBER_WORDS), pick(rng, DAYS)
    kind = rng.random()
    if polarity == 0:
        if kind < 0.45:
            return pick(rng, NEUTRAL_FACTS).format(day=day, n=n, a=a, colour=pick(rng, COLOURS)), 0
        if kind < 0.65:
            return pick(rng, NEUTRAL_GENERIC).format(neutral=pick(rng, NEUTRAL_ADJ)), 0
        return f"The {a} was {pick(rng, NEUTRAL_ADJ)}", 0
    strong = abs(polarity) == 2
    pos = polarity > 0
    adjective = pick(rng, (POS_STRONG if strong else POS_MILD) if pos else (NEG_STRONG if strong else NEG_MILD))
    if kind < 0.34:
        return f"The {a} was {adjective}", polarity
    if kind < 0.52:
        verb = pick(rng, (POS_VERBS_STRONG if strong else POS_VERBS_MILD) if pos else (NEG_VERBS_STRONG if strong else NEG_VERBS_MILD))
        return f"I {verb} the {a}", polarity
    if kind < 0.62:
        return f"The {a} is {pick(rng, INTENSIFIERS)} {pick(rng, POS_MILD if pos else NEG_MILD)}", 2 if pos else -2
    if kind < 0.58:
        return f"{adjective.capitalize()} {a}", polarity
    if kind < 0.72:      # no aspect named: "It was great", "Honestly terrible"
        return pick(rng, GENERIC).format(adj=adjective), polarity
    if kind < 0.86:      # negation: flips the polarity of the adjective it governs
        if pos:          # praise through a negated criticism
            return pick(rng, [f"The {a} was not {pick(rng, NEG_MILD)}", f"I can't complain about the {a}",
                              f"Never had a problem with the {a}", f"The {a} wasn't {pick(rng, NEG_STRONG)}",
                              f"The {a} wasn't {pick(rng, NEG_MILD)} at all", "I can't complain at all",
                              f"It was not {pick(rng, NEG_STRONG)}"]), 1
        return pick(rng, [f"The {a} was not {pick(rng, POS_MILD)}", f"I wasn't impressed with the {a}",
                          f"The {a} is not {pick(rng, POS_STRONG)} at all", f"I would not call the {a} {pick(rng, POS_MILD)}",
                          f"The {a} was not {pick(rng, POS_MILD)} at all", f"It was not {pick(rng, POS_MILD)}",
                          "I am not impressed"]), -1
    if kind < 0.92 and not pos:      # sarcasm: positive words, negative point
        return pick(rng, SARCASM).format(a=a, n=n), -2
    return f"The {a} was {adjective}", polarity


def make_review(rng):
    """Return (text, label_class)."""
    aspects = DOMAINS[pick(rng, sorted(DOMAINS))]
    target = rng.choice([S.POSITIVE, S.NEGATIVE, S.NEUTRAL], p=[0.31, 0.27, 0.42])
    mix = {S.POSITIVE: [0.70, 0.20, 0.10], S.NEGATIVE: [0.10, 0.20, 0.70], S.NEUTRAL: [0.25, 0.50, 0.25]}[target]
    count = int(rng.choice([1, 2, 3], p=[0.35, 0.45, 0.20]))
    clauses = []
    for i in range(count):
        side = rng.choice([1, 0, -1], p=mix)                         # praise, fact or criticism
        level = int(side * (2 if (side and rng.random() < 0.45) else 1))
        clauses.append(clause(rng, level, aspects))
    if rng.random() < 0.30:                                           # a closing line
        side = rng.choice([1, 0, -1], p=mix)
        text, pol = (pick(rng, CLOSERS_POS), 1) if side > 0 else (pick(rng, CLOSERS_NEG), -2) if side < 0 else (pick(rng, CLOSERS_NEUTRAL), 0)
        clauses.append((text, pol))
    parts, total = [], 0.0

    def attach(separator, text):
        """Lower-case the first letter when the clause continues a sentence ('..., but the food was great')."""
        continues = not separator.endswith((". ", "! "))
        if continues and not text.startswith("I ") and not text[:1].isdigit():
            text = text[0].lower() + text[1:]
        return separator + text

    for i, (text, pol) in enumerate(clauses):
        if i == 0:
            parts.append(text)
            total += pol
            continue
        previous = clauses[i - 1][1]
        if previous and pol and previous * pol < 0:                  # opposite feelings: "but" gives the second more weight
            parts.append(attach(pick(rng, [", but ", ", but ", ". However, ", ", although "]), text))
            total += 1.5 * pol
        else:
            parts.append(attach(pick(rng, [". ", ". ", " and ", "! "]), text))
            total += pol
    review = "".join(parts)
    label = S.POSITIVE if total > 0.75 else S.NEGATIVE if total < -0.75 else S.NEUTRAL
    return review, label


def decorate(rng, text, label):
    """Style that real writers add: emoji, exclamations, stretched words, shouting, typos, lower case."""
    if rng.random() < 0.12:
        emoji = pick(rng, POS_EMOJI) if label == S.POSITIVE else pick(rng, NEG_EMOJI) if label == S.NEGATIVE else ""
        text = f"{text} {emoji}".strip()
    if rng.random() < 0.06:
        words = text.split(" ")
        i = int(rng.integers(len(words)))
        w = words[i]
        vowels = [j for j, ch in enumerate(w) if ch in "aeiou"]
        if vowels:
            j = vowels[-1]
            words[i] = w[:j] + w[j] * int(rng.integers(3, 6)) + w[j + 1:]
        text = " ".join(words)
    if rng.random() < 0.06 and label != S.NEUTRAL:
        words = text.split(" ")
        i = int(rng.integers(len(words)))
        if len(words[i]) >= 4 and words[i].isalpha():
            words[i] = words[i].upper()
        text = " ".join(words)
    if rng.random() < 0.08:
        words = text.split(" ")
        i = int(rng.integers(len(words)))
        w = words[i]
        if len(w) > 4 and w.isalpha():
            j = int(rng.integers(len(w) - 1))
            words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
        text = " ".join(words)
    if rng.random() < 0.20 and label != S.NEUTRAL:
        text = text.rstrip(".") + "!" * int(rng.integers(1, 4))
    if rng.random() < 0.15:
        text = text.lower()
    return text


def generate(rows=3000, seed=2026, dirty=True):
    """Return a DataFrame with columns text and label ('Positive' / 'Neutral' / 'Negative'). `dirty=False` gives tidy data."""
    rng = np.random.default_rng(seed)
    texts, labels = [], []
    for _ in range(rows):
        text, label = make_review(rng)
        text = decorate(rng, text, label)
        if rng.random() < LABEL_NOISE:                                # annotators disagree
            if rng.random() < POLAR_FLIP_SHARE and label != S.NEUTRAL:
                label = S.NEGATIVE if label == S.POSITIVE else S.POSITIVE
            elif label == S.NEUTRAL:
                label = int(rng.choice([S.POSITIVE, S.NEGATIVE]))
            else:
                label = S.NEUTRAL
        texts.append(text)
        labels.append(S.CLASSES[label])
    df = pd.DataFrame({"text": texts, "label": labels})
    return make_dirty(df, rng) if dirty else df


def make_dirty(df, rng):
    """Inject realistic data-quality problems (about 10% of rows are affected)."""
    df = df.copy().astype({"label": object, "text": object})
    n = len(df)

    def rows(count):
        return rng.choice(n, count, replace=False)

    spellings = {"Positive": ["positive", "POS", "pos ", "+1", "1", "good"], "Neutral": ["neutral", "NEU", "0", "neutral "],
                 "Negative": ["negative", "NEG", "neg ", "-1", "bad", "Negative "]}
    for idx in np.flatnonzero(rng.random(n) < 0.15):                                   # label spellings
        df.loc[idx, "label"] = pick(rng, spellings[df.loc[idx, "label"]])
    for idx in rows(5):
        df.loc[idx, "label"] = np.nan                                                  # missing label
    for idx in rows(5):
        df.loc[idx, "label"] = pick(rng, ["mixed", "sarcastic", "unsure"])             # unrecognised label
    for idx in rows(8):
        df.loc[idx, "text"] = np.nan                                                   # missing text
    for idx in rows(6):
        df.loc[idx, "text"] = pick(rng, ["", "   ", "\n\t"])                           # blank text
    for idx in rows(4):
        df.loc[idx, "text"] = pick(rng, [".", "?", "!", "-"])                          # too short to judge
    for idx in np.flatnonzero(rng.random(n) < 0.03):                                   # HTML left over from web scraping
        text = df.loc[idx, "text"]
        if isinstance(text, str):
            df.loc[idx, "text"] = pick(rng, ["<p>{}</p>", "{}<br />", "<b>{}</b>", "{} &amp; more"]).format(text)
    for idx in np.flatnonzero(rng.random(n) < 0.03):                                   # stray whitespace / control characters
        text = df.loc[idx, "text"]
        if isinstance(text, str):
            df.loc[idx, "text"] = pick(rng, ["  {} ", "{}\n\n", "\t{}", "{}\x07", "  {}\r\n"]).format(text.replace(" ", "  ", 1))
    for idx in rows(3):                                                                # very long
        text = df.loc[idx, "text"]
        if isinstance(text, str):
            df.loc[idx, "text"] = (text + " ") * 150
    dupes = df.sample(int(n * 0.015), random_state=1)                                  # repeated reviews
    cased = df.sample(int(n * 0.01), random_state=2).copy()                            # same text, different capitals
    cased["text"] = cased["text"].map(lambda t: t.upper() if isinstance(t, str) else t)
    conflict = df.sample(4, random_state=4).copy()                                     # same text, a different label
    conflict["label"] = conflict["label"].map(lambda v: "Negative" if S.canonical_label(v) == S.POSITIVE else "Positive")
    return pd.concat([df, dupes, cased, conflict], ignore_index=True).sample(frac=1, random_state=3).reset_index(drop=True)
