"""The small hand-made sentiment word list shared by the rule-based scorer and the text features.

About 130 general words with scores from -3 (very negative) to +3 (very positive), written separately from the
data generator, like an off-the-shelf lexicon: it knows common words but not every domain-specific phrase.
"""

LEXICON = {
    # strongly positive
    "amazing": 3, "excellent": 3, "fantastic": 3, "wonderful": 3, "outstanding": 3, "superb": 3, "perfect": 3,
    "brilliant": 3, "love": 3, "loved": 3, "awesome": 3, "incredible": 3, "best": 3, "delightful": 3, "great": 2.5,
    # positive
    "good": 2, "nice": 2, "happy": 2, "pleased": 2, "enjoy": 2, "enjoyed": 2, "like": 1.5, "liked": 1.5, "helpful": 2,
    "friendly": 2, "recommend": 2, "recommended": 2, "reliable": 2, "smooth": 2, "fast": 1.5, "easy": 1.5, "fine": 1,
    "decent": 1, "solid": 1.5, "comfortable": 1.5, "pleasant": 2, "worth": 1.5, "impressed": 2, "satisfied": 2,
    "thanks": 1, "thank": 1, "beautiful": 2.5, "clean": 1.5, "cheap": 0.5, "better": 1.5, "works": 1,
    # negative
    "bad": -2, "poor": -2, "slow": -1.5, "disappointing": -2.5, "disappointed": -2.5, "mediocre": -1.5, "bland": -1.5,
    "noisy": -1.5, "overpriced": -2, "confusing": -1.5, "broken": -2.5, "broke": -2, "crash": -2, "crashes": -2,
    "crashed": -2, "late": -1.5, "damaged": -2.5, "rude": -2.5, "dirty": -2.5, "waste": -2.5, "problem": -1.5,
    "problems": -1.5, "issue": -1, "issues": -1, "worse": -2, "boring": -2, "annoying": -2, "unhappy": -2,
    "refund": -1, "complaint": -2, "hate": -3, "hated": -3, "dislike": -2, "disliked": -2, "avoid": -2.5,
    # strongly negative
    "terrible": -3, "awful": -3, "horrible": -3, "worst": -3, "useless": -3, "pathetic": -3, "disgusting": -3,
    "dreadful": -3, "appalling": -3, "unbearable": -3, "atrocious": -3, "garbage": -3, "scam": -3,
}
INTENSIFY = {"very": 1.3, "really": 1.3, "extremely": 1.5, "absolutely": 1.5, "totally": 1.4, "incredibly": 1.5,
             "so": 1.2, "super": 1.3, "utterly": 1.5, "highly": 1.3}
DAMPEN = {"slightly": 0.6, "somewhat": 0.7, "bit": 0.7, "kinda": 0.7, "fairly": 0.8}
CONTRAST = {"but", "however", "although", "though", "yet"}
