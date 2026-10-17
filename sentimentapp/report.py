"""Model reports: a text summary for the terminal and a self-contained HTML report with charts.

Chart conventions: one validated colour per chart, thin bars with a rounded data end, solid hairline
gridlines, text in text colours (never the series colour), a tooltip on every mark, a "show as table"
view for every chart, and light/dark themes.
"""

import html
import math
from datetime import datetime
from pathlib import Path

from . import schema as S
from .exceptions import ModelError
from .models import BASELINE, LEXICON

WIDTH = 78


# ====================================================================== terminal
def bar(value, top, width=24):
    return "█" * max(1 if value > 0 else 0, round(width * value / top)) if top else ""


def _verdict(meta):
    test, train = meta["metrics"]["test"], meta["metrics"]["train"]
    base, lex = meta["metrics"]["baseline_cv"], meta["metrics"]["lexicon_cv"]
    sel, conf = test["selective"], meta["confidence"]
    weakest = min(test["per_class"], key=lambda c: c["f1"])
    lines = [f"The model labels {test['accuracy']:.1f}% of unseen reviews correctly (macro-F1 {test['f1_macro']:.1f}%). "
             f"Always guessing the most common class would be right {test['majority_accuracy']:.0f}% of the time."]
    lines.append(f"A rule-based word list that does no learning scores {lex['accuracy'] * 100:.0f}% accuracy "
                 f"(macro-F1 {lex['f1_macro'] * 100:.0f}%) in cross-validation, so learning from examples adds "
                 f"{test['f1_macro'] - lex['f1_macro'] * 100:.0f} points of macro-F1.")
    lines.append(f"Serious mistakes, calling a positive review negative or the reverse, are rare: "
                 f"{test['serious_errors']} of {test['n']} ({test['serious_error_rate']:.1f}%). Most errors are between a polar "
                 f"class and Neutral.")
    lines.append(f"The hardest class is {weakest['class']} (F1 {weakest['f1']:.0f}%), the class with the least clear-cut wording.")
    if conf["target_met"]:
        lines.append(f"Predictions with confidence of at least {conf['confidence'] * 100:.0f}% (chosen on training data to be "
                     f"right {conf['target'] * 100:.0f}% of the time) are made automatically for {sel['coverage']:.0f}% of reviews and are "
                     f"right {sel['accuracy_when_sure']:.0f}% of the time; the other {sel['not_sure']} are marked “Not sure” for a person.")
    else:
        lines.append("Even the most confident predictions miss the accuracy target on training data, so every review is "
                     "marked “Not sure”: more or cleaner data is needed.")
    gap = train["accuracy"] - test["accuracy"]
    if gap > 8:
        lines.append(f"Overfitting warning: accuracy is {train['accuracy']:.0f}% on training data but {test['accuracy']:.0f}% on test data.")
    else:
        lines.append(f"No serious overfitting: accuracy is {train['accuracy']:.0f}% on training data and {test['accuracy']:.0f}% on test data.")
    return lines


def model_text(meta):
    test = meta["metrics"]["test"]
    rows = meta["rows"]
    out = ["=" * WIDTH, "  MODEL REPORT",
           f"  {meta['model']}{' (' + meta['model_note'] + ')' if meta.get('model_note') else ''} · trained "
           f"{meta['trained_at'].replace('T', ' ')} in {meta['training_seconds']} s · data: {meta['source']}",
           f"  {rows['raw']} rows read → {rows['clean']} after cleaning → {rows['train']} to train, {rows['test']} to test",
           "=" * WIDTH, "", "  ACCURACY ON THE UNSEEN TEST SET",
           f"   Accuracy        {test['accuracy']:.1f}%     (always the most common class: {test['majority_accuracy']:.1f}%)",
           f"   Macro-F1        {test['f1_macro']:.1f}%     (average F1 of the three classes: the fair score)",
           f"   Balanced acc.   {test['balanced_accuracy']:.1f}%     (average recall per class)",
           f"   Log loss        {test['log_loss']:.3f}      (honesty of the probabilities, lower is better)",
           f"   Serious errors  {test['serious_errors']} of {test['n']} ({test['serious_error_rate']:.1f}%)   (positive ↔ negative flips)", "",
           f"   {'CLASS':<10} {'PRECISION':>9} {'RECALL':>8} {'F1':>7} {'REVIEWS':>8}"]
    for c in test["per_class"]:
        out.append(f"   {c['class']:<10} {c['precision']:>8.1f}% {c['recall']:>7.1f}% {c['f1']:>6.1f}% {c['support']:>8}")
    matrix = meta["confusion"]
    out += ["", "  CONFUSION MATRIX (rows = real sentiment, columns = predicted)",
            f"   {'':<10} {'Negative':>9} {'Neutral':>9} {'Positive':>9}"]
    for i, name in enumerate(S.CLASSES):
        out.append(f"   {name:<10} {matrix[i][0]:>9} {matrix[i][1]:>9} {matrix[i][2]:>9}")
    out += ["", "  → " + "\n  → ".join(_verdict(meta)), "", "  MOST TELLING WORDS (log-odds in training reviews)"]
    for key, name in zip(S.KEYS, S.CLASSES):
        words = ", ".join(w["word"] for w in meta["top_words"][key][:8])
        out.append(f"   {name:<9} {words}")
    out += ["", "  ERRORS BY KIND OF TEXT (test set)"]
    for key in ("length", "negation", "contrast"):
        for s in meta["segments"][key]:
            out.append(f"   {s['group']:<36} {s['errors']:>3} of {s['n']:>4} wrong ({s['error_rate']:.1f}%)")
    return "\n".join(out)


def comparison_text(meta):
    rows = ["  MODEL COMPARISON (5-fold cross-validation on training data; higher macro-F1 is better)",
            f"  {'MODEL':<40} {'MACRO-F1':>9} {'ACCURACY':>9} {'LOG LOSS':>9}", f"  {'-' * 40} {'-' * 9} {'-' * 9} {'-' * 9}"]
    for c in meta["comparison"]:
        mark = "  ← chosen" if c["model"] == meta["metrics"]["cv"]["model"] else ""
        rows.append(f"  {c['model']:<40} {c['f1_macro'] * 100:>8.1f}% {c['accuracy'] * 100:>8.1f}% {c['log_loss']:>9.3f}{mark}")
    return "\n".join(rows)


# ====================================================================== HTML
CSS = """
.rep{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;
--axis:#c3c2b7;--series:#2a78d6;--border:rgba(11,11,11,.10);--other:#c3c2b7}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .rep{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;
--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--series:#3987e5;--border:rgba(255,255,255,.10);--other:#44443f}}
:root[data-theme="dark"] .rep{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--series:#3987e5;--border:rgba(255,255,255,.10);--other:#44443f}
*{box-sizing:border-box}html,body{margin:0}
body.rep{background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 56px}h1{font-size:24px;margin:0}h2{font-size:17px;margin:0 0 4px}
h3{font-size:15px;margin:0 0 2px}.sub{color:var(--ink2);font-size:13px}section{margin-top:28px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-top:16px}
.tile .label{color:var(--ink2);font-size:13px}.tile .value{font-size:26px;font-weight:600;line-height:1.2}
.tile.hero .value{font-size:48px}.tile .note{color:var(--muted);font-size:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:12px;margin-top:12px}
svg.chart{width:100%;height:auto;display:block;overflow:visible}svg.chart text{fill:var(--muted);font-size:13px;font-family:inherit}
svg.chart .label{fill:var(--ink2)}svg.chart .value{fill:var(--ink);font-weight:600}
svg.chart .gridline{stroke:var(--grid)}svg.chart .baseline{stroke:var(--axis)}
svg.chart .line{fill:none;stroke:var(--series);stroke-width:2.5;stroke-linejoin:round;stroke-linecap:round}
.mark{fill:var(--series);outline:none}.mark.other{fill:var(--other)}.mark:hover,.mark:focus-visible{fill-opacity:.75}
.mark:focus-visible{stroke:var(--ink);stroke-width:1.5}circle.dot{fill:var(--series);stroke:var(--surface);stroke-width:2}
ul.notes{margin:8px 0 0;padding-left:20px}ul.notes li{margin:4px 0}
details.tv{margin-top:8px}details.tv summary{cursor:pointer;color:var(--ink2);font-size:13px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--grid)}
th{color:var(--ink2);font-weight:600}.num{text-align:right;font-variant-numeric:tabular-nums}.scroll{overflow-x:auto}
td.msg{max-width:520px;word-break:break-word}td.cell{text-align:right;font-variant-numeric:tabular-nums}
#tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--border);border-radius:8px;
padding:6px 10px;font-size:13px;box-shadow:0 4px 16px rgba(0,0,0,.12);display:none;z-index:10;max-width:280px}#tip b{display:block;font-size:15px}
footer{margin-top:32px;color:var(--muted);font-size:12px}@media (max-width:520px){.tile.hero .value{font-size:40px}.grid{grid-template-columns:1fr}}
"""
TOOLTIP_JS = """(function(){var tip=document.getElementById('tip');
function show(el,x,y){tip.textContent='';var b=document.createElement('b');b.textContent=el.getAttribute('data-value');
var s=document.createElement('span');s.textContent=el.getAttribute('data-label');tip.appendChild(b);tip.appendChild(s);
tip.style.display='block';var w=tip.offsetWidth;tip.style.left=Math.min(x+14,window.innerWidth-w-8)+'px';tip.style.top=(y+14)+'px';}
function hide(){tip.style.display='none';}
document.querySelectorAll('[data-value]').forEach(function(el){
el.addEventListener('pointermove',function(e){show(el,e.clientX,e.clientY);});el.addEventListener('pointerleave',hide);
el.addEventListener('focus',function(){var r=el.getBoundingClientRect();show(el,r.right,r.top);});el.addEventListener('blur',hide);});})();"""


def _e(value):
    return html.escape(str(value), quote=True)


def _tick_ceiling(value, intervals=4):
    """Smallest axis maximum with a 'nice' step (1, 2, 2.5, 5, 10 × a power of ten) that covers `value`."""
    raw = max(value, 1e-9) / intervals
    magnitude = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * magnitude:
            return m * magnitude * intervals
    return 10 * magnitude * intervals


def _end_rounded(x, y, w, h, r=4):
    r = min(r, h / 2, w)
    return (f"M{x:.1f},{y:.1f}H{x + w - r:.1f}Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f}V{y + h - r:.1f}"
            f"Q{x + w:.1f},{y + h:.1f} {x + w - r:.1f},{y + h:.1f}H{x:.1f}Z")


def _top_rounded(x, y, w, h, r=4):
    r = min(r, w / 2, h)
    return (f"M{x:.1f},{y + h:.1f}V{y + r:.1f}Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f}H{x + w - r:.1f}"
            f"Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f}V{y + h:.1f}Z")


def hbar_chart(points, aria, highlight=None, fmt="{:g}"):
    """points [(label, value)]; with `highlight`, that bar is the series colour and the rest recede."""
    w, row, label_w, right = 420, 30, 190, 58
    h = row * len(points) + 8
    vmax = max(v for _, v in points) or 1
    pw = w - label_w - right
    parts = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="{_e(aria)}">',
             f'<line class="baseline" x1="{label_w}" x2="{label_w}" y1="0" y2="{h - 4}"/>']
    for i, (label, value) in enumerate(points):
        y = 4 + i * row
        bw = max(pw * value / vmax, 2)
        cls = "mark" + ("" if highlight in (None, label) else " other")
        shown = fmt.format(value)
        parts.append(f'<text class="label" x="{label_w - 8}" y="{y + 19}" text-anchor="end">{_e(label[:30])}</text>'
                     f'<path class="{cls}" tabindex="0" d="{_end_rounded(label_w, y + 6, bw, 18)}" data-value="{_e(shown)}" '
                     f'data-label="{_e(label)}" aria-label="{_e(label)}: {_e(shown)}"/>'
                     f'<text class="value" x="{label_w + bw + 6:.1f}" y="{y + 19}">{_e(shown)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def column_chart(points, aria, x_title="", unit="", fmt="{:g}", vmax=None):
    """points [(label, value or None)] as thin columns; None draws nothing for that label."""
    w, h, left, right, top, bottom = 420, 236, 44, 6, 20, 52
    pw, ph = w - left - right, h - top - bottom
    values = [v for _, v in points if v is not None]
    vmax = vmax or max(5, _tick_ceiling(max(values, default=0), 5))
    band = pw / max(len(points), 1)
    bar_w = min(30, band - 4)
    parts = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="{_e(aria)}">']
    for k in range(6):
        y = top + ph - ph * k / 5
        parts.append(f'<line class="{"baseline" if k == 0 else "gridline"}" x1="{left}" x2="{w - right}" y1="{y:.1f}" y2="{y:.1f}"/>'
                     f'<text x="{left - 6}" y="{y + 4:.1f}" text-anchor="end">{vmax * k / 5:g}</text>')
    for i, (label, value) in enumerate(points):
        x = left + band * i + (band - bar_w) / 2
        parts.append(f'<text class="label" x="{x + bar_w / 2:.1f}" y="{h - bottom + 18}" text-anchor="middle">{_e(label)}</text>')
        if value:
            bh = ph * value / vmax
            shown = fmt.format(value)
            parts.append(f'<path class="mark" tabindex="0" d="{_top_rounded(x, top + ph - bh, bar_w, bh)}" data-value="{_e(shown)} {_e(unit)}" '
                         f'data-label="{_e(label)}" aria-label="{_e(label)}: {_e(shown)}"/>'
                         f'<text class="value" x="{x + bar_w / 2:.1f}" y="{top + ph - bh - 5:.1f}" text-anchor="middle">{_e(shown)}</text>')
    parts.append(f'<text class="label" x="{left + pw / 2}" y="{h - 6}" text-anchor="middle">{_e(x_title)}</text></svg>')
    return "".join(parts)


def line_chart(points, aria, x_title, y_title, x_range=(0, 1), y_range=(0, 1), marker=None, marker_label=""):
    """A line chart with 0–100% axes; `marker` = (x, y) is highlighted."""
    w, h, left, bottom, top, right = 420, 340, 54, 40, 10, 14
    pw, ph = w - left - right, h - top - bottom
    x_lo, x_hi = x_range
    y_lo, y_hi = y_range
    sx = lambda v: left + pw * (v - x_lo) / ((x_hi - x_lo) or 1)
    sy = lambda v: top + ph - ph * (v - y_lo) / ((y_hi - y_lo) or 1)
    parts = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="{_e(aria)}">']
    for k in range(6):
        yv, xv = y_lo + (y_hi - y_lo) * k / 5, x_lo + (x_hi - x_lo) * k / 5
        parts.append(f'<line class="{"baseline" if k == 0 else "gridline"}" x1="{left}" x2="{w - right}" y1="{sy(yv):.1f}" y2="{sy(yv):.1f}"/>'
                     f'<text x="{left - 6}" y="{sy(yv) + 4:.1f}" text-anchor="end">{yv:.0%}</text>'
                     f'<text x="{sx(xv):.1f}" y="{h - bottom + 16}" text-anchor="middle">{xv:.0%}</text>')
    if len(points) > 1:
        path = " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for i, (x, y) in enumerate(points))
        parts.append(f'<path class="line" d="{path}"/>')
    if marker:
        parts.append(f'<circle class="dot" cx="{sx(marker[0]):.1f}" cy="{sy(marker[1]):.1f}" r="6" tabindex="0" '
                     f'data-value="Chosen threshold" data-label="{_e(marker_label)}" aria-label="{_e(marker_label)}"/>')
    parts.append(f'<text class="label" x="{left + pw / 2}" y="{h - 6}" text-anchor="middle">{_e(x_title)}</text>'
                 f'<text class="label" x="12" y="{top + ph / 2}" text-anchor="middle" transform="rotate(-90 12 {top + ph / 2})">{_e(y_title)}</text></svg>')
    return "".join(parts)


def _table(headers, rows, numeric=(), message_col=None):
    head = "".join(f'<th class="{"num" if i in numeric else ""}">{_e(h)}</th>' for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(
        f'<td class="{"num" if i in numeric else "msg" if i == message_col else ""}">{_e(c)}</td>' for i, c in enumerate(r)) + "</tr>"
        for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _table_view(headers, rows, numeric=(1,)):
    return f'<details class="tv"><summary>Show as table</summary><div class="scroll">{_table(headers, rows, numeric)}</div></details>'


def _tile(label, value, note="", hero=False):
    return (f'<div class="card tile{" hero" if hero else ""}"><div class="label">{_e(label)}</div><div class="value">{_e(value)}</div>'
            + (f'<div class="note">{_e(note)}</div>' if note else "") + "</div>")


def render_html(meta):
    if "coverage_curve" not in meta:
        raise ModelError("This model was saved without test results; train it again to build a report.")
    test, cv, rows = meta["metrics"]["test"], meta["metrics"]["cv"], meta["rows"]
    sel, conf, lex = test["selective"], meta["confidence"], meta["metrics"]["lexicon_cv"]
    body = [f'<header><h1>Sentiment Analysis App — model report</h1><div class="sub">{_e(meta["model"])}'
            f'{" (" + _e(meta["model_note"]) + ")" if meta.get("model_note") else ""} · trained {_e(meta["trained_at"].replace("T", " "))} · '
            f'data: {_e(meta["source"])} · {rows["raw"]} rows read, {rows["clean"]} after cleaning, {rows["train"]} train / {rows["test"]} test</div></header>']
    body.append('<section aria-label="Summary"><div class="tiles">' + "".join([
        _tile("Macro-F1", f"{test['f1_macro']:.1f}%", "average of the three classes", hero=True),
        _tile("Accuracy", f"{test['accuracy']:.1f}%", f"most common class alone: {test['majority_accuracy']:.0f}%"),
        _tile("Rule-based list", f"{lex['f1_macro'] * 100:.0f}%", "macro-F1 without learning (CV)"),
        _tile("Serious errors", f"{test['serious_error_rate']:.1f}%", f"{test['serious_errors']} positive ↔ negative flips"),
        _tile("Labelled automatically", f"{sel['coverage']:.0f}%", f"right {sel['accuracy_when_sure']:.0f}% of the time"),
        _tile("Sent to a person", f"{sel['not_sure']}", f"confidence below {conf['confidence'] * 100:.0f}%")]) + "</div></section>")
    body.append('<section class="card"><h2>What this means</h2><ul class="notes">'
                + "".join(f"<li>{_e(line)}</li>" for line in _verdict(meta)) + "</ul></section>")

    comp = [(c["model"], round(c["f1_macro"] * 100, 1)) for c in meta["comparison"]]
    comparison_rows = [[c["model"], f"{c['f1_macro'] * 100:.1f}%", f"{c['accuracy'] * 100:.1f}%", f"{c['log_loss']:.3f}"] for c in meta["comparison"]]
    chosen = next((c["model"] for c in meta["comparison"] if c["model"] == cv["model"]), None)
    classes = [(c["class"], round(c["f1"], 1)) for c in test["per_class"]]
    class_rows = [[c["class"], f"{c['precision']:.1f}%", f"{c['recall']:.1f}%", f"{c['f1']:.1f}%", c["support"]] for c in test["per_class"]]
    curve = meta["coverage_curve"]
    curve_rows = [[f"{x:.0%}", f"{y:.1%}"] for x, y in curve]
    curve_label = "Chosen threshold: " + f"{sel['coverage']:.0f}%" + " labelled automatically, right " + f"{sel['accuracy_when_sure']:.0f}%" + " of the time"
    bands = meta["confidence_bands"]
    band_points = [(b["band"], None if b["accuracy"] is None else round(b["accuracy"], 1)) for b in bands]
    band_rows = [[b["band"], b["n"], "–" if b["accuracy"] is None else f"{b['accuracy']:.1f}%"] for b in bands]
    word_cards = []
    for key, name in zip(S.KEYS, S.CLASSES):
        points = [(w["word"], w["log_odds"]) for w in meta["top_words"][key]]
        word_cards.append(
            f'<div class="card"><h3>Words that signal {name.lower()}</h3><div class="sub">Log-odds of appearing in {name.lower()} rather than other reviews (training data)</div>'
            f'{hbar_chart(points, "Most " + name.lower() + " words", None, "{:.1f}")}'
            f'{_table_view(["Word", "Log-odds"], [[w, f"{v:.2f}"] for w, v in points])}</div>')
    cards = [
        f'<div class="card"><h3>Model comparison</h3><div class="sub">Cross-validated macro-F1 (%) — higher is better</div>'
        f'{hbar_chart(comp, "Cross-validated macro-F1 of each model", chosen, "{:.1f}")}'
        f'{_table_view(["Model", "Macro-F1", "Accuracy", "Log loss"], comparison_rows, (1, 2, 3))}</div>',
        f'<div class="card"><h3>F1 score for each class</h3><div class="sub">Test set: how well each sentiment is recognised</div>'
        f'{hbar_chart(classes, "F1 score for each class", None, "{:.1f}")}'
        f'{_table_view(["Class", "Precision", "Recall", "F1", "Reviews"], class_rows, (1, 2, 3, 4))}</div>',
        f'<div class="card"><h3>Leaving the doubtful ones to a person</h3><div class="sub">Accuracy of the reviews labelled automatically, as the confidence threshold rises</div>'
        f'{line_chart(curve, "Accuracy against share labelled automatically", "Reviews labelled automatically", "Accuracy of those", x_range=(min(x for x, _ in curve), 1), y_range=(0.5, 1), marker=(sel["coverage"] / 100, sel["accuracy_when_sure"] / 100), marker_label=curve_label)}'
        f'{_table_view(["Labelled automatically", "Accuracy of those"], curve_rows, (0, 1))}</div>',
        f'<div class="card"><h3>Is the model right more often when it is sure?</h3><div class="sub">Accuracy (%) by confidence band, test set</div>'
        f'{column_chart(band_points, "Accuracy by confidence band", "Confidence of the prediction", "% correct", "{:.0f}", vmax=100)}'
        f'{_table_view(["Confidence", "Reviews", "Accuracy"], band_rows, (1, 2))}</div>',
    ] + word_cards
    body.append(f'<section><h2>Charts</h2><div class="grid">{"".join(cards)}</div></section>')

    matrix = meta["confusion"]
    matrix_table = _table(["Real ↓ / predicted →"] + S.CLASSES, [[S.CLASSES[i]] + matrix[i] for i in range(3)], (1, 2, 3))
    seg_head = ["Reviews", "Wrong", "Error rate"]

    def seg(key):
        return [[s["group"], s["n"], s["errors"], f"{s['error_rate']:.1f}%"] for s in meta["segments"][key]]
    length_table = _table(["Length"] + seg_head, seg("length"), (1, 2, 3))
    negation_table = _table(["Negation"] + seg_head, seg("negation"), (1, 2, 3))
    contrast_table = _table(["Contrast"] + seg_head, seg("contrast"), (1, 2, 3))
    body.append('<section><h2>Decisions and weak spots</h2><div class="grid">'
                f'<div class="card"><h3>Confusion matrix</h3><div class="sub">Rows are the real sentiment, columns what the model said</div>{matrix_table}</div>'
                f'<div class="card"><h3>By length</h3>{length_table}</div>'
                f'<div class="card"><h3>By negation</h3>{negation_table}</div>'
                f'<div class="card"><h3>By “but / however”</h3>{contrast_table}'
                '<p class="sub">Mixed reviews (“great food but slow service”) are the hardest: the label depends on which half matters more.</p></div></div></section>')

    error_rows = [[e["text"], e["actual"], e["predicted"], f"{e['confidence'] * 100:.0f}%"] for e in meta["errors"]] or [["None", "", "", ""]]
    body.append('<section class="card"><h2>The actual mistakes</h2><div class="sub">Most confident mistakes first. Check whether the label itself is right: '
                'human labels always contain some disagreement.</div>'
                f'<div class="scroll">{_table(["Review", "Real label", "Model said", "Confidence"], error_rows, (3,), message_col=0)}</div></section>')

    steps = meta["cleaning"]["steps"]
    body.append('<section class="card"><h2>Data cleaning log</h2><div class="sub">'
                f'{meta["cleaning"]["rows_in"]} rows read → {meta["cleaning"]["rows_out"]} kept</div>'
                + _table(["Problem found", "Rows", "Action"], [[s["label"], s["count"], s["outcome"]] for s in steps], (1,))
                + "</section>")
    params = ", ".join(f"{k} = {v}" for k, v in meta.get("model_params", {}).items()) or "defaults"
    env = meta["environment"]
    body.append(f'<footer>Generated {datetime.now():%d %b %Y, %H:%M} · settings: {_e(params)} · seed {meta["seed"]} · '
                f'Python {_e(env["python"])}, scikit-learn {_e(env["sklearn"])} · developed by Sneha S · '
                'Trained on synthetic reviews: for learning only, real-world accuracy would be lower.</footer>')
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Sentiment analysis model report</title><style>{CSS}</style></head><body class='rep'><main>" + "\n".join(body)
            + f"</main><div id='tip' role='status' aria-live='polite'></div><script>{TOOLTIP_JS}</script></body></html>")


def save_html(meta, path):
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_html(meta), encoding="utf-8")
    except OSError as exc:
        raise ModelError(f"Could not write {path}: {exc.strerror or exc}") from exc
    return path
