"""Local web interface for analysing text (standard library only).

    GET  /              the form (optionally pre-filled: /?example=mixed)
    POST /              the form submitted → result page
    POST /api/predict   JSON in ({"text": "..."}), JSON out
    GET  /api/health    model information

Safety measures: binds to 127.0.0.1 only; the Host header must be a local name (blocks DNS-rebinding);
request bodies are size-limited; every value is validated by the same code as the command line;
everything shown is HTML-escaped (including the text itself); the page has no scripts and a strict
Content-Security-Policy.
"""

import html
import json
import logging
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from . import schema as S
from .exceptions import SentimentError, ValidationError
from .predictor import highlight_pieces

log = logging.getLogger("sentimentapp")

MAX_BODY = 40_000
DRAIN_LIMIT = 1_000_000   # a rejected body up to this size is read and discarded before the error is sent
LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}
SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; "
                               "base-uri 'none'; frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer", "Cache-Control": "no-store",
}
CHIP_CLASS = {"Positive": "pos", "Negative": "neg", "Neutral": "neu", "Not sure": "unsure"}

CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9;
--accent:#2a78d6;--border:rgba(11,11,11,.12);--err:#b3261e;--errbg:#fdecea;--warnbg:#fff4e0;--okbg:#e6f4ea;--ok:#1e6b34;--mark:#ffe1a8;--neubg:#ececea}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
--line:#2c2c2a;--accent:#3987e5;--border:rgba(255,255,255,.12);--err:#ff8a80;--errbg:#3d1717;--warnbg:#3a2c12;--okbg:#16301d;--ok:#8fd9a0;
--mark:#5a4313;--neubg:#2c2c2a}}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1060px;margin:0 auto;padding:24px 16px 48px}h1{font-size:24px;margin:0}h2{font-size:17px;margin:0 0 8px}h3{font-size:14px;margin:16px 0 6px}
.sub{color:var(--ink2);font-size:13px}.badge{font-size:12px;border:1px solid var(--border);border-radius:999px;padding:2px 8px;margin-left:8px;color:var(--ink2)}
.layout{display:grid;grid-template-columns:minmax(300px,420px) 1fr;gap:16px;margin-top:20px;align-items:start}
@media (max-width:760px){.layout{grid-template-columns:1fr}}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px}
label{display:block;font-size:13px;color:var(--ink2)}
textarea{width:100%;min-height:150px;margin-top:4px;padding:10px;border:1px solid var(--border);border-radius:8px;background:var(--page);
color:var(--ink);font:inherit;resize:vertical}textarea:focus{outline:2px solid var(--accent);outline-offset:1px}
.err{color:var(--err);font-size:12px;margin-top:2px}.invalid textarea{border-color:var(--err)}
button{width:100%;margin-top:10px;padding:10px;border:0;border-radius:8px;background:var(--accent);color:#fff;font:inherit;font-weight:600;cursor:pointer}
button:focus-visible{outline:2px solid var(--ink);outline-offset:2px}
.examples{margin-top:12px;font-size:13px;color:var(--ink2)}.examples a{color:var(--accent);margin-right:10px;white-space:nowrap}
.hero{font-size:44px;font-weight:600;line-height:1.1;margin:4px 0}.hero small{font-size:16px;color:var(--ink2);font-weight:400}
.chip{display:inline-block;font-size:13px;font-weight:600;border-radius:999px;padding:3px 12px;border:1px solid var(--border)}
.chip.pos{background:var(--okbg);color:var(--ok)}.chip.neg{background:var(--errbg);color:var(--err)}
.chip.neu{background:var(--neubg)}.chip.unsure{background:var(--warnbg)}
.scale{position:relative;height:10px;border-radius:5px;background:var(--line);margin:16px 0 4px}
.scale i{position:absolute;top:-4px;width:6px;height:18px;border-radius:3px;background:var(--accent)}
.scale u{position:absolute;top:-5px;left:50%;width:2px;height:20px;background:var(--muted);text-decoration:none}
.scalelab{display:flex;justify-content:space-between;font-size:12px;color:var(--muted)}
.probs{margin:12px 0}.prob{display:grid;grid-template-columns:70px 1fr 48px;gap:8px;align-items:center;font-size:13px;margin:4px 0}
.prob .track{height:10px;background:var(--line);border-radius:5px;position:relative}.prob .fill{position:absolute;top:0;bottom:0;left:0;border-radius:5px;background:var(--muted)}
.prob.top .fill{background:var(--accent)}.prob .amt{text-align:right;font-variant-numeric:tabular-nums}
.msg{background:var(--page);border:1px solid var(--line);border-radius:8px;padding:10px;margin:10px 0;word-break:break-word;white-space:pre-wrap}
mark{background:var(--mark);color:inherit;border-radius:3px;padding:0 2px;text-decoration:underline;text-decoration-style:wavy;text-underline-offset:3px}
ul.list{margin:4px 0;padding-left:20px;font-size:14px}ul.list li{margin:2px 0}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media (max-width:520px){.cols{grid-template-columns:1fr}}
.part{display:grid;grid-template-columns:1fr auto;gap:8px;align-items:center;border-bottom:1px solid var(--line);padding:6px 0;font-size:14px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:6px 6px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--ink2)}
.num{text-align:right;font-variant-numeric:tabular-nums}.note{background:var(--warnbg);border-radius:8px;padding:8px 10px;margin:8px 0;font-size:13px}
.error{background:var(--errbg);color:var(--err);border-radius:8px;padding:10px;margin-bottom:12px}
footer{margin-top:24px;color:var(--muted);font-size:12px}
"""


def e(value):
    return html.escape(str(value), quote=True)


def form_html(text, errors):
    message = f'<div class="err">{e(errors["text"])}</div>' if "text" in errors else ""
    links = " ".join(f'<a href="/?example={key}">{e(title)}</a>' for key, (title, _) in S.EXAMPLES.items())
    return (f'<form method="post" action="/" novalidate><div class="{"invalid" if "text" in errors else ""}">'
            f'<label for="text">Text to analyse (a review, comment or message)'
            f'<textarea name="text" id="text" maxlength="{S.MAX_TEXT}" required>{e(text)}</textarea></label>{message}</div>'
            f'<button type="submit">Analyse sentiment</button></form>'
            f'<div class="examples">Try an example: {links}</div>')


def marked_text(analysis):
    parts = []
    for chunk, effect in highlight_pieces(analysis.text, analysis.effects):
        parts.append(f'<mark title="multiplies the odds of {e(analysis.label.lower())} by {math.exp(effect):.1f}">{e(chunk)}</mark>'
                     if effect is not None else e(chunk))
    return "".join(parts)


def result_html(a):
    verdict = a.verdict
    note = (f"Not sure: it leans {a.label.lower()} with only {a.confidence * 100:.0f}% confidence, so a person should read it."
            if a.needs_review else f"{a.confidence * 100:.0f}% confident.")
    out = [f'<h2>Result</h2><span class="chip {CHIP_CLASS[verdict]}">{e(verdict)}</span>',
           f'<div class="hero">{a.score:+.0f}<small> sentiment score · {e(a.tone.lower())}</small></div><div class="sub">{e(note)}</div>',
           f'<div class="scale" aria-hidden="true"><u></u><i style="left:calc({(a.score + 100) / 2:.1f}% - 3px)"></i></div>'
           '<div class="scalelab"><span>−100 negative</span><span>0</span><span>positive +100</span></div>']
    top = max(a.probabilities, key=a.probabilities.get)
    bars = "".join(f'<div class="prob{" top" if name == top else ""}"><span>{e(name)}</span><div class="track">'
                   f'<div class="fill" style="width:{p * 100:.0f}%"></div></div><span class="amt">{p * 100:.0f}%</span></div>'
                   for name, p in a.probabilities.items())
    out.append(f'<div class="probs">{bars}</div>')
    for w in a.warnings:
        out.append(f'<div class="note">⚠ {e(w)}</div>')
    out.append(f'<h3>Your text (<mark>marked</mark> words support “{e(a.label.lower())}”)</h3><div class="msg">{marked_text(a)}</div>')
    if a.parts:
        rows = "".join(f'<div class="part"><span>{e(p["text"])}</span><span class="chip {CHIP_CLASS[p["label"]]}">{e(p["label"])}</span></div>'
                       for p in a.parts)
        out.append(f"<h3>Part by part</h3>{rows}<div class='sub'>Mixed texts show their two sides here.</div>")
    if a.signals:
        out.append("<h3>What stands out</h3><ul class='list'>" + "".join(f"<li>{e(s)}</li>" for s in a.signals) + "</ul>")
    if a.supporting or a.opposing:
        def items(rows, supports):
            def factor(r):
                return f"odds ×{r['odds']:.1f}" if supports else f"odds ÷{1 / r['odds']:.1f}"
            return "".join(f"<li>{e(r['word'])} <span class='sub'>({factor(r)})</span></li>" for r in rows) or "<li class='sub'>nothing notable</li>"
        out.append(f"<h3>What moves the result</h3><div class='cols'><div><b>Supports “{e(a.label.lower())}”</b><ul class='list'>{items(a.supporting, True)}</ul></div>"
                   f"<div><b>Argues against</b><ul class='list'>{items(a.opposing, False)}</ul></div></div>"
                   "<div class='sub'>Each piece was taken out in turn and the text re-scored; “odds ×2” means the piece doubles the odds.</div>")
    if a.similar:
        rows = "".join(f'<tr><td>{e(c["text"])}</td><td>{e(c["label"])}</td><td class="num">{c["similarity"]:.2f}</td></tr>' for c in a.similar)
        out.append("<h3>Most similar known reviews</h3><table><thead><tr><th>Review</th><th>Labelled</th>"
                   f"<th class='num'>Similarity</th></tr></thead><tbody>{rows}</tbody></table>")
    return "".join(out)


def page(meta, text="", errors=None, analysis=None, message=None):
    errors = errors or {}
    test = meta["metrics"]["test"]
    if analysis is not None:
        right = result_html(analysis)
    elif message:
        right = f'<div class="error">{e(message)}</div>'
    else:
        right = ('<h2>Paste some text</h2><p class="sub">The app tells you whether it sounds positive, neutral or negative, '
                 'gives a score from −100 to +100, marks the words behind the answer and judges each part of a mixed review. '
                 'When it is not confident it says “Not sure” so a person can decide.</p>'
                 f'<p class="sub">Model: {e(meta["model"])} · on unseen reviews: accuracy {test["accuracy"]:.1f}%, macro-F1 {test["f1_macro"]:.1f}%</p>')
    summary = e(errors["_"]) if "_" in errors else ""
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Sentiment Analysis App</title><style>{CSS}</style></head><body><main>"
            "<h1>Sentiment Analysis App<span class='badge'>demo · synthetic data</span></h1>"
            "<div class='sub'>Reads a review or comment and says how the writer feels, and why.</div>"
            f"<div class='layout'><div class='card'>{('<div class=error>' + summary + '</div>') if summary else ''}"
            f"{form_html(text, errors)}</div><div class='card' aria-live='polite'>{right}</div></div>"
            "<footer>Developed by Sneha S · Trained on generated reviews to learn the machine-learning workflow — "
            "real-world accuracy would be lower. A helper for reading feedback, not a judge of people.</footer></main></body></html>")


def make_server(predictor, host="127.0.0.1", port=8040):
    meta = predictor.meta

    class Handler(BaseHTTPRequestHandler):
        server_version = "SentimentApp/1.0"

        def log_message(self, fmt, *args):   # keep the terminal quiet; real problems go to the log file
            log.info("web: " + fmt, *args)

        # ----- helpers
        def send(self, status, body, content_type="text/html; charset=utf-8"):
            data = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            for key, value in SECURITY_HEADERS.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(data)

        def send_json(self, status, payload):
            self.send(status, json.dumps(payload, ensure_ascii=False), "application/json; charset=utf-8")

        def host_ok(self):
            host = (self.headers.get("Host") or "").strip().lower()
            name = host[:host.index("]") + 1] if host.startswith("[") else host.rsplit(":", 1)[0]
            return name in LOCAL_HOSTS

        def read_body(self):
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                raise ValidationError("Content-Length header is required.") from None
            if length < 0 or length > MAX_BODY:
                self.discard_body(length)
                raise OverflowError
            return self.rfile.read(length)

        def discard_body(self, length):
            """Read and throw away (a bounded amount of) a rejected body before replying.

            If the server answers and closes while the client is still sending, the operating system may
            reset the connection and the client never sees the 413 message. Huge bodies are not drained
            (that would let one request tie the server up); for those the connection is simply closed.
            """
            remaining = min(max(length, 0), DRAIN_LIMIT)
            while remaining > 0:
                chunk = self.rfile.read(min(65536, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
            self.close_connection = True

        def guard(self):
            if not self.host_ok():
                self.send(403, "Forbidden: this server only answers to localhost.", "text/plain; charset=utf-8")
                return False
            return True

        # ----- routes
        def do_GET(self):
            if not self.guard():
                return
            url = urlsplit(self.path)
            if url.path == "/":
                example = parse_qs(url.query).get("example", [""])[0]
                self.send(200, page(meta, text=S.EXAMPLES.get(example, ("", ""))[1]))
            elif url.path == "/api/health":
                test = meta["metrics"]["test"]
                self.send_json(200, {"status": "ok", "model": meta["model"], "trained_at": meta["trained_at"],
                                     "test_accuracy": round(test["accuracy"], 2), "test_macro_f1": round(test["f1_macro"], 2),
                                     "confidence_threshold": round(meta["confidence"]["confidence"], 4), "classes": S.CLASSES})
            else:
                self.send(404, page(meta, message="Page not found."))

        def do_POST(self):
            if not self.guard():
                return
            path = urlsplit(self.path).path
            if path not in ("/", "/api/predict"):
                return self.send(404, page(meta, message="Page not found."))
            api = path == "/api/predict"
            try:
                body = self.read_body()
            except OverflowError:
                return self.send_json(413, {"error": "Request too large."}) if api else \
                    self.send(413, page(meta, message="That request is too large."))
            except ValidationError as exc:
                return self.send_json(411, {"error": str(exc)}) if api else self.send(411, page(meta, message=str(exc)))
            try:
                if api:
                    if "application/json" not in (self.headers.get("Content-Type") or ""):
                        return self.send_json(415, {"error": "Send JSON with Content-Type: application/json."})
                    try:
                        raw = json.loads(body.decode("utf-8"))
                    except (ValueError, UnicodeDecodeError):
                        return self.send_json(400, {"error": "The request body is not valid JSON."})
                    if not isinstance(raw, dict):
                        return self.send_json(400, {"error": 'Send an object like {"text": "your text"}.'})
                    return self.send_json(200, predictor.analyze(raw.get("text")).to_dict())
                form = parse_qs(body.decode("utf-8", "replace"), keep_blank_values=True, max_num_fields=10)
                text = form.get("text", [""])[0]
                try:
                    analysis = predictor.analyze(text)
                except ValidationError as exc:
                    return self.send(422, page(meta, text, exc.errors or {"_": str(exc)}))
                self.send(200, page(meta, text, analysis=analysis))
            except ValidationError as exc:
                self.send_json(422, {"error": str(exc), "fields": exc.errors})
            except SentimentError as exc:
                self.send_json(500, {"error": str(exc)}) if api else self.send(500, page(meta, message=str(exc)))
            except Exception:  # noqa: BLE001 - never leak internals to the browser
                log.exception("Unexpected error handling %s", path)
                self.send_json(500, {"error": "Something went wrong."}) if api else \
                    self.send(500, page(meta, message="Something went wrong. See the log file."))

    return ThreadingHTTPServer((host, port), Handler)


def serve(predictor, port=8040, host="127.0.0.1"):
    server = make_server(predictor, host, port)
    print(f"  Sentiment Analysis App is running at http://{host}:{server.server_address[1]}/  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Web interface stopped.")
    finally:
        server.server_close()
