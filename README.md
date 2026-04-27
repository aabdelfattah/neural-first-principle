# Neural Networks from First Principles — publishing kit

Two self-contained, interactive HTML articles that build neural networks from
the ground up. Each file is a single HTML page with inline CSS, inline SVGs,
and inline JavaScript. The only external dependency is Google Fonts via CDN —
drop the file on any static host and it just works.

- `lines-to-neurons.html` — *Neural Networks — From Lines to Neurons*
  Linear regression → loss → gradient descent → backprop → a working neuron.
  Includes a live training playground.

- `neurons-to-transformers.html` — *From Neurons to Transformers — Part II*
  Stacking neurons → matrices → embeddings → attention → the full transformer.

---

## 1. Host on GitHub Pages

Create a fresh public repo for the articles (any clean name — examples below
use `neural-networks-from-first-principles`):

```bash
mkdir neural-networks-from-first-principles
cd neural-networks-from-first-principles
cp /path/to/lines-to-neurons.html .
cp /path/to/neurons-to-transformers.html .
git init && git add . && git commit -m "publish articles"
gh repo create neural-networks-from-first-principles --public --source=. --push
```

On github.com → repo → **Settings → Pages**: Source = *Deploy from a branch*,
Branch = `main`, Folder = `/`. Wait ~1 minute. URLs:

```
https://<username>.github.io/neural-networks-from-first-principles/lines-to-neurons.html
https://<username>.github.io/neural-networks-from-first-principles/neurons-to-transformers.html
```

Smoke-test each in an incognito tab — fonts should load, tabs should switch,
the playground should run training.

---

## 2. Embed in wordpress.com

Paste this into a **Custom HTML** block, one per post:

```html
<iframe
  src="https://<username>.github.io/neural-networks-from-first-principles/lines-to-neurons.html"
  width="100%"
  height="2400"
  style="border:0; max-width:920px; display:block; margin:0 auto;"
  loading="lazy"
  title="Neural Networks — From Lines to Neurons">
</iframe>
```

Tune `height` per article — 2000–2800 px is realistic, since every tab is
rendered (just hidden) so the full document is tall. Cross-origin auto-resize
would need postMessage glue on both sides; not worth it for a personal blog.

### Plan caveat

| wordpress.com plan | Arbitrary `<iframe>` survives Custom HTML sanitizer? |
|---|---|
| Free / Personal | Stripped — only allowlisted embeds (YouTube, Twitter, …) |
| Premium | Often stripped, depends on current allowlist |
| **Business / Commerce ($25+/mo)** | **Yes** — Custom HTML preserved as-written |

If you're below Business, the only reliable fallback is to skip the embed and
add a "**Read the interactive version →**" button in the post linking out to
the GitHub Pages URL.

---

## 3. Trade-offs

- **Fixed iframe height.** No clean cross-origin auto-resize without extra JS.
- **No SEO benefit.** Search engines don't index iframe content as part of the
  host page. If discoverability matters, copy the prose into native WordPress
  blocks and iframe only the playground section.
- **Mobile.** Articles are responsive (max-width ~920 px). Inside an iframe at
  100% width they fit fine — verify on a phone before publishing.

---

What changed vs. the original saved-from-Chrome HTML: the broken local font
link (`./..._files/css2`) was replaced with a real Google Fonts URL covering
every family the page actually uses. Everything else — CSS, SVGs, tab nav JS,
gradient-descent playground — is untouched.
