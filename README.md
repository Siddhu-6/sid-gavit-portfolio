# siddhikesh gavit — portfolio

> a portfolio that talks back. built by a final-year CSE student who ships agentic systems, plays state-level football, and takes his coffee black.

🔗 **live at → [siddhikesh-portfolio.vercel.app](https://siddhikesh-portfolio.vercel.app)**

*(replace with your actual URL after deploy)*

---

## what's in it

things you can actually do on the site, not just look at:

- 🤖 **talk to a live AI agent** — trained on 22 years of me. ask professional stuff, past/future stuff, coffee stuff. it won't dodge the awkward questions.
- ⚽ **watch my tech stack play football** — a real 11 v 11 autoplay match on a proper 4-3-3 pitch. team sid (agentic + LLM) vs team recruiter (infra + automation). the ball tracks players in flight so passes actually land. tap the pitch to shoot.
- 🌌 **a constellation you can drag your cursor through** — grid-jittered stars, chill drift speed, connects to your cursor.
- ☕ **a coffee counter** that remembers how many i've had today (via localStorage).
- 📧 **pre-filled email + whatsapp drafts** — recruiters click once, the message writes itself.
- 🕒 **cycling "now" ticker** — what i'm reading, learning, watching. keeps the site feeling alive.
- 📱 **fully responsive** — mobile drops the dense constellation down to a chill sparse sky, subs pills stack under teams, chip layout reflows.

---

## how i made it

**frontend** — zero framework, zero build step. one `index.html` file with inline CSS and JS. everything animates on canvas or SVG. fraunces for the display serif with the SOFT axis dialed in, inter tight for body, jetbrains mono for data/labels. the whole design is a deliberate break from the "dark bg + one neon accent" template look — closer to editorial print than to a dashboard.

**backend** — python + fastapi, one file (`api/chat.py`) that Vercel picks up as a serverless function. the agent's brain is a markdown file (`about.md`) — no vector db, no RAG, just prompt-stuffed context because that's all it needs for a portfolio.

**the agent** — powered by **groq** (free tier, ~500 tokens/sec) running **openai gpt-oss 120b**. `reasoning_effort: "low"` to keep replies snappy and in my voice, not eaten by chain-of-thought. system prompt enforces short lowercase warm replies with a per-IP rate limiter so no one runs up my quota.

**the football match** — this took the most iterations. the tricky bits:
- passes now *track* the target player through flight, so the ball actually lands on the recipient's feet, not where they were 500ms ago
- players make forward/backward/lateral runs based on role + team possession + burst timer
- goal flash color matches the scoring team's color (no more "own goal" visual confusion)
- 22 players + 8 substitutes, all real tools i actually use

**deployment** — vercel free tier. auto-deploys on `git push`.

---

## setup steps for you? nope 🙃

this is *my* portfolio. it's got my face, my numbers, and an AI trained on my life story sitting in a chat box waiting to answer for me. cloning it would be like showing up to an interview in someone else's suit — technically wearable, but everyone can tell.

if you like the aesthetic, you know what to do:
1. open a blank file
2. build your own
3. make it more *you* than mine is *me*

the internet has enough templated portfolios. add one with your own weird ideas instead. go 🚀

---

## contact

sid.gavit6@gmail.com · +91 93592 79778 · [github](https://github.com/Siddhu-6) · [linkedin](https://linkedin.com/in/sid-gavit6)

or just [ask my agent](https://siddhikesh-portfolio.vercel.app#agent).

---

*built by sid · powered by ☕*
