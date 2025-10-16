---
layout: default
title: "Autumn Leaves"
---

# 🍂 Autumn Leaves

### Autumn Allmon — Developer | Computational Scientist | Lifelong Learner

*Leafnotes from a developer exploring code, math, and music.*  

{% for post in site.posts %}
  <div class="window">
    <div class="window-header">
      {{ post.title }} <small>({{ post.date | date: "%Y-%m-%d" }})</small>
    </div>
    <div class="window-content">
      {{ post.content }}
    </div>
  </div>
{% endfor %}

## Current Studies
- [Pre-optimized scheduler](https://github.com/cyancirrus/matix)
- [BLAS-style math library in Rust](https://github.com/cyancirrus/stellar-math)
- [Neural network work-in-progress](https://github.com/cyancirrus/neural-net)
- [Fun Wordle dynamic programming](https://github.com/cyancirrus/wordle)

