---
layout: default
title: "Autumn Leaves"
---

<link rel="stylesheet" href="style.css">

# 🍂 Autumn Leaves

## Recent Posts

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
- [Blas style math lib in rust](https://github.com/cyancirrus/stellar-math)
- [Neural net work-in-progress](https://github.com/cyancirrus/neural-net)
- [Fun wordle dynamic programming](https://github.com/cyancirrus/wordle)
