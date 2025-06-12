---
layout: default
title: "Autumn Leaves"
---

<link rel="stylesheet" href="style.css">

# 🍂 Autumn Leaves

## Recent Posts

<ul>
  {% for post in site.posts %}
    <li><a href="{{ post.url }}">{{ post.title }}</a> ({{ post.date | date: "%Y-%m-%d" }})</li>
  {% endfor %}
</ul>

## Current Studies
- [Pre-optimized scheduler](https://github.com/cyancirrus/matix)
- [Blas style math lib in rust](https://github.com/cyancirrus/stellar-math)
- [Neural net work-in-progress](https://github.com/cyancirrus/neural-net)
- [Fun worle simulation](https://github.com/cyancirrus/wordle)
