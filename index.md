---
layout: initial
title: "Zhanhong Cheng"
---

<div class="profile">
<img class="profile-photo" src="assets/images/zhanhong.cheng-24.jpg" alt="Zhanhong Cheng" />
<div class="profile-bio" markdown="1">

## About me

I am Zhanhong Cheng (程展鸿), a <span class="tooltip" tabindex="0" data-tooltip="百人计划研究员 — a tenure-track position under ZJU's Hundred Talents Program, equivalent to a U.S. Assistant Professor">ZJU 100 Young Professor</span> at the [Institute of Intelligent Transportation Systems](http://iits.zju.edu.cn/main.htm){:target="_blank"}, [Zhejiang University](https://www.zju.edu.cn/english/){:target="_blank"}. I received my Ph.D. (2022) in Civil Engineering from [McGill University](https://www.mcgill.ca/){:target="_blank"}, where I subsequently worked as a Postdoctoral Researcher. Before joining Zhejiang University, I was a Postdoctoral Associate at the [Urban AI Lab](http://urbanailab.com){:target="_blank"}, [University of Florida](https://www.ufl.edu){:target="_blank"}. I hold an M.S. (2018) and a B.Eng. (2016) in Transportation Engineering, both from [Harbin Institute of Technology](http://en.hit.edu.cn/){:target="_blank"}.

<div class="social-links">
<a href="mailto:zhanhong.cheng@zju.edu.cn" aria-label="Email" data-label="Email"><img src="assets/images/envelope-solid.svg" alt="" /></a>
<a href="https://scholar.google.com/citations?user=YhrxIBAAAAAJ&hl=en" target="_blank" rel="noopener noreferrer" aria-label="Google Scholar" data-label="Google Scholar"><img src="assets/images/google-scholar-square.svg" alt="" /></a>
<a href="https://github.com/chengzhanhong" target="_blank" rel="noopener noreferrer" aria-label="GitHub" data-label="GitHub"><img src="assets/images/github-brands-solid.svg" alt="" /></a>
<a href="https://www.linkedin.com/in/zhanhong-cheng/" target="_blank" rel="noopener noreferrer" aria-label="LinkedIn" data-label="LinkedIn"><img src="assets/images/linkedin-brands-solid.svg" alt="" /></a>
<a href="https://person.zju.edu.cn/chengzhanhong" target="_blank" rel="noopener noreferrer" aria-label="ZJU Homepage" data-label="ZJU Homepage"><img src="assets/images/building-columns-solid.svg" alt="" /></a>
<a href="{{ '/assets/files/ZhanhongCV.pdf' | relative_url }}" target="_blank" rel="noopener noreferrer" aria-label="CV (PDF)" data-label="CV (PDF)"><img src="assets/images/file-pdf-solid.svg" alt="" /></a>
</div>

</div>
</div>

{% if site.data.pinned.size > 0 %}
<div class="pinned-box">
{% for item in site.data.pinned %}
<p>{{ item.content | markdownify | remove: '<p>' | remove: '</p>' | strip }}</p>
{% endfor %}
</div>
{% endif %}

## Research Interests
My research investigates how Artificial Intelligence (AI) can help predict and simulate complex transportation systems. Building on my previous work in spatiotemporal data modeling, travel behavior analysis, and public transit operations, my current interests include:

- **Generative Mobility and Policy Analysis**: Generative models of daily activities and mobility trajectories for travel-demand synthesis, missing-data recovery, and transportation-policy scenarios.
- **AI-driven Traffic Simulation**: Learning realistic vehicle-pedestrian and vehicle-vehicle interactions from real-world multimodal data, and developing realistic simulation methods for intersections and other fine-grained traffic units.
- **Forecasting with Large Language Models and Multimodal Data**: Studying how large language models can combine quantitative data and textual evidence to forecast transportation demand, infrastructure performance, and policy outcomes.


<div class="section-head">
<h2>News</h2>
<a class="section-link" href="/news/">View all news →</a>
</div>

{% include news_list.html items=site.data.news limit=5 %}


<div class="section-head">
<h2>Selected publications</h2>
<a class="section-link" href="https://scholar.google.com/citations?user=YhrxIBAAAAAJ&hl=en">View all publications →</a>
</div>
- **Cheng, Z.**, Trepanier, M., & Sun, L. (2022). Real-time forecasting of metro origin-destination matrices with high-order weighted dynamic mode decomposition. Transportation Science, 56(4), 904-918.
  [[Full-text]](https://arxiv.org/abs/2101.00466) [[Code]](https://github.com/mcgill-smart-transport/high-order-weighted-DMD) [[Slides]](https://easychair.org/smart-slide/slide/hws4n#) **(2nd best paper at [CASPT](http://www.caspt.org/) and TransitData 2022🏅)**
- **Cheng, Z.**, Hu, L., Bu, Y., Zhou, Y., & Wang, S. (2026). Graph neural networks for residential location choice: connection to classical logit models. Transportation Research Part B: Methodological, 209, 103464.
  [[Full-text]](https://arxiv.org/abs/2507.21334)
- **Cheng, Z.**, Wang, J., Trépanier, M., & Sun, L. (2025). Abnormal metro passenger demand is predictable from alighting and boarding correlation. Transportation Research Part C: Emerging Technologies, 178, 105239.
  [[Full-text]](https://www.sciencedirect.com/science/article/pii/S0968090X25002438) [[Slides]](assets/files/TransitData-24_ABTransformer.pdf) [[Poster]](assets/files/Poster_2025_TRB_ABTransformer.pdf) [[Code]](https://github.com/chengzhanhong/abnormal_metro_demand_predictable)
- **Cheng, Z.**, Trépanier, M., & Sun, L. (2021). Incorporating travel behavior regularity into passenger flow forecasting. Transportation Research Part C: Emerging Technologies, 128, 103200.
  [[Full-text]](https://arxiv.org/abs/2004.00992v2)
- **Cheng, Z.**, Trépanier, M., & Sun, L. (2021). Probabilistic model for destination inference and travel pattern mining from smart card data. Transportation, 48(4), 2035-2053.
  [[Full-text]](https://www.researchgate.net/publication/342077959_Probabilistic_model_for_destination_inference_and_travel_pattern_mining_from_smart_card_data) [[Code]](https://github.com/mcgill-smart-transport/destination_inference)
- Chen, X., **Cheng, Z.**, Jin, J. G., Trépanier, M., & Sun, L. (2023). Probabilistic forecasting of bus travel time with a Bayesian Gaussian mixture model. Transportation Science, 57(6), 1516-1535.
  [[Full-text]](https://arxiv.org/abs/2206.06915) [[Slides]](assets/files/BayesianGMM_caspt.pdf)
- Chen, X., **Cheng, Z.**, Cai, H., Saunier, N., & Sun, L. (2024). Laplacian convolutional representation for traffic time series imputation. IEEE Transactions on Knowledge and Data Engineering.
  [[Full-text]](https://arxiv.org/abs/2212.01529) [[Slides]](https://xinychen.github.io/slides/LCR24.pdf) [[Code]](https://github.com/xinychen/LCR)
