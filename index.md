---
layout: initial
title: "Zhanhong Cheng"
---

<div class="profile">
<img class="profile-photo" src="assets/images/zhanhong.cheng-24.jpg" alt="Zhanhong Cheng" />
<div class="profile-bio" markdown="1">

## About me

I am Zhanhong Cheng (程展鸿), a <span class="tooltip" tabindex="0" data-tooltip="百人计划研究员 — a tenure-track position under ZJU's Hundred Talents Program, equivalent to a U.S. Assistant Professor">ZJU 100 Young Professor</span> at the [Institute of Intelligent Transportation Systems](http://iits.zju.edu.cn/main.htm){:target="_blank"}, [Zhejiang University](https://www.zju.edu.cn/english/){:target="_blank"}. I received my Ph.D. (2022) in Civil Engineering from [McGill University](https://www.mcgill.ca/){:target="_blank"}, I continued there as a Postdoctoral Researcher, then joined the [Urban AI Lab](http://urbanailab.com){:target="_blank"} at the [University of Florida](https://www.ufl.edu){:target="_blank"} as a Postdoctoral Associate. I hold an M.S. (2018) and a B.Eng. (2016) in Transportation Engineering, both from [Harbin Institute of Technology](http://en.hit.edu.cn/){:target="_blank"}.

<div class="social-links">
<a href="mailto:zhanhong.cheng@zju.edu.cn" target="_blank" title="Email"><img src="assets/images/envelope-solid.svg" alt="Email" /></a>
<a href="https://scholar.google.com/citations?user=YhrxIBAAAAAJ&hl=en" target="_blank" title="Google Scholar"><img src="assets/images/google-scholar-square.svg" alt="Google Scholar" /></a>
<a href="https://github.com/chengzhanhong" target="_blank" title="GitHub"><img src="assets/images/github-brands-solid.svg" alt="GitHub" /></a>
<a href="https://www.linkedin.com/in/zhanhong-cheng/" target="_blank" title="LinkedIn"><img src="assets/images/linkedin-brands-solid.svg" alt="LinkedIn" /></a>
</div>

</div>
</div>


## Research Interests
My research explores the intersection of Artificial Intelligence (AI) and transportation systems, with extensive work in spatiotemporal data modeling, travel behavior analysis, and public transit operations. My current research interests include, but are not limited to:

- **Spatiotemporal Data**: Integrating AI and statistical learning with multimodal data (e.g., trajectories, traffic flow, text, and networks) for high-fidelity travel demand forecasting and behavior analysis.
- **Generative Mobility**: Leveraging Generative AI to address fundamental challenges in transportation, including data privacy protection, scenario generation, simulation, and system optimization.
- **Emerging Transportation**: Exploring, understanding, and optimizing emerging transportation modes, such as Robotaxi, autonomous transit, and Urban Air Mobility (UAM).



<div class="section-head">
<h2>News</h2>
<a class="section-link" href="/news/">View all news →</a>
</div>

{% if site.data.pinned.size > 0 %}
<div class="pinned-box">
{% for item in site.data.pinned %}
<p>{{ item.content | markdownify | remove: '<p>' | remove: '</p>' | strip }}</p>
{% endfor %}
</div>
{% endif %}

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