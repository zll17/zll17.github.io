---
layout: post
title: "DDPM, Explained as Progressive Corruption and Reconstruction"
date: 2024-06-25 11:12:56
tags:
  - DDPM
  - Diffusion Models
  - Generative Models
  - DIY
mathjax: true
_styles: |
  #markdown-content figure img {
    height: auto !important;
    margin-left: auto;
    margin-right: auto;
    display: block;
  }
---


When people talk about generative models, VAE and GAN are usually the first names that come up. There are also some less mainstream but still influential alternatives, such as flow-based models and VQ-VAE. In particular, VQ-VAE and its variant VQ-GAN have recently become popular as **image tokenizers**, making it possible to apply a wide range of pretrained NLP methods directly to images.

Beyond these, there is another option that used to be even more niche: **diffusion models**. But diffusion models have now emerged as one of the most important directions in generative modeling. The two most advanced text-to-image systems of their time—OpenAI’s [DALL·E 2](https://arxiv.org/abs/2204.06125) and Google’s [Imagen](https://arxiv.org/abs/2205.11487)—are both built on diffusion models.

## A New Starting Point

Most articles introducing diffusion models begin with energy-based models, score matching, and Langevin dynamics. In the classical story, one first trains an energy model using techniques such as score matching, and then generates samples from it via Langevin dynamics.

From a theoretical perspective, this is a well-developed framework. In principle, it can model and sample any continuous object, such as speech or images. In practice, however, training a good energy function is difficult—especially in high-dimensional settings like high-resolution image generation. On top of that, sampling from an energy model through Langevin dynamics is itself unstable, and often produces noisy results. As a result, for a long time, diffusion-style methods in this traditional sense were mostly limited to relatively low-resolution images.

The current wave of excitement around generative diffusion models began with [DDPM](https://arxiv.org/abs/2006.11239) (Denoising Diffusion Probabilistic Models), introduced in 2020. Although DDPM uses the term “diffusion model,” it is actually quite different from the classical diffusion framework based on energy models and Langevin sampling. Apart from some superficial similarity in the sampling process, the two are fundamentally different. DDPM marks a genuinely new starting point.

In fact, “progressive model” might be a more accurate name than “diffusion model.” The latter can be misleading, because concepts like energy functions, score matching, and Langevin equations are not really central to DDPM or its later variants. Interestingly, the mathematical framework behind DDPM had already appeared in the ICML 2015 paper [*Deep Unsupervised Learning using Nonequilibrium Thermodynamics*](https://arxiv.org/abs/1503.03585). What DDPM achieved was not inventing the entire framework from scratch, but making it work convincingly for high-resolution image generation—and that was what sparked the field’s explosive growth.

This is often how machine learning advances: a model may exist in theory for years before the right timing, engineering, and applications finally bring it to the forefront.

## From Random Noise to Data

Like GANs, a generative model ultimately aims to transform random noise $$\boldsymbol{z}$$ into a data sample $$\boldsymbol{x}$$:

$$
\begin{equation}
\text{random noise }\boldsymbol{z}
\quad \longrightarrow \quad
\text{sample }\boldsymbol{x}
\end{equation}
$$

Directly learning such a transformation is hard. Generating realistic data in one shot from unstructured noise is a highly nontrivial task.

So instead of trying to learn the whole mapping at once, suppose we first define an easier **forward process**: gradually corrupt a clean sample $$\boldsymbol{x}_0$$ into pure noise $$\boldsymbol{x}_T$$. Let this process take $$T$$ steps:

$$
\begin{equation}
\boldsymbol{x}=\boldsymbol{x}_0 \to \boldsymbol{x}_1 \to \boldsymbol{x}_2 \to \cdots \to \boldsymbol{x}_{T-1} \to \boldsymbol{x}_T=\boldsymbol{z}
\end{equation}
$$

The key idea is simple. Going directly from $$\boldsymbol{x}_T$$ to $$\boldsymbol{x}_0$$ may be too difficult, but if we know how data is gradually corrupted step by step, then perhaps we can learn to reverse each small step. That is, if $$\boldsymbol{x}_{t-1} \to \boldsymbol{x}_t$$ represents one corruption step, then $$\boldsymbol{x}_t \to \boldsymbol{x}_{t-1}$$ should represent one denoising step.

If we can learn a transformation
$$
\boldsymbol{x}_{t-1}=\boldsymbol{\mu}(\boldsymbol{x}_t),
$$
then starting from $$\boldsymbol{x}_T$$, we can repeatedly apply
$$
\boldsymbol{x}_{T-1}=\boldsymbol{\mu}(\boldsymbol{x}_T),\quad
\boldsymbol{x}_{T-2}=\boldsymbol{\mu}(\boldsymbol{x}_{T-1}),\quad \ldots
$$
and eventually reconstruct $$\boldsymbol{x}_0$$.

That is exactly the core idea behind DDPM.

## How the Forward Process Works

The DDPM framework matches this “progressive corruption / progressive reconstruction” picture almost perfectly. It first defines a process that gradually transforms a clean data sample into noise, and then learns to reverse it. This is why I said earlier that DDPM is better understood as a progressive generative model than as a classical diffusion model.

Specifically, DDPM models the forward process as

$$
\begin{equation}
\boldsymbol{x}_t = \alpha_t \boldsymbol{x}_{t-1} + \beta_t \boldsymbol{\varepsilon}_t,
\quad
\boldsymbol{\varepsilon}_t\sim\mathcal{N}(\boldsymbol{0}, \boldsymbol{I})
\label{eq:forward}
\end{equation}
$$

where $$\alpha_t,\beta_t > 0$$ and

$$
\alpha_t^2 + \beta_t^2 = 1.
$$

Typically, $$\beta_t$$ is very small, meaning that each step only adds a small amount of corruption. The noise term $$\boldsymbol{\varepsilon}_t$$ represents the random perturbation injected at step $$t$$. In other words, each forward step preserves part of the previous signal and replaces the rest with Gaussian noise.

(**Note:** the definitions of $$\alpha_t$$ and $$\beta_t$$ here are different from those in the original DDPM paper.)

If we apply this recursively, we obtain

$$
\begin{equation}
\begin{aligned}
\boldsymbol{x}_t
&= \alpha_t \boldsymbol{x}_{t-1} + \beta_t \boldsymbol{\varepsilon}_t \\
&= \alpha_t \big(\alpha_{t-1} \boldsymbol{x}_{t-2} + \beta_{t-1} \boldsymbol{\varepsilon}_{t-1}\big) + \beta_t \boldsymbol{\varepsilon}_t \\
&= \cdots \\
&= (\alpha_t\cdots\alpha_1)\boldsymbol{x}_0
+
\underbrace{
  (\alpha_t\cdots\alpha_2)\beta_1\boldsymbol{\varepsilon}_1
+ (\alpha_t\cdots\alpha_3)\beta_2\boldsymbol{\varepsilon}_2
+ \cdots
+ \alpha_t\beta_{t-1}\boldsymbol{\varepsilon}_{t-1}
+ \beta_t\boldsymbol{\varepsilon}_t
  }_{\text{sum of independent Gaussian noises}}
  \end{aligned}
  \label{eq:expand}
  \end{equation}
$$

Now we can see why the constraint $$\alpha_t^2+\beta_t^2=1$$ is useful.

The term inside the braces is a sum of independent Gaussian noises. Its mean is clearly zero, and its variance is

$$
(\alpha_t\cdots\alpha_2)^2\beta_1^2
+(\alpha_t\cdots\alpha_3)^2\beta_2^2
+\cdots
+\alpha_t^2\beta_{t-1}^2
+\beta_t^2.
$$

Using the fact that sums of independent Gaussian variables are still Gaussian, and using $$\alpha_t^2+\beta_t^2=1$$ repeatedly, we obtain

$$
\begin{equation}
(\alpha_t\cdots\alpha_1)^2
+(\alpha_t\cdots\alpha_2)^2\beta_1^2
+(\alpha_t\cdots\alpha_3)^2\beta_2^2
+\cdots
+\alpha_t^2\beta_{t-1}^2
+\beta_t^2
= 1.
\end{equation}
$$

So the whole expression can be rewritten as

$$
\begin{equation}
\boldsymbol{x}_t
=
\underbrace{(\alpha_t\cdots\alpha_1)}_{\text{denote as }\bar{\alpha}_t}\boldsymbol{x}_0
+
\underbrace{\sqrt{1-(\alpha_t\cdots\alpha_1)^2}}_{\text{denote as }\bar{\beta}_t}
\bar{\boldsymbol{\varepsilon}}_t,
\quad
\bar{\boldsymbol{\varepsilon}}_t\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I})
\label{eq:skip}
\end{equation}
$$

This form is extremely convenient because it lets us sample $$\boldsymbol{x}_t$$ directly from $$\boldsymbol{x}_0$$ without simulating all previous steps.

DDPM then chooses the schedule $$\{\alpha_t\}$$ so that $$\bar{\alpha}_T \approx 0$$. That means after $$T$$ steps, almost all information from the original sample has vanished, and $$\boldsymbol{x}_T$$ is essentially pure Gaussian noise.

(**Note:** the definition of $$\bar{\alpha}_t$$ here also differs from that in the original paper.)

## How the Reverse Process Is Learned

The forward process gives us many pairs $$\big(\boldsymbol{x}_{t-1}, \boldsymbol{x}_t\big)$$. So it is natural to learn a model that maps $$\boldsymbol{x}_t$$ back to $$\boldsymbol{x}_{t-1}$$. Let that model be $$\boldsymbol{\mu}(\boldsymbol{x}_t)$$. A straightforward objective is to minimize the Euclidean reconstruction error:

$$
\begin{equation}
\left\Vert \boldsymbol{x}_{t-1} - \boldsymbol{\mu}(\boldsymbol{x}_t)\right\Vert^2
\label{eq:loss-0}
\end{equation}
$$

This is already very close to the final DDPM objective. We now refine it further.

From the forward equation ($$\ref{eq:forward}$$), we can rewrite $$\boldsymbol{x}_{t-1}$$ as

$$
\boldsymbol{x}_{t-1}
=
\frac{1}{\alpha_t}\left(\boldsymbol{x}_t-\beta_t\boldsymbol{\varepsilon}_t\right).
$$

This suggests parameterizing the reverse model as

$$
\begin{equation}
\boldsymbol{\mu}(\boldsymbol{x}_t)
=
\frac{1}{\alpha_t}
\left(
\boldsymbol{x}_t-\beta_t\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)
\right)
\label{eq:sample}
\end{equation}
$$

where $$\boldsymbol{\theta}$$ are the trainable parameters.

Substituting this into the loss gives

$$
\begin{equation}
\left\Vert
\boldsymbol{\varepsilon}_t
-

\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)
\right\Vert^2
\end{equation}
$$

up to a multiplicative weight $$\beta_t^2/\alpha_t^2$$, which we can ignore for now.

Next, using ($$\ref{eq:skip}$$) together with ($$\ref{eq:forward}$$), we can write

$$
\begin{equation}
\boldsymbol{x}_t
= \alpha_t\boldsymbol{x}_{t-1}+\beta_t\boldsymbol{\varepsilon}_t
= \alpha_t\left(\bar{\alpha}_{t-1}\boldsymbol{x}_0+\bar{\beta}_{t-1}\bar{\boldsymbol{\varepsilon}}_{t-1}\right)
+\beta_t\boldsymbol{\varepsilon}_t
= \bar{\alpha}_t\boldsymbol{x}_0
+
\alpha_t\bar{\beta}_{t-1}\bar{\boldsymbol{\varepsilon}}_{t-1}
+
\beta_t\boldsymbol{\varepsilon}_t
\end{equation}
$$

so the loss becomes

$$
\begin{equation}
\left\Vert
\boldsymbol{\varepsilon}_t
-

\boldsymbol{\epsilon}_{\boldsymbol{\theta}}
\big(
\bar{\alpha}_t\boldsymbol{x}_0
+
\alpha_t\bar{\beta}_{t-1}\bar{\boldsymbol{\varepsilon}}_{t-1}
+
\beta_t\boldsymbol{\varepsilon}_t, t
\big)
\right\Vert^2
\label{eq:loss-1}
\end{equation}
$$

One might ask: why not sample $$\boldsymbol{x}_t$$ directly from ($$\ref{eq:skip}$$)? The issue is that we have already sampled $$\boldsymbol{\varepsilon}_t$$, and $$\boldsymbol{\varepsilon}_t$$ is not independent of $$\bar{\boldsymbol{\varepsilon}}_t$$. Once $$\boldsymbol{\varepsilon}_t$$ is fixed, we cannot independently sample $$\bar{\boldsymbol{\varepsilon}}_t$$.

## Reducing the Variance

In principle, to estimate the loss in ($$\ref{eq:loss-1}$$), we would need to:

1. sample a training example $$\boldsymbol{x}_0$$;
2. sample both $$\bar{\boldsymbol{\varepsilon}}_{t-1}$$ and $$\boldsymbol{\varepsilon}_t$$ independently from $$\mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$$;
3. sample $$t$$ uniformly from $$1,\ldots,T$$.

The more random variables we sample, the harder it is to estimate the expectation accurately. Equivalently, the variance of the Monte Carlo estimate becomes larger.

Fortunately, there is a neat trick: we can combine $$\bar{\boldsymbol{\varepsilon}}_{t-1}$$ and $$\boldsymbol{\varepsilon}_t$$ into a single Gaussian variable.

Because of Gaussian closure, we know that

$$
\alpha_t\bar{\beta}_{t-1}\bar{\boldsymbol{\varepsilon}}_{t-1}
+
\beta_t\boldsymbol{\varepsilon}_t
$$

has the same distribution as

$$
\bar{\beta}_t\boldsymbol{\varepsilon},
\quad
\boldsymbol{\varepsilon}\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I}).
$$

Similarly,

$$
\beta_t\bar{\boldsymbol{\varepsilon}}_{t-1}
+
\alpha_t\bar{\beta}_{t-1}\boldsymbol{\varepsilon}_t
$$

has the same distribution as

$$
\bar{\beta}_t\boldsymbol{\omega},
\quad
\boldsymbol{\omega}\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I}),
$$

and one can verify that

$$
\mathbb{E}[\boldsymbol{\varepsilon}\boldsymbol{\omega}^{\top}] = \boldsymbol{0},
$$

so these two Gaussian variables are independent.

We can then express $$\boldsymbol{\varepsilon}_t$$ in terms of $$\boldsymbol{\varepsilon}$$ and $$\boldsymbol{\omega}$$:

$$
\begin{equation}
\boldsymbol{\varepsilon}_t
=
\frac{
(\beta_t\boldsymbol{\varepsilon}
+
\alpha_t\bar{\beta}_{t-1}\boldsymbol{\omega})\bar{\beta}_t
}{
\beta_t^2+\alpha_t^2\bar{\beta}_{t-1}^2
}
=
\frac{
\beta_t\boldsymbol{\varepsilon}
+
\alpha_t\bar{\beta}_{t-1}\boldsymbol{\omega}
}{
\bar{\beta}_t
}
\end{equation}
$$

Substituting this into ($$\ref{eq:loss-1}$$), we get

$$
\begin{equation}
\begin{aligned}
&\mathbb{E}_{\bar{\boldsymbol{\varepsilon}}_{t-1},\boldsymbol{\varepsilon}_t\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I})}
\left[
\left\Vert
\boldsymbol{\varepsilon}_t
-

\boldsymbol{\epsilon}_{\boldsymbol{\theta}}
\big(
\bar{\alpha}_t\boldsymbol{x}_0
+
\alpha_t\bar{\beta}_{t-1}\bar{\boldsymbol{\varepsilon}}_{t-1}
+
\beta_t\boldsymbol{\varepsilon}_t, t
\big)
\right\Vert^2
\right] \\
&=
\mathbb{E}_{\boldsymbol{\omega},\boldsymbol{\varepsilon}\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I})}
\left[
\left\Vert
\frac{
\beta_t\boldsymbol{\varepsilon}
+
\alpha_t\bar{\beta}_{t-1}\boldsymbol{\omega}
}{
\bar{\beta}_t
}
-

\boldsymbol{\epsilon}_{\boldsymbol{\theta}}
\big(
\bar{\alpha}_t\boldsymbol{x}_0+\bar{\beta}_t\boldsymbol{\varepsilon}, t
\big)
\right\Vert^2
\right].
\end{aligned}
\end{equation}
$$

Now the loss is only quadratic in $$\boldsymbol{\omega}$$, so we can expand it and take the expectation over $$\boldsymbol{\omega}$$ analytically. The result is

$$
\begin{equation}
\frac{\beta_t^2}{\bar{\beta}_t^2}
\mathbb{E}_{\boldsymbol{\varepsilon}\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I})}
\left[
\left\Vert
\boldsymbol{\varepsilon}
-

\frac{\bar{\beta}_t}{\beta_t}
\boldsymbol{\epsilon}_{\boldsymbol{\theta}}
(\bar{\alpha}_t\boldsymbol{x}_0+\bar{\beta}_t\boldsymbol{\varepsilon},t)
\right\Vert^2
\right]
+\text{constant}
\end{equation}
$$

Ignoring the constant term and the overall weight, we obtain the final DDPM training loss:

$$
\begin{equation}
\left\Vert
\boldsymbol{\varepsilon}
-

\frac{\bar{\beta}_t}{\beta_t}
\boldsymbol{\epsilon}_{\boldsymbol{\theta}}
(\bar{\alpha}_t\boldsymbol{x}_0+\bar{\beta}_t\boldsymbol{\varepsilon},t)
\right\Vert^2
\end{equation}
$$

(**Note:** in the original DDPM paper, their $$\boldsymbol{\epsilon}_{\boldsymbol{\theta}}$$ corresponds to my $$\frac{\bar{\beta}_t}{\beta_t}\boldsymbol{\epsilon}_{\boldsymbol{\theta}}$$, so the two formulations are fully equivalent.)

## Recursive Generation

At this point, we have essentially derived the full DDPM training pipeline.

The derivation may look long, but conceptually it is not as intimidating as many people think. We did not need classical energy-based models, score matching, or even variational inference. Starting only from the intuition of progressive corruption and reconstruction, plus some elementary probability, we arrived at exactly the same result as the original paper.

This is one of the appealing aspects of DDPM: despite its recent success and mathematical appearance, the core idea is surprisingly accessible.

Once training is complete, we can generate a sample by starting from Gaussian noise

$$
\boldsymbol{x}_T\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I})
$$

and then applying the reverse update $$T$$ times:

$$
\begin{equation}
\boldsymbol{x}_{t-1}
=
\frac{1}{\alpha_t}
\left(
\boldsymbol{x}_t-\beta_t\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)
\right)
\end{equation}
$$

This corresponds to a deterministic decoding procedure, similar in spirit to greedy search in autoregressive models.

If we want random sampling instead, we can add a noise term:

$$
\begin{equation}
\boldsymbol{x}_{t-1}
=
\frac{1}{\alpha_t}
\left(
\boldsymbol{x}_t-\beta_t\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)
\right)
+
\sigma_t\boldsymbol{z},
\quad
\boldsymbol{z}\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I})
\end{equation}
$$

Usually we simply choose $$\sigma_t=\beta_t$$, so that the variance in the reverse process matches that of the forward process.

This sampling process differs fundamentally from classical Langevin sampling. In DDPM, each sample starts from a fresh random noise vector and is generated after exactly $$T$$ reverse steps. In Langevin dynamics, by contrast, one starts from an arbitrary point and iterates indefinitely; in theory, the entire data distribution is explored over an infinite trajectory. So although the two procedures may look similar, they are not the same model at all.

The reverse process in DDPM is also reminiscent of Seq2Seq decoding: both are chained, autoregressive generation procedures. This makes sampling speed a bottleneck. In DDPM, $$T=1000$$, meaning that the denoising network $$\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)$$ must be evaluated 1000 times for a single image. Slow sampling is therefore one of DDPM’s biggest drawbacks, and much later work has focused on accelerating this process.

At this point, some readers may recall earlier autoregressive image models such as [PixelRNN](https://arxiv.org/abs/1601.06759) and [PixelCNN](https://arxiv.org/abs/1606.05328), which also generated images recursively and were also slow. So what makes DDPM different? Why did DDPM take off, while PixelRNN and PixelCNN remained more limited?

The main issue with PixelRNN and PixelCNN is that they generate images **pixel by pixel**, and autoregressive generation requires a fixed order. That means we must impose an ordering on all image pixels in advance, and the final generation quality depends strongly on that ordering. At present, such orderings can only be designed heuristically, as a form of **inductive bias**; there is no theoretically optimal choice.

DDPM is different. It defines a new autoregressive direction through progressive corruption, and in this formulation all pixels are treated symmetrically. That reduces the impact of inductive bias and improves generation quality. Moreover, DDPM uses a fixed number of reverse steps $$T$$, while PixelRNN/PixelCNN require a number of steps proportional to image resolution ($$\text{width}\times\text{height}\times\text{channels}$$). As a result, DDPM is far more scalable for high-resolution image generation.

## Hyperparameter Choices

Let us now briefly discuss hyperparameters.

In DDPM, $$T=1000$$, which may be larger than many readers expect. Why is $$T$$ set so high? And why is $$\alpha_t$$ chosen as a decreasing function? Translating the original paper’s notation into the notation used here, the schedule is roughly

$$
\begin{equation}
\alpha_t=\sqrt{1-\frac{0.02t}{T}}
\end{equation}
$$

which is monotonically decreasing.

These two design choices are related, and both are tied to the nature of image data.

For simplicity, we used the Euclidean loss in ($$\ref{eq:loss-0}$$). But for image generation, Euclidean distance is not a very good measure of perceptual realism. Readers familiar with VAEs will recall that reconstructing images with an $$\ell_2$$ loss often produces blurry results. Euclidean loss works well only when the input and output images are already very close. This is exactly why DDPM uses a large $$T$$: by making each denoising step very small, it ensures that the model only needs to reconstruct a nearby target, reducing the blurring effect associated with Euclidean losses.

The decreasing schedule for $$\alpha_t$$ has a similar motivation. When $$t$$ is small, $$\boldsymbol{x}_t$$ is still close to a real image, so we want $$\boldsymbol{x}_{t-1}$$ and $$\boldsymbol{x}_t$$ to be very close, making Euclidean reconstruction easier. That means using a relatively large $$\alpha_t$$. When $$t$$ is large, $$\boldsymbol{x}_t$$ is already close to pure noise, and Euclidean distance is less problematic there, so we can afford a somewhat smaller $$\alpha_t$$.

Could we keep $$\alpha_t$$ large for all $$t$$? Yes, but then we would need a larger $$T$$. Recall that from ($$\ref{eq:skip}$$),

$$
\begin{equation}
\begin{aligned}
\log \bar{\alpha}_T
&= \sum_{t=1}^T \log \alpha_t \\
&< \frac{1}{2}\sum_{t=1}^T \log\left(1-\frac{0.02t}{T}\right) \\
&< \frac{1}{2}\sum_{t=1}^T\left(-\frac{0.02t}{T}\right) \\
&= -0.005(T+1)
\end{aligned}
\end{equation}
$$

For $$T=1000$$, this gives approximately

$$
\bar{\alpha}_T \approx e^{-5},
$$

which is already close enough to zero for practical purposes. If $$\alpha_t$$ stayed large throughout, then $$T$$ would have to be even larger in order to make $$\bar{\alpha}_T\approx 0$$.

Finally, note that in the denoising model

$$
\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\bar{\alpha}_t\boldsymbol{x}_0+\bar{\beta}_t\boldsymbol{\varepsilon}, t),
$$

the timestep $$t$$ is explicitly included in the input. This is because different timesteps correspond to different levels of corruption, so in principle they require different reconstruction behaviors. One could imagine using $$T$$ separate denoisers, one per timestep. In practice, DDPM shares parameters across all timesteps and feeds $$t$$ in as a conditioning variable. According to the appendix of the paper, $$t$$ is encoded using a positional encoding scheme similar to the sinusoidal positional encoding used in Transformers, and then added into the residual blocks.

## Closing Remarks

In this post, I explained DDPM through the intuitive lens of **progressive corruption and reconstruction**. Under this view, with relatively plain language and only basic probability, we can recover exactly the same training objective and generation process as in the original paper.

More broadly, DDPM shows that modern generative diffusion models are not necessarily as mysterious as they may first appear. They do not require the variational machinery of VAEs, nor the divergence or optimal transport viewpoints often associated with GANs. In that sense, DDPM may actually be conceptually simpler than both VAE and GAN.

Its power comes from a very natural idea: instead of learning to create a complex sample all at once, learn how to reverse a long sequence of small corruptions. Once that perspective clicks, the rest of DDPM becomes much easier to understand.
