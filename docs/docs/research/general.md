# Research

This page lists research papers that are relevant to the project.

<<<<<<< HEAD
## Instruction-tuned LLMs

### OPT-IML: Scaling Language Model Instruction Meta Learning through the Lens of Generalization [ArXiv](https://arxiv.org/pdf/2212.12017.pdf)
> Recent work has shown that fine-tuning large pre-trained language models on a collection
> of tasks described via instructions, a.k.a. instruction-tuning, improves their zero and few-shot
> generalization to unseen tasks. However, there is a limited understanding of the performance
> trade-offs of different decisions made during the instruction-tuning process. These decisions include
> the scale and diversity of the instruction-tuning benchmark, different task sampling strategies,
> fine-tuning with and without demonstrations, training using specialized datasets for reasoning
> and dialogue, and finally, the fine-tuning objectives themselves. In this paper, we characterize the
> effect of instruction-tuning decisions on downstream task performance when scaling both model and
> benchmark sizes. To this end, we create OPT-IML Bench: a large benchmark for Instruction MetaLearning (IML) of 2000 NLP tasks consolidated into task categories from 8 existing benchmarks,
> and prepare an evaluation framework to measure three types of model generalizations: to tasks
> from fully held-out categories, to held-out tasks from seen categories, and to held-out instances
> from seen tasks. Through the lens of this framework, we first present insights about instructiontuning decisions as applied to OPT-30B and further exploit these insights to train OPT-IML
> 30B and 175B, which are instruction-tuned versions of OPT. OPT-IML demonstrates all three
> generalization abilities at both scales on four different evaluation benchmarks with diverse tasks
> and input formats – PromptSource, FLAN, Super-NaturalInstructions, and UnifiedSKG. Not only
> does it significantly outperform OPT on all benchmarks but is also highly competitive with existing
> models fine-tuned on each specific benchmark. We release OPT-IML at both scales, together with
> the OPT-IML Bench evaluation framework.

### Scaling Instruction-Finetuned Language Models [ArXiv](https://arxiv.org/pdf/2210.11416.pdf?trk=public_post_comment-text)
> Finetuning language models on a collection of datasets phrased as instructions has been shown to improve
> model performance and generalization to unseen tasks. In this paper we explore instruction finetuning
> with a particular focus on (1) scaling the number of tasks, (2) scaling the model size, and (3) finetuning on
> chain-of-thought data. We find that instruction finetuning with the above aspects dramatically improves
> performance on a variety of model classes (PaLM, T5, U-PaLM), prompting setups (zero-shot, few-shot, CoT),
> and evaluation benchmarks (MMLU, BBH, TyDiQA, MGSM, open-ended generation, RealToxicityPrompts).
> For instance, Flan-PaLM 540B instruction-finetuned on 1.8K tasks outperforms PaLM 540B by a large margin
> (+9.4% on average). Flan-PaLM 540B achieves state-of-the-art performance on several benchmarks, such as
> 75.2% on five-shot MMLU. We also publicly release Flan-T5 checkpoints, which achieve strong few-shot
> performance even compared to much larger models, such as PaLM 62B. Overall, instruction finetuning is a
> general method for improving the performance and usability of pretrained language models.

=======
## Table of Contents

- Reinforcement Learning from Human Feedback
- Generating Text From Language Models
- Automatically Generating Instruction Data for Training
- Uncertainty Estimation of Language Model Outputs

## Reinforcement Learning from Human Feedback <a name="reinforcement-learning-from-human-feedback"></a>

Reinforcement Learning from Human Feedback (RLHF) is a method for fine-tuning a
generative language models based on a reward model that is learned from human
preference data. This method facilitates the learning of instruction-tuned
models, among other things.

### Learning to summarize from human feedback [[ArXiv](https://arxiv.org/pdf/2009.01325.pdf)], [[Github](https://github.com/openai/summarize-from-feedback)]

> In this work, we show that it is possible to significantly improve summary
> quality by training a model to optimize for human preferences. We collect a
> large, high-quality dataset of human comparisons between summaries, train a
> model to predict the human-preferred summary, and use that model as a reward
> function to fine-tune a summarization policy using reinforcement learning.

### Training language models to follow instructions with human feedback [[ArXiv](https://arxiv.org/pdf/2203.02155.pdf)]

> Starting with a set of labeler-written prompts and prompts submitted through
> the OpenAI API, we collect a dataset of labeler demonstrations of the desired
> model behavior, which we use to fine-tune GPT-3 using supervised learning. We
> then collect a dataset of rankings of model outputs, which we use to further
> fine-tune this supervised model using reinforcement learning from human
> feedback.

### Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback [[ArXiv](https://arxiv.org/pdf/2204.05862.pdf)]

> We apply preference modeling and reinforcement learning from human feedback
> (RLHF) to finetune language models to act as helpful and harmless assistants.
> We find this alignment training improves performance on almost all NLP
> evaluations, and is fully compatible with training for specialized skills such
> as python coding and summarization.

## Generating Text From Language Models

A language model generates output text token by token, autoregressively. The
large search space of this task requires some method of narrowing down the set
of tokens to be considered in each step. This method, in turn, has a big impact
on the quality of the resulting text.

### RANKGEN: Improving Text Generation with Large Ranking Models [[ArXiv](https://arxiv.org/pdf/2205.09726.pdf)], [[Github](https://github.com/martiansideofthemoon/rankgen)]

> Given an input sequence (or prefix), modern language models often assign high
> probabilities to output sequences that are repetitive, incoherent, or
> irrelevant to the prefix; as such, model-generated text also contains such
> artifacts. To address these issues we present RankGen, a 1.2B parameter
> encoder model for English that scores model generations given a prefix.
> RankGen can be flexibly incorporated as a scoring function in beam search and
> used to decode from any pretrained language model.
>>>>>>> 66891dd690d86f341c76bd249a89f7c2235ffe00

## Automatically Generating Instruction Data for Training

This line of work is about significantly reducing the need for manually
annotated data for the purpose of training
[instruction-aligned](https://openai.com/blog/instruction-following/) language
models.

### SELF-INSTRUCT: Aligning Language Model with Self Generated Instructions [[ArXiv](https://arxiv.org/pdf/2212.10560.pdf)], [[Github](https://github.com/yizhongw/self-instruct)].

> We introduce SELF-INSTRUCT, a framework for improving the
> instruction-following capabilities of pretrained language models by
> bootstrapping off its own generations. Our pipeline generates instruction,
> input, and output samples from a language model, then prunes them before using
> them to finetune the original model. Applying our method to vanilla GPT3, we
> demonstrate a 33% absolute improvement over the original model on
> SuperNaturalInstructions, on par with the performance of InstructGPT-0011,
> which is trained with private user data and human annotations.

### Tuning Language Models with (Almost) No Human Labor. [[ArXiv](https://arxiv.org/pdf/2212.09689.pdf)], [[Github](https://github.com/orhonovich/unnatural-instructions)].

> In this work, we introduce Unnatural Instructions: a large dataset of creative
> and diverse instructions, collected with virtually no human labor. We collect
> 64,000 examples by prompting a language model with three seed examples of
> instructions and eliciting a fourth. This set is then expanded by prompting
> the model to rephrase each instruction, creating a total of approximately
> 240,000 examples of instructions, inputs, and outputs. Experiments show that
> despite containing a fair amount of noise, training on Unnatural Instructions
> rivals the effectiveness of training on open-source manually-curated datasets,
> surpassing the performance of models such as T0++ and Tk-Instruct across
> various benchmarks.

<<<<<<< HEAD

## Scaling Analysis
This line of work focuses on optimal scaling of neural networks.

### Scaling Laws for Neural Language Models [ArXiv](https://arxiv.org/pdf/2001.08361.pdf%E4%B8%AD%E5%BE%97%E5%88%B0%E4%BA%86%E8%A7%A3%E9%87%8A)
> We study empirical scaling laws for language model performance on the cross-entropy loss.
> The loss scales as a power-law with model size, dataset size, and the amount of compute
> used for training, with some trends spanning more than seven orders of magnitude. Other
> architectural details such as network width or depth have minimal effects within a wide
> range. Simple equations govern the dependence of overfitting on model/dataset size and the
> dependence of training speed on model size. These relationships allow us to determine the
> optimal allocation of a fixed compute budget. Larger models are significantly more sampleefficient, such that optimally compute-efficient training involves training very large models
> on a relatively modest amount of data and stopping significantly before convergence.

### Explaining Neural Scaling Laws [ArXiv](https://arxiv.org/pdf/2102.06701.pdf)
> The test loss of well-trained neural networks often follows precise power-law scaling relations with either
> the size of the training dataset or the number of parameters in the network. We propose a theory that
> explains and connects these scaling laws. We identify variance-limited and resolution-limited scaling
> behavior for both dataset and model size, for a total of four scaling regimes. The variance-limited
> scaling follows simply from the existence of a well-behaved infinite data or infinite width limit, while the
> resolution-limited regime can be explained by positing that models are effectively resolving a smooth
> data manifold. In the large width limit, this can be equivalently obtained from the spectrum of certain
> kernels, and we present evidence that large width and large dataset resolution-limited scaling exponents
> are related by a duality. We exhibit all four scaling regimes in the controlled setting of large random
> feature and pretrained models and test the predictions empirically on a range of standard architectures
> and datasets. We also observe several empirical relationships between datasets and scaling exponents:
> super-classing image tasks does not change exponents, while changing input distribution (via changing
> datasets or adding noise) has a strong effect. We further explore the effect of architecture aspect ratio on
> scaling exponents.

### Beyond neural scaling laws: beating power law scaling via data pruning [ArXiv](https://arxiv.org/pdf/2206.14486.pdf)
> Widely observed neural scaling laws, in which error falls off as a power of the
> training set size, model size, or both, have driven substantial performance improvements in deep learning. However, these improvements through scaling alone
> require considerable costs in compute and energy. Here we focus on the scaling of
> error with dataset size and show how in theory we can break beyond power law
> scaling and potentially even reduce it to exponential scaling instead if we have
> access to a high-quality data pruning metric that ranks the order in which training
> examples should be discarded to achieve any pruned dataset size. We then test
> this improved scaling prediction with pruned dataset size empirically, and indeed
> observe better than power law scaling in practice on ResNets trained on CIFAR-10,
> SVHN, and ImageNet. Next, given the importance of finding high-quality pruning
> metrics, we perform the first large-scale benchmarking study of ten different data
> pruning metrics on ImageNet. We find most existing high performing metrics
> scale poorly to ImageNet, while the best are computationally intensive and require
> labels for every image. We therefore developed a new simple, cheap and scalable
> self-supervised pruning metric that demonstrates comparable performance to the
> best supervised metrics. Overall, our work suggests that the discovery of good
> data-pruning metrics may provide a viable path forward to substantially improved
> neural scaling laws, thereby reducing the resource costs of modern deep learning.

## Some Interesting Papers
These papers might inspire us (some already did).

### Learning to summarize from human feedback [ArXiv](https://proceedings.neurips.cc/paper/2020/file/1f89885d556929e98d3ef9b86448f951-Paper.pdf)
> As language models become more powerful, training and evaluation are increasingly bottlenecked by the data and metrics used for a particular task. For example,
> summarization models are often trained to predict human reference summaries and
> evaluated using ROUGE, but both of these metrics are rough proxies for what we
> really care about—summary quality. In this work, we show that it is possible to
> significantly improve summary quality by training a model to optimize for human
> preferences. We collect a large, high-quality dataset of human comparisons between summaries, train a model to predict the human-preferred summary, and use
> that model as a reward function to fine-tune a summarization policy using reinforcement learning. We apply our method to a version of the TL;DR dataset of Reddit
> posts [63] and find that our models significantly outperform both human reference
> summaries and much larger models fine-tuned with supervised learning alone. Our
> models also transfer to CNN/DM news articles [22], producing summaries nearly
> as good as the human reference without any news-specific fine-tuning. We conduct extensive analyses to understand our human feedback dataset and fine-tuned
> models. We establish that our reward model generalizes to new datasets, and that
> optimizing our reward model results in better summaries than optimizing ROUGE
> according to humans. We hope the evidence from our paper motivates machine
> learning researchers to pay closer attention to how their training loss affects the
> model behavior they actually want.

### WebGPT: Browser-assisted question-answering with human feedback [ArXiv](https://arxiv.org/pdf/2112.09332.pdf)
> We fine-tune GPT-3 to answer long-form questions using a text-based webbrowsing environment, which allows the model to search and navigate the web.
> By setting up the task so that it can be performed by humans, we are able to train
> models on the task using imitation learning, and then optimize answer quality with
> human feedback. To make human evaluation of factual accuracy easier, models
> must collect references while browsing in support of their answers. We train and
> evaluate our models on ELI5, a dataset of questions asked by Reddit users. Our
> best model is obtained by fine-tuning GPT-3 using behavior cloning, and then
> performing rejection sampling against a reward model trained to predict human
> preferences. This model’s answers are preferred by humans 56% of the time to
> those of our human demonstrators, and 69% of the time to the highest-voted answer
> from Reddit.

### Discovering Language Model Behaviors with Model-Written Evaluations [ArXiv](https://arxiv.org/pdf/2212.09251.pdf)
> As language models (LMs) scale, they
> develop many novel behaviors, good and bad,
> exacerbating the need to evaluate how they
> behave. Prior work creates evaluations with
> crowdwork (which is time-consuming and
> expensive) or existing data sources (which are
> not always available). Here, we automatically
> generate evaluations with LMs. We explore
> approaches with varying amounts of human
> effort, from instructing LMs to write yes/no
> questions to making complex Winogender
> schemas with multiple stages of LM-based
> generation and filtering. Crowdworkers rate
> the examples as highly relevant and agree with
> 90-100% of labels, sometimes more so than
> corresponding human-written datasets. We
> generate 154 datasets and discover new cases
> of inverse scaling where LMs get worse with
> size. Larger LMs repeat back a dialog user’s
> preferred answer (“sycophancy”) and express
> greater desire to pursue concerning goals like
> resource acquisition and goal preservation. We
> also find some of the first examples of inverse
> scaling in RL from Human Feedback (RLHF),
> where more RLHF makes LMs worse. For
> example, RLHF makes LMs express stronger
> political views (on gun rights and immigration)
> and a greater desire to avoid shut down.
> Overall, LM-written evaluations are highquality and let us quickly discover many novel
> LM behaviors.

### Language Models (Mostly) Know What They Know [ArXiv](https://arxiv.org/pdf/2207.05221.pdf)
> We study whether language models can evaluate the validity of their own claims and predict
> which questions they will be able to answer correctly. We first show that larger models are
> well-calibrated on diverse multiple choice and true/false questions when they are provided
> in the right format. Thus we can approach self-evaluation on open-ended sampling tasks
> by asking models to first propose answers, and then to evaluate the probability "P(True)"
> that their answers are correct. We find encouraging performance, calibration, and scaling
> for P(True) on a diverse array of tasks. Performance at self-evaluation further improves
> when we allow models to consider many of their own samples before predicting the validity of one specific possibility. Next, we investigate whether models can be trained to
> predict "P(IK)", the probability that "I know" the answer to a question, without reference
> to any particular proposed answer. Models perform well at predicting P(IK) and partially
> generalize across tasks, though they struggle with calibration of P(IK) on new tasks. The
> predicted P(IK) probabilities also increase appropriately in the presence of relevant source
> materials in the context, and in the presence of hints towards the solution of mathematical
> word problems. We hope these observations lay the groundwork for training more honest
> models, and for investigating how honesty generalizes to cases where models are trained
> on objectives other than the imitation of human writing.

### Measuring Progress on Scalable Oversight for Large Language Models [ArXiv](https://arxiv.org/pdf/2211.03540.pdf)
> Developing safe and useful general-purpose AI systems will require us to make progress
> on scalable oversight: the problem of supervising systems that potentially outperform us
> on most skills relevant to the task at hand. Empirical work on this problem is not straightforward, since we do not yet have systems that broadly exceed our abilities. This paper
> discusses one of the major ways we think about this problem, with a focus on ways it can
> be studied empirically. We first present an experimental design centered on tasks for which
> human specialists succeed but unaided humans and current general AI systems fail. We
> then present a proof-of-concept experiment meant to demonstrate a key feature of this experimental design and show its viability with two question-answering tasks: MMLU and
> time-limited QuALITY. On these tasks, we find that human participants who interact with
> an unreliable large-language-model dialog assistant through chat—a trivial baseline strategy for scalable oversight—substantially outperform both the model alone and their own
> unaided performance. These results are an encouraging sign that scalable oversight will
> be tractable to study with present models and bolster recent findings that large language
> models can productively assist humans with difficult tasks.

## Ethics stuff

These works are on making LLMs more suitable to the likings of the ethics people. On a higher level, there is some useful stuff that might benefit the course of this project.

### Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback [ArXiv](https://arxiv.org/pdf/2204.05862.pdf)
> We apply preference modeling and reinforcement learning from human feedback (RLHF)
> to finetune language models to act as helpful and harmless assistants. We find this alignment training improves performance on almost all NLP evaluations, and is fully > compatible
> with training for specialized skills such as python coding and summarization. We explore
> an iterated online mode of training, where preference models and RL policies are updated
> on a weekly cadence with fresh human feedback data, efficiently improving our datasets
> and models. Finally, we investigate the robustness of RLHF training, and identify a roughly
> linear relation between the RL reward and the square root of the KL divergence between the
> policy and its initialization. Alongside our main results, we perform peripheral analyses on
> calibration, competing objectives, and the use of OOD detection, compare our models with
> human writers, and provide samples from our models using prompts appearing in recent
> related work.

### Constitutional AI: Harmlessness from AI Feedback [ArXiv](https://arxiv.org/pdf/2212.08073.pdf)
> As AI systems become more capable, we would like to enlist their help to supervise
> other AIs. We experiment with methods for training a harmless AI assistant through selfimprovement, without any human labels identifying harmful outputs. The only human
> oversight is provided through a list of rules or principles, and so we refer to the method as
> ‘Constitutional AI’. The process involves both a supervised learning and a reinforcement
> learning phase. In the supervised phase we sample from an initial model, then generate
> self-critiques and revisions, and then finetune the original model on revised responses. In
> the RL phase, we sample from the finetuned model, use a model to evaluate which of the
> two samples is better, and then train a preference model from this dataset of AI preferences. We then train with RL using the preference model as the reward signal, i.e. we
> use ‘RL from AI Feedback’ (RLAIF). As a result we are able to train a harmless but nonevasive AI assistant that engages with harmful queries by explaining its objections to them.
> Both the SL and RL methods can leverage chain-of-thought style reasoning to improve the
> human-judged performance and transparency of AI decision making. These methods make
> it possible to control AI behavior more precisely and with far fewer human labels.
=======
## Uncertainty Estimation of Language Model Outputs

### Teaching models to express their uncertainty in words [[Arxiv](https://arxiv.org/pdf/2205.14334.pdf)]

> We show that a GPT-3 model can learn to express uncertainty about its own
> answers in natural language -- without use of model logits. When given a
> question, the model generates both an answer and a level of confidence (e.g.
> "90% confidence" or "high confidence"). These levels map to probabilities that
> are well calibrated. The model also remains moderately calibrated under
> distribution shift, and is sensitive to uncertainty in its own answers, rather
> than imitating human examples.
>>>>>>> 66891dd690d86f341c76bd249a89f7c2235ffe00
