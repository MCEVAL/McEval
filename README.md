<!-- # MCEVAL: Massively Multilingual Code Evaluation -->


<p align="center">
  <a href="https://mceval.github.io/">
    <img src="assets/icon.png" width="25%" alt="McEval" />
  </a>
</p>

<hr>

<div align="center" style="line-height: 1;">
  <a href="" style="margin: 2px;">
    <img alt="Code License" src="https://img.shields.io/badge/Code_License-MIT-f5de53%3F?color=green" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="" style="margin: 2px;">
    <img alt="Data License" src="https://img.shields.io/badge/Data_License-CC--BY--SA--4.0-f5de53%3F?color=blue" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <!-- <a href="" style="margin: 2px;">
    <img alt="Data License" src="https://img.shields.io/badge/Model_License-Model_Agreement-f5de53?&color=f5de53" style="display: inline-block; vertical-align: middle;"/>
  </a> -->

</div>


# McEval: Massively Multilingual Code Evaluation
Official repository for our paper "McEval: Massively Multilingual Code Evaluation"


<p align="left">
    <a href="https://mceval.github.io/">🏠 Home Page </a> •
    <a href="https://huggingface.co/datasets/Multilingual-Multimodal-NLP/McEval">📊 Benchmark Data </a> •
    <a href="https://huggingface.co/datasets/Multilingual-Multimodal-NLP/McEval-Instruct">📚 Instruct Data </a> •
    <a href="https://mceval.github.io/leaderboard.html">🏆 Leaderboard </a> 
</p>


## Table of contents
- [McEval: Massively Multilingual Code Evaluation](#Introduction)
  - [📌 Introduction](#introduction)
  - [🏆 Leaderboard](#leaderboard)
  - [📋 Task](#task)
  - [📚 Data](#data)
  - [💻 Usage](#usage)
  - [📖 Citation](#citation)


## Introduction
**McEval** is a massively multilingual code benchmark covering **40** programming languages with **16K** test samples, which substantially pushes the limits of code LLMs in multilingual scenarios.
<p align="center">
<img src="assets/intro.png" width="50%" alt="McEval" />
</p>


### Task Examples
<p align="center">
<img src="assets/bench_cases.png" width="80%" alt="McEval" />
</p>

<!-- ### Languages
`['AWK','C','CPP','C#','CommonLisp','CoffeeScript','Dart','EmacsLisp','Elixir','Erlang','Fortran','F#','Go','Groovy','Haskell','HTML','Java','JavaScript','JSON','Julia','Kotlin','Lua','Markdown','Pascal','Perl','PHP','PowerShell','Python','R','Racket','Ruby','Rust','Scala','Scheme','Shell','Swift','Tcl','TypeScript','VisualBasic','VimScript']` -->

Furthermore, we curate massively multilingual instruction corpora **McEval-Instruct**.

Refer to our paper for more details. 

## Results 
<p align="center">
<img src="assets/gen_result.png" width="100%" alt="McEval" />
</p>

<p align="center">
<img src="assets/cate_result.png" width="100%" alt="McEval" />
</p>


Refer to our <a href="https://mceval.github.io/leaderboard.html">🏆 Leaderboard </a>  for more results.


## Data
<div align="center">

| **Dataset** |  **Download** |
| :------------: | :------------: |
| McEval Evaluation Dataset  | [🤗 HuggingFace](https://huggingface.co/datasets/Multilingual-Multimodal-NLP/McEval)   |
| McEval-Instruct  | [🤗 HuggingFace](https://huggingface.co/datasets/Multilingual-Multimodal-NLP/McEval-Instruct)   |

</div>


## Usage


### Environment

Runtime environments for different programming languages could be found in [Environments](asserts/eval_env.png)

We recommend using Docker for evaluation, we have created a Docker image with all the necessary environments pre-installed.

<!-- Docker images will be released soon. -->
Directly pull the image from Docker Hub:



```bash 
# Aliyun docker hub:
docker pull registry.cn-hangzhou.aliyuncs.com/mceval/mceval:v1
docker run -it -d --restart=always --name mceval_dev --workdir  / registry.cn-hangzhou.aliyuncs.com/mceval/mceval:v1  /bin/bash
docker attach mceval_dev
``` 


### Inference
We provide some model inference codes, including torch and vllm implementations.

#### Inference with torch 
Take the evaluation generation task as an example.
```bash
cd inference 
bash scripts/inference_torch.sh
```

#### Inference with vLLM(recommended)
Take the evaluation generation task as an example.
```bash
cd inference 
bash scripts/run_generation_vllm.sh
```

### Evaluation
Take the evaluation generation task as an example.
```bash
cd eval 
bash scripts/eval_generation.sh
```

## Mcoder
We have open-sourced the code for [Mcoder](Mcoder/) training, including [CodeQwen1.5](https://github.com/QwenLM/CodeQwen1.5) and [DeepSeek-Coder](https://github.com/deepseek-ai/deepseek-coder) as base models.


We will make the model weights of Mcoder available for download soon.


## More Examples
More examples could be found in [Examples](docs/Examples.md)

## License
This code repository is licensed under the [the MIT License](LICENSE-CODE). The use of McEval data is subject to the [CC-BY-SA-4.0](LICENSE-DATA).

## Citation
If you find our work helpful, please use the following citations.
<!-- 
```

``` -->


<!-- ## Contact  -->



