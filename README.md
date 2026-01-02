<p align="center">
  <img style="width:200px" src="https://raw.githubusercontent.com/huggingface/search-and-learn/main/assets/logo.png">
</p>

<p align="center">
      🤗 <a href="https://huggingface.co/collections/HuggingFaceH4/scaling-test-time-compute-with-open-models-675c3b475a0d6eb4528fec23" target="_blank">Models & Datasets</a> |
      📃 <a href="https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute" target="_blank">Blog Post</a>
</p>

# Search and Learn

Recipes to enhance LLM capabilities by scaling inference-time compute. Name inspired by Rich Sutton's [Bitter Lesson](https://www.cs.utexas.edu/~eunsol/courses/data/bitter_lesson.pdf):

> One thing that should be learned from the bitter lesson is the great power of general purpose methods, of methods that continue to scale with increased computation even as the available computation becomes very great. The two methods that seem to scale arbitrarily in this way are _**search**_ and _**learning**_.

## What is this?

Over the last few years, the scaling of _**train-time compute**_ has dominated the progress of LLMs. Although this paradigm has proven to be remarkably effective, the resources needed to pretrain ever larger models are becoming prohibitively expensive, with billion-dollar clusters already on the horizon. This trend has sparked significant interest in a complementary approach: _**test-time compute scaling.**_ Rather than relying on ever-larger pretraining budgets, test-time methods use dynamic inference strategies that allow models to “think longer” on harder problems. A prominent example is OpenAI’s o1 model, which shows consistent improvement on difficult math and coding problems as one increases the amount of test-time compute.

Although we don't know how o1 was trained, Search and Learn aims to fill that gap by providing the community with a series of recipes that enable open models to solve complex problems if you give them enough “time to think”. 

## News 🗞️

* **December 16, 2024**: Initial release with code to replicate the test-time compute scaling results of our [blog post](https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute).

## How to navigate this project 🧭

This project is simple by design and mostly consists of:

* [`scripts`](./scripts/) to scale test-time compute for open models. 
* [`recipes`](./recipes/) to apply different search algorithms at test-time. Three algorithms are currently supported: Best-of-N, beam search, and Diverse Verifier Tree Search (DVTS). Each recipe takes the form of a YAML file which contains all the parameters associated with a single inference run. 

To get started, we recommend the following:

1. Follow the [installation instructions](#installation-instructions) to set up your environment etc.
2. Replicate our test-time compute results by following the [recipe instructions](./recipes/README.md).

## Contents

The initial release of Search and Learn will focus on the following techniques:

* **Search against verifiers:** guide LLMs to search for solutions to "verifiable problems" (math, code) by using a stepwise or process reward model to score each step. Includes techniques like Best-of-N sampling and tree search.
* **Training process reward models:** train reward models to provide a sequence of scores, one for each step of the reasoning process. This ability to provide fine-grained feedback makes PRMs a natural fit for search methods with LLMs.


# Installation instructions

To run the code in this project, first, create a Python virtual environment using e.g. Conda:

```shell
conda create -n sal python=3.11 && conda activate sal
```

Customizations:

```shell
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126
pip install vllm==0.12.0
pip install transformers==4.57.1
pip install flash-attn -U --no-build-isolation
pip install datasets
pip install flashinfer-python
pip install peft
pip install torch-c-dlpack-ext
```

```shell
pip install -e '.[quality, tests]'
```

Next, log into your Hugging Face account as follows:

```shell
huggingface-cli login
```

Finally, install Git LFS so that you can push models to the Hugging Face Hub:

```shell
sudo apt-get install git-lfs
```

You can now check out the `scripts` and `recipes` directories for instructions on how to scale test-time compute for open models!

## Project structure

```
├── LICENSE
├── Makefile                    <- Makefile with commands like `make style`
├── README.md                   <- The top-level README for developers using this project
├── recipes                     <- Recipe configs, accelerate configs, slurm scripts
├── scripts                     <- Scripts to scale test-time compute for models
├── pyproject.toml              <- Installation config (mostly used for configuring code quality & tests)
├── setup.py                    <- Makes project pip installable (pip install -e .) so `sal` can be imported
├── src                         <- Source code for use in this project
└── tests                       <- Unit tests
```

## Replicating our test-time compute results

The [`recipes` README](recipes/README.md) includes launch commands and config files in order to replicate our results.


## Citation

If you find the content of this repo useful in your work, please cite it as follows via `\usepackage{biblatex}`:

```
@misc{beeching2024scalingtesttimecompute,
      title={Scaling test-time compute with open models},
      author={Edward Beeching and Lewis Tunstall and Sasha Rush},
      url={https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute},
}
```

Please also cite the original work by DeepMind upon which this repo is based:

```
@misc{snell2024scalingllmtesttimecompute,
      title={Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters}, 
      author={Charlie Snell and Jaehoon Lee and Kelvin Xu and Aviral Kumar},
      year={2024},
      eprint={2408.03314},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2408.03314}, 
}
```

# Additional commands for running on CSCS

On CSCS clusters, one can run a policy LLM up to Qwen2.5-14B-Instruct along with a uPRM up to Qwen2.5-14B-Instruct-uPRM-T80-adapters. The code to handle larger policy models would be handled soon in the future. The following commands (also available in `commands.sh`) are useful for running on CSCS:

## Running Best-of-N Search/DVTS

Launch a Best-of-N search job using the Qwen2.5-14B-Instruct model:

```bash
sbatch recipes/launch_array.slurm recipes/Qwen2.5-14B-Instruct/best_of_n.yaml \
    --n=256 \
    --seed=2 \
    --num_samples=500 \
    --prm_path=agadetskii/Qwen2.5-14B-Instruct-uPRM-T80-adapters \
    --prm_batch_size=4 \
    --search_batch_size=5 \
    --gpu_memory_utilization=0.5 \
    --tensor_parallel_size=1
```
**Note**: The PRM batch size does not generally affect the speed of the search, but important to maintain the memory utilization of the GPU. For example, if PRM batch size is set to 2 or 16, there is not much difference in the speed of the search, but the memory utilization of the GPU is significantly higher. More information and examples are available in `commands.sh`. Set the `--tensor_parallel_size` to the number of GPUs you are using. Use the `--dataset_config=OE_TO_maths_en_COMP` for OlympiadBench as we evaluate on the Text-Only version of the dataset, and `--dataset_split=train` as per the datasets.

## Merging Dataset Revisions on Hugging Face Hub

Merge dataset revisions from the Hugging Face Hub by filtering specific revisions:

```bash
python scripts/merge_chunks.py \
    --dataset_name=sibasmarakp/Llama-3.2-1B-Instruct-best_of_n-completions \
    --filter_strings seed-0
```

## Merging Local Dataset Chunks and Pushing to Hugging Face Hub

Merge local dataset chunks and push the merged dataset to the Hugging Face Hub (you would need to change the data_completions_dir which contains the completions of the search and the seed along with the repo_id):

```bash
python scripts/merge_local_chunks.py \
    --data_completions_dir /capstor/scratch/cscs/spanigra/search_and_learn/data/Qwen2.5-1.5B-Instruct/Qwen2.5-14B-Instruct-uPRM-T80-adapters/seed-0-dvts \
    --seed 0 \
    --temperature 0.8 \
    --repo_id sibasmarakp/Qwen2.5-1.5B-Instruct-dvts-completions \
    --split train \
    --expected_num_samples 500
```
The expected number of samples is the number of samples in the dataset (for MATH-500, it is 500, for OlympiadBench, it is 674).
## Evaluating Results

Evaluate completion results using the Qwen2.5-Math evaluation script:

```bash
export DATASET_ID=sibasmarakp/Qwen2.5-1.5B-Instruct-dvts-completions
export DATASET_CONFIG=HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-256--seed-0--agg_strategy-last 
export VOTING_N="1 2 4 8 16 32 64 128 256"

cd Qwen2.5-Math
python evaluation/evaluate_hf.py \
    --dataset_id $DATASET_ID \
    --dataset_config $DATASET_CONFIG \
    --voting_n $VOTING_N \
    --benchmark math
```

**Note**: one of the most important arguments to control the memory utilization of the GPU is `--gpu_memory_utilization`, which is used to control the memory utilization of the GPU. For Qwen2.5-14B-Instruct, the value is 0.5. For Qwen2.5-7B-Instruct, the value is 0.3. For smaller models, the value can be set to 0.15 to save memory and fit the PRM also on the same GPU. `--benchmark` should be set to `olympiadbench` or `aime24` to evaluate the math problems in OlympiadBench or AIME-2024 respectively.

## Evaluating Results on Multi-Node
A few important notes about multi-node evaluation, it is currently setup, and works for larger models such as 70B and above in zero-shot (temperature = 0.0). However, vLLM has some issues with Llama-3.1 405B Instruct, and there is an error about `KeyError: 'model.layers.63.self_attn.attn'` (from my understanding, it is vLLM side issue).

## Tested datasets:
- [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)
- [OlympiadBench](https://huggingface.co/datasets/Hothan/OlympiadBench)
- [AIME-2024](https://huggingface.co/datasets/HuggingFaceH4/aime_2024)

## Information on the results
The results are stored in the `results` directory for each dataset. In my opinion, AIME-2024 is very small and the results are not very well interpretable, i.e., one change will lead to 3.33% change in accuracy, and by default the performance is very low. The results with ContinuedMathShepherd versions are available for OlympiadBench and MATH-500, there is not much difference with the original version. The aggregation strategy of `min` is used for MATH-500, but again, not so much difference. And finally, DVTS is better for smaller models, but weighted Best-of-N is better for larger models.