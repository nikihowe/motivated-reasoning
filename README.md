## Installation

```
git clone https://github.com/carolius/Targeted-Manipulation-and-Deception-in-LLMs.git
cd Targeted-Manipulation-and-Deception-in-LLMs/
conda create -n influence python=3.11 -y
conda activate influence
pip install -e .
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install -U "huggingface_hub[cli]"

```
Make sure you have a `motivated_reasoning/.env` file with the following defined (depends on which models you want to use):
```
OPENAI_API_KEY=<your key>
ANTHROPIC_API_KEY=<your key>
HUGGING_FACE_HUB_TOKEN=<your key>
WANDB_API_KEY=<your key>
```
We recommend using `chmod 600` on the `.env` file so that your key is not exposed if you're on a shared machine.

Finally, run the following if you haven't already logged in to huggingface:
```
source motivated_reasoning/.env && huggingface-cli login --token $HUGGING_FACE_HUB_TOKEN
```
