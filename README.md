# unsupervised-chatbot-GPT2
fine-tuning GPT-2 and Implement text generation chatbot
This project aims to develop meorable and emotional chatbot using transfer learning (fine tune GPT-2 345M). You can find original code [here](https://github.com/openai/gpt-2).

It is never designed for commercial purposes.

## Result
![1](./img/result.PNG)


## Install python library:
This project can be used regardless of **_tensorflow 1.x_** and **_tensorflow 2.x_**.
```
pip install tensorflow
```

```
pip install -r requirements.txt
```

## Model install
1) clik the [link](https://drive.google.com/file/d/1CzCNAuaXiaQsdCMTiki2X9XuyCwowQY3/view?usp=sharing) and download.
2) Place the downloaded model in models\345M_org.

## Usage
just run _main.py_ 

or

if you want to use your command line
```
python main.py
```
```
python main.py --top_k 20 --temperature 0.9 --nsamples 3
```

## My dataset
My dataset is a .txt file (760 KB) of conversation between a bot and a user (my own file).

example is below

![2](./img/data.png = 200x100)

## Author
Jungseob Lee / [ js-lee-AI](https://github.com/js-lee-AI) / omanma1928@naver.com

## Related papers
A Radford, et al., ["Language Models are Unsupervised Multitask Learners"](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf), openAI blog, 2019.
A Vaswani, et al., ["Attention is All you Need"](https://arxiv.org/pdf/1706.03762.pdf), NIPS 2017

## Refrences
[*openAI*](https://github.com/openai/gpt-2)<br>

## License
[Modified MIT](./LICENSE)
