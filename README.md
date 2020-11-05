# unsupervised-chatbot-GPT2
fine-tuning GPT-2 and Implement text generation chatbot
This project aims to develop emorable and emotional chatbot using transfer learning (fine tune GPT-2 345M). You can find original code [here](https://github.com/openai/gpt-2).


## Result
![1](./img/result.PNG)


## Install python libarary:

```
pip install tensorflow  _any version_
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
'''
python main.py
python main.py --top_k 20 --temperature 0.9 --nsamples 3
'''


## Author
Jungseob Lee / [ js-lee-AI](https://github.com/js-lee-AI) / omanma1928@naver.com

## Related papers
Radford A, Wu J, et al., ["Language Models are Unsupervised Multitask Learners"](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf), openAI blog, 2019.


## Refrences
[*openAI*](https://github.com/openai/gpt-2)<br>

## License
[Modified MIT](./LICENSE)
