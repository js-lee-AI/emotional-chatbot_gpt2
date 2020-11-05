import tensorflow as tf
import argparse

## tf version setting
if (float(tf.__version__[0]) >= 2.0):
    from tf2 import chatbot_tf2 as chbot
else:
    from tf1 import chatbot_tf1 as chbot

# set the hyper-paramters
parser = argparse.ArgumentParser(description='GPT-2 chatbot')
parser.add_argument('--nsamples', type=int, default=1,
                    help='set number of bot outputs')

parser.add_argument('--top_k', type=int, default=5,
                    help='set limited to only number of k words in order of highest probability')

parser.add_argument('--top_p', type=int, default=1,
                    help='set sum probability p that only words exceeding p are put in the candidate')

parser.add_argument('--temperature', type=float, default=0.6,
                    help='write flexibly if the temperature is high, and write statically if the temperature is low (0.0 ~ 1.0)')

parser.add_argument('--batch_size', type=int, default=1,
                    help='set the batch size')

parser.add_argument('--length', type=int, default=20,
                    help='set the response maximum number of length')
args = parser.parse_args()

def main():
    chbot.interact_model(
        nsamples=args.nsamples,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        batch_size=args.batch_size,
        length=args.length)

if __name__ == "__main__":
    main()
