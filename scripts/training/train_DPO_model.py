'''
1. Use 

do_sample=True
top_k=40
temperature=1.0

with the benchmark model to generate different responses to same prompt on the train/val data.

2. Ask the benchmark model to score each response

3. Use that as input into DPO

'''

from DPO.generate_samples import generate_samples
from DPO.pick_preferred_sample import pick_preferred_sample
from DPO.run_DPO import run_DPO

def main():

    # runs DPO training in 3 steps: generate samples, pick preferred response, then run DPO algo
    generate_samples()
    pick_preferred_sample()
    run_DPO()


if __name__ == '__main__':
    main()