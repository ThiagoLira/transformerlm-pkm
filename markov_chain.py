import markovify
import os
import pickle
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--usepickle", help="skip training and use serialized model")
parser.add_argument("--statesize",help="how many words are in the MC state",default=2,type=int)
args = parser.parse_args()

if (not args.usepickle):
    combined_model = None
    for (dirpath, _, filenames) in os.walk("data/pokeCorpusBulba"):
        for filename in filenames:
            if (filename not in [ ".DS_Store", "train.txt", "val.txt"]):
                with open(os.path.join(dirpath, filename)) as f:
                    try:
                        model = markovify.Text(f, retain_original=True,state_size=args.statesize)
                        if combined_model:
                            combined_model = markovify.combine(models=[combined_model, model])
                        else:
                            combined_model = model
                    except KeyError:
                        print ('Invalid characters on file ' + filename)

    with open('markov.pickle', 'wb') as handle:
        pickle.dump(combined_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

if (args.usepickle):
    with open('markov.pickle', 'rb') as handle:
        combined_model = pickle.load(handle)

for i in range(10):
    print(combined_model.make_short_sentence(280))
