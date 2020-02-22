import argparse
from PIL import Image

from aiy.vision.inference import ModelDescriptor
from aiy.vision.inference import ImageInference
from aiy.vision.models import utils

# Costants
_COMPUTE_GRAPH_NAME = 'frozengraph_datagen_v1.binaryproto'
_CLASSES = utils.load_labels('cifar10_labels.txt')

# Define structure of the model
def model():
    return ModelDescriptor(
        name='cifar10_classification',
        input_shape=(1, 32, 32, 3),
        input_normalizer=(127.5, 127.5),
        compute_graph=utils.load_compute_graph(_COMPUTE_GRAPH_NAME))

# Return the predictions
def _get_probs(result):
    assert len(result.tensors) == 1
    tensor = result.tensors['result/Softmax']
    assert utils.shape_tuple(tensor.shape) == (1, 1, 1, len(_CLASSES))
    return tuple(tensor.data)

# Return the most probable classes 
def get_classes(result, top_k=None, threshold=0.0):
    probs = _get_probs(result)
    pairs = [pair for pair in enumerate(probs) if pair[1] > threshold]
    pairs = sorted(pairs, key=lambda pair: pair[1], reverse=True)
    pairs = pairs[0:top_k]
    return [('/'.join(_CLASSES[index]), prob) for index, prob in pairs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', dest='input', required=True)
    args = parser.parse_args()

    with ImageInference(model()) as inference:
        image = Image.open(args.input)
        classes = get_classes(inference.run(image), top_k=5, threshold=0.1)
        for i, (label, score) in enumerate(classes):
            print('Result %d: %s (prob=%f)' % (i, label, score))


if __name__ == '__main__':
    main()
