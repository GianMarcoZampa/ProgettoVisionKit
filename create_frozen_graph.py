import tensorflow as tf
import tensorflow.keras as ks

MODEL_H5_PATH = "model_datagen_v1.h5"
MODEL_TXT_PATH = "frozengraph_datagen_v1.pbtxt"
MODEL_PATH = "frozengraph_datagen_v1.pb"


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


# Loading .h5 model and save it as SavedModel
model = tf.keras.models.load_model(MODEL_H5_PATH)
tf.saved_model.save(model, "models/")

# Print input and output layers info
print(model.layers[0].name, model.layers[-1].name)
print(model.inputs)
print(model.outputs)

frozen_graph = freeze_session(ks.backend.get_session(),
                              output_names=[out.op.name for out in model.outputs])
tf.train.write_graph(frozen_graph, logdir="graphs", name=MODEL_PATH, as_text=False)
tf.train.write_graph(frozen_graph, logdir="graphs", name=MODEL_TXT_PATH, as_text=True)
