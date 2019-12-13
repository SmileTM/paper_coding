import tensorflow as tf
import json
import data_process
import requests


units = 1024
# encoder_ref，decoder_ref更具自己的docker 进行配置
encoder_ref = 'http://localhost:8501/v1/models/encoder:predict'
decoder_ref = 'http://localhost:8502/v1/models/decoder:predict'

input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer = data_process.load_dataset('cmn.txt')
max_length_inp, max_length_tar = data_process.max_length(input_tensor), data_process.max_length(target_tensor)

'''
其中值得注意的 是 instance处，[]里面是每个实例。 也就是说[]里面的 每一个大括号对应一个预测样本，因此 
输入的 维度不包括example_size,
如 有2个30维的图片，正常的表示为 （2 30 30 3）， 在传入时，就应转换成 "instance":[{"input":( 30 30 3)},{"input":( 30 30 3)}]
每个大括号代表一个样例


instance 中的 input_1, input_2, 以及output_1 output_2,是根具model的输入 输出 顺序来的。
如果不清楚可以 在加载模型后 通过 encoder.inputs  encoder.outputs 来进行查看
在终端中 输入以下命令 将模型部署 映射到8501端口
docker run -p 8501:8501 --name encoder --mount source=path/encoder_zh,type=bind,target=/models/encoder -e MODEL_NAME=encoder -t tensorflow/serving
encoder 的rest_client
'''


def encoder_rest(input, hidden):
    data = json.dumps({"instances": [{"input_1": input.numpy().tolist(), "input_2": hidden.numpy().tolist()}]})
    json_response = requests.post(encoder_ref, data=data)

    predictions = json.loads(json_response.text)['predictions']

    en_output = predictions[0]['output_1']
    en_state = predictions[0]['output_2']
    return en_state, en_output


# 在终端中 输入以下命令 将模型部署 映射到8502端口
# docker run -p 8502:8501 --name decoder --mount source=path/decoder_zh,type=bind,target=/models/decoder -e MODEL_NAME=decoder -t tensorflow/serving
# decoder 的rest_client
def decoder_rest(x, en_hidden, en_output):
    data = json.dumps({"instances": [{"input_1": x.numpy().tolist(), "input_2": en_hidden, "input_3": en_output}]})
    json_response = requests.post(decoder_ref, data=data)
    predictions = json.loads(json_response.text)['predictions']
    x = predictions[0]['output_1']
    state = predictions[0]['output_2']
    attention_weights = predictions[0]['output_3']
    return x, state, attention_weights


# 进行翻译， 并返回结果
def translate(sentence):
    sentence = data_process.preprocess_sentence_zh(sentence)
    inputs = [inp_lang_tokenizer.word_index[i] for i in sentence.split()]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    result = ''
    hidden = tf.zeros((1, units))
    en_state, en_out = encoder_rest(inputs[0], hidden[0])
    de_inputs = tf.expand_dims([targ_lang_tokenizer.word_index['<start>']], 0)
    de_input = de_inputs[0]
    decoder_rest(de_input, en_state, en_out)
    de_hideen = en_state

    result = ''
    for t in range(max_length_tar):
        x, state, _ = decoder_rest(de_input, de_hideen, en_out)
        prediction_id = tf.argmax(x, axis=-1).numpy()
        if targ_lang_tokenizer.index_word[prediction_id] == '<end>':
            break
        result += targ_lang_tokenizer.index_word[prediction_id] + ' '
        de_input = tf.expand_dims([prediction_id], 0)[0]
        de_hideen = state

    return result


sentence = '我喜欢你！'
print(sentence)
print(translate(sentence))
