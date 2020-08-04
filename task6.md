# task6基于深度学习的文本分类

transformer原理

transformer是一个由编码器和解码器组成的模型，论文中编码器由留个编码器堆叠而成，解码器则由相同数量的解码器组成  
  
编码器结构完全相同，但是不共享网络参数，每个编码器可以拆解为两个部分。  
对输入序列进行序列化之后，序列首先流过一个self-attention层，这一层使得编码器能够看到其他输入序列的单词。  
self-attention的输出流向一个前向网络，每个前向网络互不干扰，最后输入下一个编码器。  
  
transformer中，每个单词仅仅流过它自己的编码器路径。在self-attention中，路径相互依赖，但进入前向网络后依赖取消。  
self-Attention使用多头机制，使得不同token但同形的词汇得到不同的转译序列  
基于预训练语言模型的词表示
基于预训练语言模型的词表示可以根据上下文的关系来为不同token同样type的词建立不停dembedding  
最早的对于此类问题所提出的模型所ELMO模型，ELMO用一个正向的RNN记录上下文联系给出词汇的embedding，  
再通过一个反向的RNN给出不同方向的embedding，结合两个embedding给出单词的最终表示

# BERT

bert模型的训练目标所利用大规模无标注语料训练、获得丰富的语义文本的方式。相当于预训练一个范用的模型，面对不同任务，  
通过微调既可以进行应用。  
在模型预训练阶段，bert通过利用上下文来预测缺失词汇，类似于人类的玩型填空，这使得在进行预测时可以利用来自不同方向的  
语料信息。 
在第二阶段，bert使用微调来进行下游任务的部署。再最后一层输出层只需针对不同问题添加不同网络结构即可得到相应任务解答  
如，要进行语句分类任务，只需要在句子中加入分隔符即可。  


```python
class WhitespaceTokenizer(object):
    """WhitespaceTokenizer with vocab."""
    def __init__(self, vocab_file):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def tokenize(self, text):
        split_tokens = whitespace_tokenize(text)
        output_tokens = []
        for token in split_tokens:
            if token in self.vocab:
                output_tokens.append(token)
            else:
                output_tokens.append("[UNK]")
        return output_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids)
```


```python
def create_segments_from_document(document, max_segment_length):
    """Split single document to segments according to max_segment_length."""
    assert len(document) == 1
    document = document[0]
    document_len = len(document)

    index = list(range(0, document_len, max_segment_length))
    other_len = document_len % max_segment_length
    if other_len > max_segment_length / 2:
        index.append(document_len)

    segments = []
    for i in range(len(index) - 1):
        segment = document[index[i]: index[i+1]]
        segments.append(segment)

    return segments
```


```python
(masked_lm_loss, masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
    bert_config, model.get_sequence_output(), model.get_embedding_table(),
    masked_lm_positions, masked_lm_ids, masked_lm_weights)

total_loss = masked_lm_loss
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-3-2b60f3a08f45> in <module>
    ----> 1 (masked_lm_loss, masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
          2     bert_config, model.get_sequence_output(), model.get_embedding_table(),
          3     masked_lm_positions, masked_lm_ids, masked_lm_weights)
          4 
          5 total_loss = masked_lm_loss


    NameError: name 'get_masked_lm_output' is not defined



```python
def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):
    # Initialise PyTorch model
    config = BertConfig.from_json_file(bert_config_file)
    print("Building PyTorch model from configuration: {}".format(str(config)))
    model = BertForPreTraining(config)

    # Load weights from tf checkpoint
    load_tf_weights_in_bert(model, config, tf_checkpoint_path)

    # Save pytorch-model
    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)
```


```python
wget https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
```


      File "<ipython-input-5-d9398247854c>", line 1
        wget https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
                 ^
    SyntaxError: invalid syntax




```python
export BERT_BASE_DIR=//mnt/NAS/数据处理在这个文件夹进行！！！/XXH/to/bert/chinese_L-12_H-768_A-12
```


      File "<ipython-input-6-ff80a27105b3>", line 1
        export BERT_BASE_DIR=//mnt/NAS/数据处理在这个文件夹进行！！！/XXH/to/bert/chinese_L-12_H-768_A-12
                           ^
    SyntaxError: invalid syntax




```python

```
