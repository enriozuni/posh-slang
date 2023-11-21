import logging
import pandas as pd
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)

slang_text = pd.read_csv("train_data/slang.txt", sep = '\n', header = None, names = ["input_text"])
posh_text = pd.read_csv("train_data/posh.txt", sep ='\n', header = None, names = ["target_text"])
train_data = pd.merge(slang_text, posh_text, left_index=True, right_index=True)

model_args = Seq2SeqArgs()
model_args.do_sample = True
model_args.fp16 = False
model_args.do_lower_case = True
model_args.use_multiprocessing = False
model_args.overwrite_output_dir = True
model_args.reprocess_input_data = True
model_args.num_beams = None
model_args.learning_rate = 5e-5
model_args.max_length = 30
model_args.max_seq_length = 30
model_args.train_batch_size = 2
model_args.dataloader_num_workers = 2
model_args.process_count = 2
model_args.num_return_sequences = 1
model_args.num_train_epochs = 3
model_args.top_k = 50
model_args.top_p = 0.95

model = Seq2SeqModel(
    encoder_decoder_type="bart",
    encoder_decoder_name="facebook/bart-large",
    use_cuda=True,
    args=model_args
)

model.train_model(train_data)

