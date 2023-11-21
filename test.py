import logging
import pandas as pd
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

model_args = Seq2SeqArgs()
model_args.do_sample = False
model_args.fp16 = False
model_args.num_beams = None
model_args.overwrite_output_dir = True
model_args.reprocess_input_data = True
model_args.use_multiprocessing = False
model_args.learning_rate = 5e-5
model_args.max_length = 20
model_args.max_seq_length = 20
model_args.num_return_sequences = 1
model_args.num_train_epochs = 1
model_args.top_k = 50
model_args.top_p = 0.95
model_args.train_batch_size = 1
model_args.dataloader_num_workers = 2
model_args.process_count = 2
#model_args.reprocess_input_data = True

model_reloaded = Seq2SeqModel(
    encoder_decoder_type="bart",
    encoder_decoder_name="output/", # Location of the trained model
    args=model_args,
    use_cuda=False,
)

# Test model with a simple sentence
print(
    model_reloaded.predict(
        [
            "Tyson is a Cyclops, a son of Poseidon, and Percy Jackson’s half brother. He is the current general of the Cyclopes army."
        ]
    )
)