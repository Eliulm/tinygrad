import os
import argparse
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from examples.mlperf.dataloader import batch_load_train_bert
from test.external.mlperf_bert.dataloader.reference_dataloader import input_fn_builder

def test_batch_load_train_bert(batch_size, max_seq_length, max_predictions_per_seq, input_files):
    your_input_fn = batch_load_train_bert(batch_size, 0)
    reference_input_fn = input_fn_builder(
        input_files,
        batch_size,
        max_seq_length,
        max_predictions_per_seq,
        is_training=True
    )

    your_batches = [next(your_input_fn) for _ in range(10)]
    reference_batches = [next(reference_input_fn({})) for _ in range(10)]

    for i, (your_batch, reference_batch) in tqdm(enumerate(zip(your_batches, reference_batches)), desc="Checking batches", total=len(your_batches)):
        assert len(your_batch) == len(reference_batch), f"Batch sizes do not match at index {i}"
        for key in reference_batch:
            assert key in your_batch, f"Key '{key}' not found in your batch at index {i}"
            assert your_batch[key].shape == reference_batch[key].shape, f"Shape mismatch for key '{key}' at index {i}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify the correctness of the dataloader",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_files", type=str, nargs='+', default=None,
                        help="Input files for the reference dataloader")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--max_predictions_per_seq", type=int, default=76, help="Max predictions per sequence")
    args = parser.parse_args()

    test_batch_load_train_bert(args.batch_size, args.max_seq_length, args.max_predictions_per_seq, args.input_files)
    print("Test passed!")
