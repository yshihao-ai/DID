import argparse
import time
from pathlib import Path

import numpy as np
import pickle
from tqdm import tqdm

import vocab_utils


class Event2tuneidx:
    def __init__(
        self,
        dataset: str,
        encoding_scheme: str,
        num_features: int,
        in_dir: Path,
        out_dir: Path,
        debug: bool,
    ):
        self.dataset = dataset
        self.encoding_scheme = encoding_scheme
        self.encoding_name = encoding_scheme + str(num_features)
        self.in_dir = in_dir / f"events_{self.dataset}" / self.encoding_name
        self.out_dir = out_dir / f"tuneidx_{self.dataset}" / self.encoding_name
        self.debug = debug

        vocab_name = {
            "remi": "LangTokenVocab",
            "cp": "MusicTokenVocabCP",
            "nb": "MusicTokenVocabNB",
            "did": "MusicTokenVocabDID",
        }
        selected_vocab_name = vocab_name[encoding_scheme]
        in_vocab_file_path = Path(f"../vocab/vocab_{dataset}/vocab_{dataset}_{encoding_scheme}{num_features}.json")
        self.vocab = getattr(vocab_utils, selected_vocab_name)(
            in_vocab_file_path=in_vocab_file_path,
            event_data=None,
            encoding_scheme=encoding_scheme,
            num_features=num_features,
        )

    def _convert_event_to_tune_in_idx(self, tune_in_event):
        tune_in_idx = []
        if self.encoding_scheme == "did":
            for event in tune_in_event:
                # DID now uses only note token sequence (chord prefix is already in notes_event).
                if len(event) == 2:
                    pos_event, notes_event = event[0], event[1]
                else:
                    raise ValueError(f"Event {event} has not 2 components. Expected 2 components.")

                pos_idx = self.vocab(pos_event)
                all_instruments_idx = self.vocab.get_all_instruments_per_timestep(pos_event)

                notes_idx = []
                for note_event in notes_event:
                    notes_idx.append(self.vocab(note_event))

                if pos_idx is not None and notes_idx is not None:
                    tune_in_idx.append((pos_idx, all_instruments_idx, notes_idx))
                else:
                    raise ValueError(f"Event {event} has None pos_idx or notes_idx.")

            if tune_in_idx is not None and len(tune_in_idx) > 0:
                note_feature_len = len(tune_in_idx[0][2][0])
                padding_note_token = [self.vocab.event2idx["type"]["STO"]] + [0] * (note_feature_len - 1)
                notes_max_length = max(len(event[2]) for event in tune_in_idx)

                padding_instrument_token = 0
                instruments_max_length = max(len(event[1]) for event in tune_in_idx)

                pos_in_idx = []
                padded_instruments_in_idx = []
                padded_notes_in_idx = []

                for pos_idx, instrument_list, note_list in tune_in_idx:
                    padded_notes = note_list + [padding_note_token] * (notes_max_length - len(note_list))
                    padded_instruments = instrument_list + [padding_instrument_token] * (
                        instruments_max_length - len(instrument_list)
                    )
                    padded_notes_in_idx.append(padded_notes)
                    padded_instruments_in_idx.append(padded_instruments)
                    pos_in_idx.append(pos_idx)

                return pos_in_idx, padded_instruments_in_idx, padded_notes_in_idx
            else:
                raise ValueError("Tune_in_idx is empty.")

        for event in tune_in_event:
            event_in_idx = self.vocab(event)
            if event_in_idx is not None:
                tune_in_idx.append(event_in_idx)
        return tune_in_idx

    def _load_single_event_and_make_tune_in_idx(self, file_path):
        with open(file_path, "rb") as f:
            tune_in_event = pickle.load(f)
        tune_in_idx = self._convert_event_to_tune_in_idx(tune_in_event)
        return file_path.name, tune_in_idx

    def make_tune_in_idx(self):
        print("preprocessing events data to tune_in_idx data")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        start_time = time.time()
        event_list = sorted(list(self.in_dir.rglob("*.pkl")))

        for filepath_name, tune_in_idx in tqdm(
            map(self._load_single_event_and_make_tune_in_idx, event_list), total=len(event_list)
        ):
            if tune_in_idx is None:
                continue

            if self.encoding_scheme == "remi":
                tune_in_idx = np.array(tune_in_idx, dtype=np.int16)
            elif self.encoding_scheme == "did":
                pos_in_idx, padded_instruments_in_idx, padded_notes_in_idx = tune_in_idx
                if pos_in_idx is None:
                    raise ValueError("pos_in_idx is None.")
                if padded_notes_in_idx is None:
                    raise ValueError("padded_notes_in_idx is None.")
                if padded_instruments_in_idx is None:
                    raise ValueError("padded_instruments_in_idx is None.")
                pos_in_idx = np.array(pos_in_idx, dtype=np.int16)
                padded_notes_in_idx = np.array(padded_notes_in_idx, dtype=np.int16)
                padded_instruments_in_idx = np.array(padded_instruments_in_idx, dtype=np.int16)
            else:
                tune_in_idx = np.array(tune_in_idx, dtype=np.int16)
                if np.max(tune_in_idx) < 256:
                    tune_in_idx = np.array(tune_in_idx, dtype=np.uint8)

            file_name = filepath_name.replace(".pkl", ".npz")
            if self.encoding_scheme == "did":
                np.savez_compressed(
                    self.out_dir / file_name,
                    pos=pos_in_idx,
                    notes=padded_notes_in_idx,
                    instruments=padded_instruments_in_idx,
                )
                del pos_in_idx
                del padded_notes_in_idx
                del padded_instruments_in_idx
            else:
                np.savez_compressed(self.out_dir / file_name, tune_in_idx)
                del tune_in_idx

        print(f"taken time for making tune_in_idx is {time.time() - start_time}")


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        choices=("BachChorale", "Pop1k7", "Pop909", "SOD", "LakhClean", "SymphonyMIDI", "EMOPIA+", "PIAST"),
        type=str,
        help="dataset names",
    )
    parser.add_argument(
        "-e",
        "--encoding",
        required=True,
        choices=("remi", "cp", "nb", "did"),
        type=str,
        help="encoding scheme",
    )
    parser.add_argument(
        "-f",
        "--num_features",
        required=True,
        choices=(4, 5, 7, 8),
        type=int,
        help="number of features",
    )
    parser.add_argument(
        "-i",
        "--in_dir",
        default="../dataset/represented_data/events/",
        type=Path,
        help="input data directory",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default="../dataset/represented_data/tuneidx/",
        type=Path,
        help="output data directory",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="enable debug mode",
    )
    return parser


def main():
    parser = get_argument_parser()
    args = parser.parse_args()

    event2tuneidx = Event2tuneidx(args.dataset, args.encoding, args.num_features, args.in_dir, args.out_dir, args.debug)
    event2tuneidx.make_tune_in_idx()


if __name__ == "__main__":
    main()
