import argparse
import os

from scripts._create_ciphertext_train import create_ciphertext_train
from scripts._create_ciphertexts_test import create_ciphertext_test
from scripts._create_keys_test import create_keys_test
from scripts._create_keys_train import create_keys_train
from scripts._create_npy_data_test import create_npy_data_test
from scripts._create_npy_data_train import create_npy_data_train


def main() -> None:
    """Create the complete dataset used for training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", help="path where the created data will be stored")
    parser.add_argument("-c", "--cpu_cores", help="amount of cores that should be used", type=int, default=os.cpu_count())
    args = parser.parse_args()

    create_keys_train(cpu_cores=args.cpu_cores, output_path=args.output_path)

    create_ciphertext_train(cpu_cores=args.cpu_cores, output_path=args.output_path)

    create_npy_data_train(cpu_cores=args.cpu_cores, output_path=args.output_path)

    create_keys_test(cpu_cores=args.cpu_cores, output_path=args.output_path)

    create_ciphertext_test(cpu_cores=args.cpu_cores, output_path=args.output_path)

    create_npy_data_test(cpu_cores=args.cpu_cores, output_path=args.output_path)


if __name__ == "__main__":
    main()
