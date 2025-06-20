import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="ai_job_dataset.csv")
    args = parser.parse_args()

    # Ganti hardcoded path dengan:
    X_train, X_test, y_train, y_test, _, _ = load_and_preprocess(args.dataset_path)
    ...