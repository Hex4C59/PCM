import pandas as pd

def split_set(label_path: str):
    df = pd.read_csv(label_path, encoding='utf-8', sep='\t')
    
    df["session"] = df['name'].str[:5]

    split_map = {
        "Ses01": "train",
        "Ses02": "train",
        "Ses03": "train",
        "Ses04": "val",
        "Ses05": "test"
    }
    
    df["split"] = df["session"].map(split_map)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    train_df.drop(columns=["session", "split"]).to_csv("data/labels/train.csv", encoding="utf-8", sep="\t", index=False)
    val_df.drop(columns=["session", "split"]).to_csv("data/labels/val.csv", encoding="utf-8", sep='\t', index=False)
    test_df.drop(columns=["session","split"]).to_csv("data/labels/test.csv", encoding="utf-8", sep="\t", index=False)

if __name__ == "__main__":
    input_dir = "data/labels/iemocap.csv"
    split_set(input_dir)