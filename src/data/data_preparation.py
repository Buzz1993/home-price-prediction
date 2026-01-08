import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import logging
from pathlib import Path

TARGET = "price"   # <-- change if your target column is different

# =====================================================
# LOGGER SETUP
# =====================================================
logger = logging.getLogger("data_preparation")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(handler)


# =====================================================
# FUNCTIONS
# =====================================================
def load_data(data_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded from {data_path} | Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error("The file to load does not exist")
        raise


def split_data(data: pd.DataFrame, test_size: float, random_state: int):
    train_data, test_data = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state
    )
    return train_data, test_data


def read_params(file_path: Path) -> dict:
    with open(file_path, "r") as f:
        params_file = yaml.safe_load(f)
    return params_file


def save_data(data: pd.DataFrame, save_path: Path) -> None:
    data.to_csv(save_path, index=False)


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":

    # root path
    root_path = Path(__file__).parent.parent.parent

    # ---------------- Paths ----------------
    data_path = root_path / "data" / "cleaned" / "property_cleaned.csv"

    save_data_dir = root_path / "data" / "interim"
    save_data_dir.mkdir(parents=True, exist_ok=True)

    train_filename = "train.csv"
    test_filename = "test.csv"

    save_train_path = save_data_dir / train_filename
    save_test_path = save_data_dir / test_filename

    params_file_path = root_path / "params.yaml"

    # ---------------- Load data ----------------
    df = load_data(data_path)

    # ---------------- Read params ----------------
    params = read_params(params_file_path)["Data_Preparation"]
    test_size = params["test_size"]
    random_state = params["random_state"]

    logger.info("Parameters read successfully")

    # ---------------- Split data ----------------
    train_data, test_data = split_data(
        df,
        test_size=test_size,
        random_state=random_state
    )

    logger.info(
        f"Dataset split | Train: {train_data.shape}, Test: {test_data.shape}"
    )

    # ---------------- Save data ----------------
    save_data(train_data, save_train_path)
    logger.info("Train data saved")

    save_data(test_data, save_test_path)
    logger.info("Test data saved")
